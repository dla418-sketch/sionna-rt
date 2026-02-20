#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Utilities for ray tracing"""

import drjit as dr
import mitsuba as mi
from typing import Callable

from sionna.rt.constants import EPSILON_FLOAT
from sionna.rt.utils.geometry import rotate_vector_around_axis,\
    theta_phi_from_unit_vec
from sionna.rt.utils.wedges import wedge_interior_angle

def fibonacci_lattice(num_points: int) -> mi.Point2f:
    r"""
    Generates a Fibonacci lattice of size ``num_points`` on the unit square
    :math:`[0, 1] \times [0, 1]`

    :param num_points: Size of the lattice
    """

    golden_ratio = (1.+dr.sqrt(mi.Float64(5.)))/2.
    ns = dr.arange(mi.Float64, 0, num_points)

    x = ns/golden_ratio
    x = x - dr.floor(x)
    y = ns/(num_points-1)

    return mi.Point2f(x, y)

def spawn_ray_from_sources(
    lattice: Callable[[int], mi.Point2f],
    samples_per_src: int,
    src_positions: mi.Point3f
) -> mi.Ray3f:
    r"""
    Spawns ``samples_per_src`` rays for each source at the positions specified
    by ``src_positions``, oriented in the directions defined by the ``lattice``

    The spawned rays are ordered samples-first.

    :param lattice: Callable that generates the lattice used as directions for
        the rays
    :param samples_per_src: Number of rays per source to spawn
    :param src_positions: Positions of the sources
    """

    num_sources = dr.shape(src_positions)[1]

    # Ray directions
    samples_on_square = lattice(samples_per_src)
    k_world = mi.warp.square_to_uniform_sphere(samples_on_square)

    # Samples-first ordering is used, i.e., the samples are ordered as
    # follows:
    # [source_0_samples..., source_1_samples..., ...]
    # Each source has its own lattice
    k_world = dr.tile(k_world, num_sources)
    # Rays origins are the source locations
    origins = dr.repeat(src_positions, samples_per_src)

    # Spawn rays from the sources
    ray = mi.Ray3f(o=origins, d=k_world)
    # Minor workaround to avoid loop retracing due to size mismatch.
    ray.time = dr.zeros(mi.Float, dr.width(k_world))

    return ray

def offset_p(p: mi.Point3f, d: mi.Vector3f, n: mi.Vector3f) -> mi.Point3f:
    # pylint: disable=line-too-long
    r"""
    Adds a small offset to :math:`\mathbf{p}` along :math:`\mathbf{n}` such that
    :math:`\mathbf{n}^{\textsf{T}} \mathbf{d} \gt 0`

    More precisely, this function returns :math:`\mathbf{o}` such that:

    .. math::
        \mathbf{o} = \mathbf{p} + \epsilon\left(1 + \max{\left\{|p_x|,|p_y|,|p_z|\right\}}\right)\texttt{sign}(\mathbf{d} \cdot \mathbf{n})\mathbf{n}

    where :math:`\epsilon` depends on the numerical precision and :math:`\mathbf{p} = (p_x,p_y,p_z)`.

    :param p: Point to offset
    :param d: Direction toward which to offset along ``n``
    :param n: Direction along which to offset
    """

    a = (1. + dr.max(dr.abs(p), axis=0)) * EPSILON_FLOAT
    # Detach this operation to ensure these is no gradient computation
    a = dr.detach(dr.mulsign(a, dr.dot(d,n)))
    po = dr.fma(a, n, p)
    return po

def spawn_ray_towards(
    p: mi.Point3f,
    t: mi.Point3f,
    n: mi.Vector3f | None = None
) -> mi.Ray3f:
    r"""
    Spawns a ray with infinite length from :math:`\mathbf{p}` toward
    :math:`\mathbf{t}`

    If :math:`\mathbf{n}` is not :py:class:`None`, then a small offset is added
    to :math:`\mathbf{p}` along :math:`\mathbf{n}` in the direction of
    :math:`\mathbf{t}`.

    :param p: Origin of the ray
    :param t: Point towards which to spawn the ray
    :param n: (Optional) Direction along which to offset :math:`\mathbf{p}`
    """

    # Adds a small offset to `p` to avoid self-intersection
    if n is None:
        po = p
    else:
        po = offset_p(p, t - p, n)
    # Ray direction towards `t`
    d = dr.normalize(t - po)
    #
    ray = mi.Ray3f(po, d)
    # Small optimization to avoid loop retracing due to size mismatch.
    # This is still a literal.
    ray.time = dr.zeros(mi.Float, dr.width(ray.d))
    return ray

def spawn_ray_to(
    p: mi.Point3f,
    t: mi.Point3f,
    n: mi.Vector3f | None = None
) -> mi.Ray3f:
    r"""
    Spawns a finite ray from :math:`\mathbf{p}` to :math:`\mathbf{t}`

    The length of the ray is set to :math:`\|\mathbf{p} - \mathbf{t}\|`.

    If :math:`\mathbf{n}` is not :py:class:`None`, then a small offset is added
    to :math:`\mathbf{p}` along :math:`\mathbf{n}` in the direction of
    :math:`\mathbf{t}`.

    :param p: Origin of the ray
    :param t: Point towards which to spawn the ray
    :param n: (Optional) Direction along which to offset :math:`\mathbf{p}`
    """

    # Adds a small offset to `p`
    if n is None:
        po = p
    else:
        po = offset_p(p, t - p, n)
    # Ray direction towards `t`
    d = t - po
    maxt = dr.norm(d)
    d /= maxt
    maxt *= (1. - EPSILON_FLOAT)
    #
    ray = mi.Ray3f(po, d,  maxt=maxt, time=0., wavelengths=mi.Color0f())
    # Small optimization to avoid loop retracing due to size mismatch.
    # This is still a literal.
    ray.time = dr.zeros(mi.Float, dr.width(ray.d))
    return ray

def first_order_diffraction_point(s: mi.Point3f,
                                  t: mi.Point3f,
                                  o: mi.Point3f,
                                  zeta: mi.Vector3f) -> mi.Float:
    # pylint: disable=line-too-long
    r"""
    Computes the scalar the diffraction point on an edge for a
    single-order diffracted path connecting a source and a target

    This function determines the parameter :math:`x` that defines the position
    on an edge where diffraction occurs for a ray path from source
    :math:`\mathbf{s}` to target :math:`\mathbf{t}`. The edge is parameterized
    as :math:`\mathbf{o} + x \boldsymbol{\zeta}`, where :math:`\mathbf{o}` is
    the edge origin and :math:`\boldsymbol{\zeta}` is the normalized direction
    vector.


    :param s: Source position :math:`\mathbf{s}`
    :param t: Target position :math:`\mathbf{t}`
    :param o: Origin point of the edge :math:`\mathbf{o}`
    :param zeta: Normalized direction vector of the edge :math:`\boldsymbol{\zeta}`

    :return: Parameter :math:`x` defining the diffraction point position
        :math:`\mathbf{o} + x \boldsymbol{\zeta}` on the edge
    """

    ## Use edge origin as reference point
    to = t - o
    so = s - o

    ## Projection of the source and target on the edge
    sp = dr.dot(so, zeta)*zeta
    tp = dr.dot(to, zeta)*zeta

    ## Rotate the target around the edge to be coplanar
    ## with the source and the edge

    # Angle of rotation
    v1 = dr.normalize(to - tp)
    v2 = dr.normalize(so - sp)
    theta = dr.pi - dr.safe_acos(dr.dot(v1, v2))

    # Axis of rotation
    rot_axis = dr.cross(so - sp, to - tp)
    rot_axis_norm = dr.norm(rot_axis)
    rot_axis *= dr.rcp(rot_axis_norm)
    rot_axis = dr.select(rot_axis_norm > 0, rot_axis, zeta)

    # Apply the rotation
    tc = rotate_vector_around_axis(to, rot_axis, theta)

    ## Compute the diffraction point

    u0 = dr.normalize(tc - so)
    u1 = dr.cross(so, u0)
    u2 = dr.cross(zeta, u0)
    sign = dr.sign(dr.dot(u1, u2))

    # Position of the diffraction point on the edge
    # relative to the edge origin
    x = sign*dr.norm(u1)*dr.rcp(dr.norm(u2))

    return x

def sample_keller_cone(e_hat: mi.Vector3f,
                       n0: mi.Vector3f,
                       nn: mi.Vector3f,
                       sample: mi.Float,
                       ki: mi.Vector3f,
                       lit_region: bool) -> mi.Vector3f:
    r"""
    Samples a direction on the Keller cone for diffraction

    This function samples a diffracted ray direction on the Keller cone.
    The Keller cone is such that the angle between the diffracted rays and
    the edge direction is equal to the angle between the incident ray and
    the edge direction.

    :param e_hat: Normalized edge direction vector
    :param n0: Normal to the 0-face
    :param nn: Normal to the n-face
    :param sample: Random sample in [0, 1) used for sampling the Keller cone
    :param ki: Incident ray direction
    :param lit_region: If set to `True`, then the lit region is sampled.
        If set to `False`, then only the shadow region is sampled.

    :return: Normalized direction vector of the diffracted ray in the
        local coordinate system
    """

    # Local coordinate system (t0, n0, e_fwd), where
    # e_fwd is the edge direction oriented in the direction
    # of the incident ray
    t0 = dr.normalize(dr.cross(n0, e_hat))
    e_fwd = dr.select(dr.dot(e_hat, ki) > 0,
                      e_hat, -e_hat)

    # k_i in local coordinate system
    ki_local = mi.Vector3f(dr.dot(ki, t0),
                           dr.dot(ki, n0),
                           dr.dot(ki, e_fwd))

    beta0, phi_i = theta_phi_from_unit_vec(ki_local)

    # Read the wedge opening angle
    wedge_angle = wedge_interior_angle(n0, nn)

    # Sample an angle on the Keller cone
    if lit_region:
        phi = sample * (dr.two_pi - wedge_angle)
    else:
        # Ensures that phi_i is in [0, 2*pi]
        phi_i = dr.select(phi_i < 0, phi_i + dr.two_pi, phi_i)
        phi = phi_i + sample * (dr.two_pi - wedge_angle - phi_i)


    # Direction of propagation of diffracted ray
    sin_beta0, cos_beta0 = dr.sincos(beta0)
    sin_phi, cos_phi = dr.sincos(phi)
    k_diffr = sin_phi*sin_beta0*n0 \
        + cos_phi*sin_beta0*t0 \
        + cos_beta0*e_fwd

    return k_diffr
