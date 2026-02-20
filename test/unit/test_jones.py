#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import mitsuba as mi
import numpy as np
from scipy.spatial.transform import Rotation as scipy_rotation

from sionna.rt.utils import jones_matrix_rotator, to_world_jones_rotator,\
    jones_vec_dot, implicit_basis_vector, jones_matrix_rotator_flip_forward

#############################################################
# Constants
#############################################################

# Threshold for the relative squared error above which a test fails
MAX_RSE = 1e-8

#############################################################
# Utilities
#############################################################

def batch_matvec(m, v):
    """
    Batchified matvec op
    """

    return (m@np.expand_dims(v, axis=-1))[:,:,0]

def vec_mi_to_np(v):
    """
    Cast a mi.Vector3f to an equivalent numpy array
    """
    return np.transpose(v.numpy(), [1,0])

def vec_np_to_mi(v):
    """
    Cast a numpy array with shape [batch_size, 3] to a mi.Vector3f
    """
    return mi.Vector3f(v[:,0], v[:,1], v[:,2])

def mat_mi_to_np(m):
    """
    Casts a JonesMatrix object to an equivalent complex-valued numpy array
    """
    a = np.transpose(m.numpy(), [2, 0, 1])
    return a

def max_rel_se(u, v, axes=-1):
    """
    Computes the relative max squared error (SE) between `u` and `v` by reducing along `axes`.
    `u` is the reference value.
    """

    if axes is None:
        rse = np.square(np.abs(u-v)) / np.square(np.abs(u))
    else:
        rse = np.sum(np.square(np.abs(u-v)), axis=axes) / np.sum(np.square(np.abs(u)), axis=axes)
    return np.max(rse)

#############################################################
# Tests
#############################################################

def test_jones_matrix_rotator():
    r"""
    Tests the `jones_matrix_rotator()` utility
    """

    batch_size = 100

    # Random vector
    u = np.random.normal(size=[batch_size,3])

    # Current basis
    s1 = np.random.normal(size=[batch_size,3])
    s1 = s1/np.linalg.norm(s1, axis=1, keepdims=True)

    # Forward direction
    fwd = np.cross(u,s1)
    fwd = fwd/np.linalg.norm(fwd, axis=1, keepdims=True)

    # Target basis (must be orthogonal to forward)
    s2 = np.random.normal(size=[batch_size,3])
    s2 -= np.sum(s2*fwd, axis=1, keepdims=True)*fwd
    s2 = s2/np.linalg.norm(s2, axis=1, keepdims=True)

    # p1 and p2 components
    p1 = np.cross(fwd,s1)
    p2 = np.cross(fwd,s2)

    # u in the (s1, p1, k)
    # Drop the k-dim as it is 0
    u_in = np.stack([np.sum(u*s1, axis=1),
                    np.sum(u*p1, axis=1)],
                    axis=-1)
    # u in the (s2, p2, k) basis
    # Drop the k-dim as it is 0
    u_out_ref = np.stack([np.sum(u*s2, axis=1),
                        np.sum(u*p2, axis=1)],
                        axis=-1)

    # Compute change-of-basis matrix
    fwd_mi = vec_np_to_mi(fwd)
    s1_mi = vec_np_to_mi(s1)
    s2_mi = vec_np_to_mi(s2)
    P = jones_matrix_rotator(fwd_mi, s1_mi, s2_mi)
    P = mat_mi_to_np(P)

    # Apply the change of basis
    u_out = batch_matvec(P, u_in)

    assert max_rel_se(u_out_ref, u_out, axes=(-1,)) < MAX_RSE

def test_to_world_jones_rotator():
    r"""
    Tests `to_world_jones_rotator()`
    """

    batch_size = 100

    np.random.seed(42)

    # Generate random to-world transform
    to_world_angles = np.random.uniform(low=0., high=0.5*np.pi, size=[batch_size,3])
    to_world = scipy_rotation.from_euler('xyz', to_world_angles).as_matrix()
    to_world_mi = mi.Matrix3f(np.transpose(to_world, [1, 2, 0]))

    # Generate random local forward direction
    k_local = np.random.normal(size=[batch_size, 3])
    k_local = k_local / np.linalg.norm(k_local, axis=-1, keepdims=True)
    k_local_mi = vec_np_to_mi(k_local)
    #
    k_world = batch_matvec(to_world, k_local)
    k_world_mi = vec_np_to_mi(k_world)

    # Compute implicit S and P basis vector for the local frame
    s_local_mi = implicit_basis_vector(k_local_mi)
    s_local = vec_mi_to_np(s_local_mi)
    p_local = np.cross(k_local, s_local)

    # Compute implicit S and P basis vector for the world frame
    s_world_mi = implicit_basis_vector(k_world_mi)
    s_world = vec_mi_to_np(s_world_mi)
    p_world = np.cross(k_world, s_world)

    # Generate random local Jones vectors
    jones_local = np.random.random(size=[batch_size, 2])
    # 3D field vector in local frame
    field_vec_local = np.concatenate([jones_local,
                                    np.zeros([batch_size, 1])], axis=1)
    field_vec_local = field_vec_local[:,:1]*s_local + field_vec_local[:,1:2]*p_local
    # Field vector in world frame
    field_vec_world = batch_matvec(to_world, field_vec_local)

    # Compute the rotator using the function to test, and rotates the Jones vector
    rotator_mi = to_world_jones_rotator(to_world_mi, k_local_mi)
    rotator = mat_mi_to_np(rotator_mi)
    jones_world = batch_matvec(rotator, jones_local)

    # Test that `jones_world` is accurate
    ref_jones_world_x = np.sum(field_vec_world*s_world, axis=-1)
    assert max_rel_se(ref_jones_world_x, jones_world[:,0]) < MAX_RSE
    ref_jones_world_y = np.sum(field_vec_world*p_world, axis=-1)
    assert max_rel_se(ref_jones_world_y, jones_world[:,1]) < MAX_RSE

def test_jones_vec_dot():
    """Test `jones_vec_dot()` against numpy"""

    def batch_dot_product(vectors_a, vectors_b):
        # Batch dot product using np.einsum
        return np.einsum('ij,ij->i', np.conj(vectors_a), vectors_b)

    np.random.seed(42)
    batch_size = 100

    u = np.random.normal(size=[batch_size, 4])
    v = np.random.normal(size=[batch_size, 4])

    u_np = u[:,:2] + 1j*u[:,2:]
    v_np = v[:,:2] + 1j*v[:,2:]

    u_mi = mi.Vector4f(u[:,0], u[:,1], u[:,2], u[:,3])
    v_mi = mi.Vector4f(v[:,0], v[:,1], v[:,2], v[:,3])

    p_mi = jones_vec_dot(u_mi, v_mi)
    p_np = p_mi.numpy()
    ref_p = batch_dot_product(u_np,v_np)
    assert max_rel_se(ref_p, p_np, axes=None) < MAX_RSE

def test_jones_matrix_rotator_flip_forward():
    r"""Test `jones_matrix_rotator_flip_forward()`"""

    batch_size = 100

    np.random.seed(42)

    # Forward direction
    k = np.random.normal(size=[batch_size, 3])
    k = k / np.linalg.norm(k, axis=-1, keepdims=True)
    k_mi = vec_np_to_mi(k)

    # Basis for the current forward direction
    s = implicit_basis_vector(k_mi).numpy().T
    p = np.cross(k, s)

    # Basis for the flipped forward direction
    sf = implicit_basis_vector(-k_mi).numpy().T
    pf = np.cross(-k, sf)

    # Rotator that flip the forward direction
    flip_rotator = np.transpose(jones_matrix_rotator_flip_forward(k_mi).numpy(),
                                [2,0,1])

    # Generate random Jones vector, assumed to be represented
    # in the non-flipped basis
    j_real = np.random.normal(size=[batch_size, 2])
    j_imag = np.random.normal(size=[batch_size, 2])
    j = j_real + 1j*j_imag

    # Rotate to the basis with flipped fwd
    jf = batch_matvec(flip_rotator, j)

    # `j` and `jf` should correspond to the same vector
    v1 = j[:,:1]*s + j[:,1:]*p
    v2 = jf[:,:1]*sf + jf[:,1:]*pf
    assert max_rel_se(v1, v2) < MAX_RSE
