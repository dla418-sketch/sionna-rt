#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Stores shooting and bouncing of rays samples data"""

import mitsuba as mi
import drjit as dr
from dataclasses import dataclass
from typing import Tuple
from sionna.rt.constants import InteractionType
from sionna.rt.utils import WedgeGeometry


class SampleData:
    r"""
    Class used to store shoot-and-bounce samples data

    A sample is a path spawn from a source when running shooting and
    bouncing of rays. However, to distinguish these paths from the ones stored
    in a :class:`~sionna.rt.PathsBuffer`, which are further processed and/or
    returned by the solver, the paths spawn during shooting and bouncing of rays
    are referred to as "samples".

    Each sample can lead to the recording in a :class:`~sionna.rt.PathsBuffer`
    of one or more paths.

    :param num_sources: Number of sources
    :param samples_per_src: Number of samples spawn per source
    :param max_depth: Maximum depth
    """

    @dataclass
    class SampleDataFields:
        r"""
        Dataclass used to store information of a single sample and for a single
        interaction

        :data interaction_type: Type of interaction
        :data shape: Pointer to the intersected shape as an unsigned integer
        :data primitive: Index of the intersected primitive
        :data vertex: Coordinates of the intersection point with the scene
        """

        interaction_type    : mi.UInt
        shape               : mi.UInt
        primitive           : mi.UInt
        vertex              : mi.Point3f
        prob                : mi.Float

    def __init__(self,
                 num_sources: int,
                 samples_per_src: int,
                 max_depth: int,
                 diffraction: bool):

        # Size of the array
        # If `max_depth` is 0, we need to allocate at least one element to
        # store the LoS path data
        array_size = max(1, max_depth)

        # Index of the source corresponding to samples.
        # Samples-first ordering is used, i.e., the samples are ordered as
        # follows:
        # [source_0_samples..., source_1_samples..., ...]
        src_indices = dr.arange(mi.UInt, 0, num_sources)
        src_indices = dr.repeat(src_indices, samples_per_src)
        self._src_indices = src_indices

        # Structure storing the data for a single sample.
        # A DrJit local memory is used to enable read-after-write dependencies.
        # That implies that a buffer is created for each thread, which ray trace
        # a single sample, to store information about this sample.
        # The size of the buffer is set to `max_depth`, as a sample can consists
        # of up to that many interactions with the scene.
        # See https://drjit.readthedocs.io/en/latest/misc.html#local-memory
        self._local_mem = dr.alloc_local(SampleData.SampleDataFields,
                                         array_size)

        # Diffracting wedge geometry. As diffraction is only supported for first
        # order, we do not need to store the wedge geometry for each depth.
        self._diffraction = diffraction
        if diffraction:
            self._diffracting_wedges = dr.alloc_local(WedgeGeometry, 1)

    def insert(self,
               depth: mi.UInt,
               interaction_types: mi.UInt,
               shapes: mi.ShapePtr,
               primitives: mi.UInt,
               diffracting_wedges: WedgeGeometry,
               vertices: mi.Point3f,
               probs: mi.Float,
               active: mi.Mask):
        # pylint: disable=line-too-long
        r"""
        Stores interaction data for depth ``depth``

        :param depth: Depth for which to store sample data
        :param interaction_types: Type of interaction represented using :class:`~sionna.rt.constants.InteractionType`
        :param shapes: Pointers to the intersected shapes
        :param primitives: Indices of the intersected primitives
        :param wedge_geometry: Diffracting wedge geometry
        :param vertices: Coordinates of the intersection points
        :param probs: Probabilities of the sampled interaction types
        :param active: Mask of active samples
        """
        # Depth ranges from 1 to max_depth
        index = depth - 1

        # Shape pointers are converted to unsigned integers
        shapes = dr.reinterpret_array(mi.UInt, shapes)

        # Store data in the buffer
        data = SampleData.SampleDataFields(interaction_types, shapes,
                                           primitives, vertices, probs)
        self._local_mem.write(data, index, active=active)

        # Store the local edge index and edge properties
        if self._diffraction:
            diffraction = interaction_types == InteractionType.DIFFRACTION
            self._diffracting_wedges[0] = dr.select(
                diffraction,
                diffracting_wedges,
                self._diffracting_wedges[0]
            )

    def get(self,
            depth: mi.UInt,
            active: mi.Mask
        ) -> Tuple[mi.UInt, mi.UInt, mi.UInt, mi.Point3f, mi.Float]:
        # pylint: disable=line-too-long
        r"""
        Returns data about the sample for depth ``depth``

        :param depth: Depths for which to return sample data
        :param active: Mask of active samples

        :return: Type of interaction represented using :class:`~sionna.rt.constants.InteractionType`
        :return: Pointers to the intersected shapes
        :return: Indices of the intersected primitives
        :return: Local indices of the diffracting edges
        :return: Coordinates of the intersection points
        :return: Probabilities of the sampled interaction types
        """
        index = depth - 1
        data = self._local_mem.read(index, active=active)

        interaction_types = data.interaction_type
        shapes = data.shape
        primitives = data.primitive
        vertices = data.vertex
        probs = data.prob

        return interaction_types, shapes, primitives, vertices, probs

    @property
    def diffracting_wedges(self):
        r"""Diffracting wedge geometry

        :type: :py:class:`WedgeGeometry`
        """
        if self._diffraction:
            return self._diffracting_wedges[0]
        return None

    @property
    def src_indices(self):
        r"""Indices of the sources from which samples originate

        :type: :py:class:`mi.UInt`
        """
        return self._src_indices
