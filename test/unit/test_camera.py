#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import os

import drjit as dr
import mitsuba as mi
import numpy as np

from sionna import rt
from sionna.rt import Camera


def test01_camera_create():
    def check(camera, expected):
        # print('---')
        # print(camera.world_transform.matrix.numpy())
        # print(np.array(expected))
        assert np.allclose(camera.world_transform.matrix.numpy()[:,:,0],
                           expected, atol=1e-5)

    c = Camera(position=[0, 3, 0.5])
    check(c, [
        [0, 0, 1, 0],
        [1, 0, 0, 3],
        [0, 1, 0, 0.5],
        [0, 0, 0, 1],
    ])

    c = Camera(position=[0, 3, 0.5], orientation=(np.pi, 0, 0))
    check(c, [
        [ 0, 0, -1, 0],
        [-1, 0,  0, 3],
        [ 0, 1,  0, 0.5],
        [ 0, 0,  0, 1],
    ])

    c = Camera(position=[0, 0, 3], look_at=[0, 0, 10])
    check(c, [
        [0,      -1, 1.4e-4, 0],
        [1,       0,      0, 0],
        [0,  1.4e-4,      1, 3],
        [0,       0,      0, 1],
    ])

    c.look_at([0, 10, 0])
    expected = mi.ScalarTransform4f().look_at(origin=[0, 0, 3],
                                        target=[0, 10, 0],
                                        up=[0, 0, 1]).matrix.numpy()
    check(c, expected)

    c.position = np.array([1, 2, 3])
    expected[:3, 3] = [1, 2, 3]
    check(c, expected)
