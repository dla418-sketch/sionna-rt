#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import drjit as dr
import mitsuba as mi
import sionna
from sionna.rt import load_scene, SceneObject


def test_scene_object_clone():
    scene = load_scene(sionna.rt.scene.box_one_screen,merge_shapes=False)

    screen = scene.objects["screen"]
    screen_clone = screen.clone(name="my-screen-clone")

    assert isinstance(screen_clone, SceneObject)

    # Check the name is correctly set
    assert screen_clone.name == "my-screen-clone"

    # The object should not be in the scene
    assert screen_clone.name not in scene.objects

    # Same radio material
    assert screen_clone.radio_material is screen.radio_material

    # Identical but not shared geometry
    assert dr.all(screen_clone.mi_mesh.faces_buffer() == screen.mi_mesh.faces_buffer())
    assert dr.all(screen_clone.mi_mesh.vertex_positions_buffer() == screen.mi_mesh.vertex_positions_buffer())
    assert screen_clone.mi_mesh.faces_buffer() is not screen.mi_mesh.faces_buffer()
    assert screen_clone.mi_mesh.vertex_positions_buffer() is not screen.mi_mesh.vertex_positions_buffer()

    ##################
    # Test is_mesh
    ##################

    screen = scene.objects["screen"]
    screen_clone = screen.clone(name="my-screen-clone", as_mesh=True)

    assert isinstance(screen_clone, mi.Mesh)

    # Check the name is correctly set
    assert screen_clone.id() == "my-screen-clone"

    # Same radio material
    assert screen_clone.bsdf() is screen.radio_material

    # Identical but not shared geometry
    assert dr.all(screen_clone.faces_buffer() == screen.mi_mesh.faces_buffer())
    assert dr.all(screen_clone.vertex_positions_buffer() == screen.mi_mesh.vertex_positions_buffer())
    assert screen_clone.faces_buffer() is not screen.mi_mesh.faces_buffer()
    assert screen_clone.vertex_positions_buffer() is not screen.mi_mesh.vertex_positions_buffer()
