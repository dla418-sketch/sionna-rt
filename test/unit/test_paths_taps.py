#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import numpy as np

import sionna.rt
from sionna.rt import load_scene, Transmitter, Receiver, \
                      PlanarArray, PathSolver

@pytest.mark.parametrize('synthetic_array', [True, False])
@pytest.mark.parametrize('normalize_delays', [True, False])
@pytest.mark.parametrize('normalize', [True, False])
def test_taps_computation(synthetic_array, normalize_delays, normalize):
    """
    Test that channel taps are correctly computed under various settings
    """
    scene = load_scene(sionna.rt.scene.simple_reflector)
    scene.add(Transmitter("tx-1", [-10,0,10]))
    scene.add(Receiver("rx-1", [10,0,10]))
    scene.add(Transmitter("tx-2", [-10,1,10]))
    scene.add(Receiver("rx-2", [10,-1,10]))
    scene.tx_array = PlanarArray(num_cols=3, num_rows=3,
                            pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_cols=1, num_rows=2,
                            pattern="iso", polarization="V")
    scene.get("tx-1").velocity = [-10, 10, 10]
    scene.get("rx-2").velocity = [-5, 8, 6]
    p_solver = PathSolver()
    paths = p_solver(scene, los=True, specular_reflection=True,
                diffuse_reflection=False, refraction=False,
                diffraction=True, edge_diffraction=True,
                synthetic_array=synthetic_array)

    sampling_frequency = 10**5
    num_time_steps = 10
    bandwidth = 100e6
    l_min=-6
    l_max=100
    h_rt = paths.taps(
                bandwidth=bandwidth,
                l_min=l_min,
                l_max=l_max,
                sampling_frequency=sampling_frequency,
                num_time_steps=num_time_steps,
                normalize=normalize,
                normalize_delays=normalize_delays,
                out_type="numpy")

    # Get CIR from which taps will be computed
    a, tau = paths.cir(sampling_frequency=sampling_frequency,
                    num_time_steps=num_time_steps,
                    normalize_delays=normalize_delays,
                    out_type="numpy")

    # Compute taps
    t = np.arange(l_min, l_max+1)
    e = np.sinc(t-bandwidth*np.expand_dims(tau, -1))
    if synthetic_array:
        e = np.expand_dims(np.expand_dims(e, 1), 3)
    h_ref = np.sum(np.expand_dims(a, -1)*np.expand_dims(e, -2), -3)

    if normalize:
        norm = np.sqrt(np.mean(np.sum(np.abs(h_ref)**2, -1, keepdims=True),
                               axis=(1,3,4), keepdims=True))
        h_ref /= norm
        err = np.max(np.abs(h_rt-h_ref))
    else:
        err = np.max(np.abs(h_rt-h_ref)/np.max(np.abs(h_rt),
                                               axis=-1, keepdims=True))
    assert err < 1e-2
