#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import numpy as np

import sionna.rt
from sionna.rt import load_scene, Transmitter, Receiver, \
                      PlanarArray, PathSolver, subcarrier_frequencies


@pytest.mark.parametrize('synthetic_array', [True, False])
def test_cfr_computation(synthetic_array):
    """Verify that CFR is correctly computed
       TX and RX are placed such that some paths are invalid
    """
    scene = load_scene(sionna.rt.scene.simple_reflector)
    scene.add(Transmitter("tx-1", [-10,0,10]))
    scene.add(Receiver("rx-1", [10,0,10]))
    scene.add(Transmitter("tx-2", [-10,1,10]))
    scene.add(Receiver("rx-2", [10,-1,10]))
    scene.tx_array = PlanarArray(num_cols=3, num_rows=3,
                                pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_cols=1, num_rows=2,
                                pattern="iso", polarization="VH")
    scene.get("tx-1").velocity = [-10, 10, 10]
    scene.get("rx-2").velocity = [-5, 8, 6]
    p_solver = PathSolver()
    paths = p_solver(scene, los=True, specular_reflection=True,
                    diffuse_reflection=False, refraction=False,
                    diffraction=True, edge_diffraction=True,
                    synthetic_array=synthetic_array)

    sampling_frequency = 10**6
    num_time_steps = 10
    num_subcarriers = 128
    subcarrier_spacing = 30e3
    frequencies = subcarrier_frequencies(num_subcarriers, subcarrier_spacing)
    h_rt = paths.cfr(
                    frequencies=frequencies,
                    sampling_frequency=sampling_frequency,
                    num_time_steps=num_time_steps,
                    normalize=False,
                    normalize_delays=False, out_type="numpy")

    a = paths.a[0].numpy() + 1j*paths.a[1].numpy()
    tau = paths.tau.numpy()
    doppler = paths.doppler.numpy()
    if synthetic_array:
        # Add dimensions for broadcasting
        num_rx, num_tx, num_paths = tau.shape
        tau = np.reshape(tau, [num_rx, 1, num_tx, 1, num_paths])
        doppler = np.reshape(doppler, [num_rx, 1, num_tx, 1, num_paths])

    # Compute baseband euivalent channel coefficients
    a_b = a*np.exp(-1j*2*np.pi*scene.frequency.numpy()*tau)

    # Apply Doppler
    t = np.arange(0, num_time_steps)/sampling_frequency
    doppler = np.expand_dims(doppler, axis=-1)
    a_ref = np.expand_dims(a_b, -1)*np.exp(1j*2*np.pi*doppler*t)

    # Compute channel frequency response
    e = np.exp(-1j*2*np.pi*np.expand_dims(tau, -1)*frequencies.numpy())
    e = np.expand_dims(e, -2)
    a_ref = np.expand_dims(a_ref, -1)
    h_ref = np.sum(a_ref*e, axis=-3)

    # Compute maximum relative error
    err = np.max(np.abs(h_rt-h_ref))

    assert err < 1e-6

@pytest.mark.parametrize('synthetic_array', [True, False])
def test_cfr_normalization(synthetic_array):
    """
    Check that normalization of the CFR across a slot is correctly applied
    """
    scene = load_scene(sionna.rt.scene.simple_reflector)
    scene.add(Transmitter("tx-1", [-10,0,10]))
    scene.add(Receiver("rx-1", [10,0,10]))
    scene.add(Transmitter("tx-2", [-10,1,10]))
    scene.add(Receiver("rx-2", [10,-1,10]))
    scene.tx_array = PlanarArray(num_cols=3, num_rows=3,
                                pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_cols=1, num_rows=2,
                                pattern="iso", polarization="VH")
    scene.get("tx-1").velocity = [-10, 10, 10]
    scene.get("rx-2").velocity = [-5, 8, 6]
    p_solver = PathSolver()
    paths = p_solver(scene, los=True, specular_reflection=True,
                    diffuse_reflection=False, refraction=False,
                    diffraction=True, edge_diffraction=True,
                    synthetic_array=synthetic_array)

    sampling_frequency = 10**6
    num_time_steps = 10
    num_subcarriers = 128
    subcarrier_spacing = 30e3
    frequencies = subcarrier_frequencies(num_subcarriers, subcarrier_spacing)
    h_rt = paths.cfr(
                    frequencies=frequencies,
                    sampling_frequency=sampling_frequency,
                    num_time_steps=num_time_steps,
                    normalize=True,
                    normalize_delays=False, out_type="numpy")
    norm = np.mean(np.abs(h_rt)**2, axis=(1,3,4,5))
    err = np.max(np.abs(norm-1))

    assert err < 1e-3
