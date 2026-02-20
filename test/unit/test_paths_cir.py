#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import numpy as np
import drjit as dr
import mitsuba as mi

import sionna.rt
from sionna.rt import load_scene, Transmitter, Receiver, \
                      PlanarArray, PathSolver, r_hat

@pytest.mark.parametrize('synthetic_array', [True, False])
def test_cir_computation(synthetic_array):
    """Verify that CIR is correctly computed
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
    a_rt, _ = paths.cir(sampling_frequency=sampling_frequency,
                    num_time_steps=num_time_steps,
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
    err = np.max(np.abs(a_rt-a_ref)/np.where(a_rt==0, 1, np.abs(a_rt)))
    assert err < 1e-3

@pytest.mark.parametrize('synthetic_array', [True, False])
def test_cir_delay_normalization(synthetic_array):
    """Test that normalization of path delays works as expected.
       TX and RX are placed such that some paths are invalid.
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
    p_solver = PathSolver()

    # Compute paths
    paths = p_solver(scene, los=True, specular_reflection=True,
                    diffuse_reflection=False, refraction=False,
                    diffraction=True, edge_diffraction=True,
                    synthetic_array=synthetic_array)
    _, tau = paths.cir(normalize_delays=False, out_type="numpy")
    _, tau_norm = paths.cir(normalize_delays=True, out_type="numpy")

    if not synthetic_array:
        num_rx, _, num_tx, _, _ = tau.shape
        for i in range(num_rx):
            for j in range(num_tx):
                z = tau[i,:,j,:]
                z = np.where(z<0, np.inf, z)
                t_min = np.min(z)
                z -= t_min
                z = np.where(z==np.inf, -1, z)
                assert np.array_equal(z, tau_norm[i,:,j,:])
    else:
        num_rx, num_tx, _ = tau.shape
        for i in range(num_rx):
            for j in range(num_tx):
                z = tau[i,j]
                z = np.where(z<0, np.inf, z)
                t_min = np.min(z)
                z -= t_min
                z = np.where(z==np.inf, -1, z)
                assert np.array_equal(z, tau_norm[i,j])

def test_cir_doppler_vs_geometry_updates():
    scene = load_scene(sionna.rt.scene.simple_reflector, merge_shapes=False)
    scene.add(Transmitter("tx", [-10,0,10]))
    scene.add(Receiver("rx", [10,0,10]))
    scene.tx_array = PlanarArray(num_cols=1, num_rows=1,
                                 pattern="iso", polarization="V")
    scene.rx_array = scene.tx_array
    v_tx = np.array([3., 0., -3.])
    v_rx = np.array([-3., 0., -3.])
    v_ref =  np.array([0., 0., 3.])
    scene.get("tx").velocity = v_tx
    scene.get("rx").velocity = v_rx
    scene.get("reflector").velocity = v_ref

    p_solver = PathSolver()

    paths = p_solver(scene, los=True, specular_reflection=True,
                    diffuse_reflection=False, refraction=False,
                    diffraction=False, edge_diffraction=False,
                    synthetic_array=False)

    # Doppler-based time evolution
    sampling_frequency = 10**4
    num_time_steps = 10
    a, _ = paths.cir(sampling_frequency=sampling_frequency,
                    num_time_steps=num_time_steps,
                    normalize_delays=False, out_type="numpy")
    a = np.squeeze(a)

    # Geometrical time evolution
    d_tx = v_tx/sampling_frequency
    d_rx = v_rx/sampling_frequency
    d_ref = v_ref/sampling_frequency
    a_geo = np.zeros([2,0])
    for _ in range(num_time_steps):
        paths = p_solver(scene, los=True, specular_reflection=True,
                    diffuse_reflection=False, refraction=False,
                    diffraction=False, edge_diffraction=False,
                    synthetic_array=False)
        a_geo_, _ = paths.cir(normalize_delays=False, out_type="numpy")
        a_geo_ = np.squeeze(a_geo_, axis=(0,1,2,3))
        a_geo = np.concatenate([a_geo, a_geo_], axis=1)
        scene.get("tx").position  += d_tx
        scene.get("rx").position  += d_rx
        scene.get("reflector").position  += d_ref

    assert np.max(np.abs(a - a_geo)/ np.abs(a)) < 0.005

@pytest.mark.parametrize('synthetic_array', [True, False])
def test_cir_reverse_direction(synthetic_array):

    scene = load_scene(sionna.rt.scene.box)

    scene.tx_array = PlanarArray(num_rows=4, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=8, num_cols=1, pattern="iso", polarization="V")

    scene.add(Transmitter("tx-1",
                        position=(4, 3, 1.5)))
    scene.add(Transmitter("tx-2",
                        position=(4, 3, 2.5)))
    scene.add(Transmitter("tx-3",
                        position=(4, 3, 3.5)))

    scene.add(Receiver("rx-1",
                    position=(-4, 0, 2.0)))
    scene.add(Receiver("rx-2",
                    position=(-4, 0, 3.0)))

    p_solver = PathSolver()
    paths = p_solver(scene, synthetic_array=synthetic_array)

    a, tau = paths.cir(sampling_frequency=1e8, out_type="numpy",
                    reverse_direction=False)

    a_r, tau_r = paths.cir(sampling_frequency=1e8, out_type="numpy",
                    reverse_direction=True)

    # Reverse direction using numpy
    a_ref = np.transpose(a, (2, 3, 0, 1, 4, 5))
    if synthetic_array:
        tau_ref = np.transpose(tau, (1, 0, 2))
    else:
        tau_ref = np.transpose(tau, (2, 3, 0, 1, 4))

    assert a_ref.shape == a_r.shape
    assert tau_ref.shape == tau_r.shape

    assert np.allclose(a_ref, a_r)
    assert np.allclose(tau_ref, tau_r)

def test_aoa_aod():
    scene = load_scene(sionna.rt.scene.box)

    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )

    tx_position = [0, 0, 1.0]
    rx_position = [0, 0, 1.0]


    tx = Transmitter(name="tx", position=tx_position)
    rx = Receiver(name="rx", position=rx_position)

    scene.add(tx)
    scene.add(rx)

    p_solver = PathSolver()

    MAX_DEPTH = [1, 2]

    for max_depth in MAX_DEPTH:
        paths = p_solver(
            scene=scene,
            max_depth=max_depth,
            los=False,
            specular_reflection=True,
            diffuse_reflection=False,
            refraction=False,
            diffraction=False,
            synthetic_array=True,
            seed=1,
        )

        # If paths have an even number of bounces, then the direction of arrival
        # should be opposite to the direction of departure
        int_types = paths.interactions.numpy()[:,0,0,:]
        int_types = int_types > 0
        depth = np.sum(int_types, axis=0)
        flip_direction = (-1)**(depth+1)

        theta_r = paths.theta_r.array
        phi_r = paths.phi_r.array
        theta_t = paths.theta_t.array
        phi_t = paths.phi_t.array

        k_tx = r_hat(theta_t, phi_t)
        k_rx = r_hat(theta_r, phi_r)

        ktx_dot_krx = dr.dot(k_tx, k_rx).numpy()
        max_d = np.max(np.abs(ktx_dot_krx - flip_direction))
        assert max_d < 1e-5

@pytest.mark.parametrize("phi",  np.linspace(-np.pi*0.5, np.pi*0.5, 10))
def test_synthetic_array(phi):
    """Test that synthetic and real arrays elad to similar channel impulse
    in the far field responses"""

    # Load empty scene
    scene = load_scene()
    wavelength = scene.frequency.numpy()[0]

    # Solver
    solver = PathSolver()

    # Set arrays
    scene.tx_array = PlanarArray(num_rows=3,
                                 num_cols=3,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern="tr38901",
                                 polarization="VH")
    scene.rx_array = PlanarArray(num_rows=3,
                                 num_cols=3,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern="dipole",
                                 polarization="cross")

    d = 100.0
    # Transmitter and receiver
    x = d*dr.cos(phi)
    y = d*dr.sin(phi)
    rx = Receiver("rx", position=mi.Point3f(0., 0., 0.))
    tx = Transmitter("tx", position=mi.Point3f(x, y, 0))
    tx.look_at(rx)
    scene.add(tx)
    scene.add(rx)

    # Synthetic array
    paths = solver(scene, synthetic_array = True)
    a_real, a_imag = paths.a
    a_real = a_real.numpy()
    a_imag = a_imag.numpy()
    a = a_real + 1j*a_imag
    a_synthetic = a[0,:,0,:,0]
    tau_synthetic = paths.tau.numpy()[0,0,0]
    del paths, a_real, a_imag, a

    # Non-synthetic array
    paths = solver(scene, synthetic_array = False)
    a_real, a_imag = paths.a
    a_real = a_real.numpy()
    a_imag = a_imag.numpy()
    a = a_real + 1j*a_imag
    a_non_synthetic = a[0,:,0,:,0]
    tau_non_synthetic = paths.tau.numpy()[0,:,0,:,0]
    del paths, a_real, a_imag, a

    # Apply phase shift due to different propagation delays between the antenna for
    # the simulation using non-synthetic arrays
    a_non_synthetic = a_non_synthetic*np.exp(-1j*2.*np.pi*(tau_non_synthetic-tau_synthetic)*wavelength)

    max_err = np.max(np.abs(a_synthetic - a_non_synthetic)/np.abs(a_synthetic))
    assert max_err < 1e-2
