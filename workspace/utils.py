from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import mitsuba as mi
from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter


def plot_vertices_index(road_positions, figsize=(8, 6), marker_size=10):
    """Plot XY vertices with index labels."""
    vertices = np.asarray(road_positions, dtype=np.float32)
    if vertices.size == 0:
        raise ValueError("road_positions is empty.")

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(vertices[:, 0], vertices[:, 1], s=marker_size, c="tab:blue")
    for i, (x, y) in enumerate(vertices[:, :2]):
        ax.text(x, y, str(i), fontsize=7, color="black")
    ax.set_title("Road Vertices (XY) with Indices")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_path_from_indices(road_positions, path_indices, figsize=(8, 6)):
    """Plot selected path on XY plane."""
    vertices = np.asarray(road_positions, dtype=np.float32)
    idx = np.asarray(path_indices, dtype=np.int32)
    if idx.size < 2:
        raise ValueError("path_indices must contain at least 2 indices.")

    waypoints = vertices[idx]
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(vertices[:, 0], vertices[:, 1], s=6, c="lightgray", label="vertices")
    ax.plot(waypoints[:, 0], waypoints[:, 1], "-o", c="tab:red", ms=4, lw=1.8, label="path")
    ax.scatter([waypoints[0, 0]], [waypoints[0, 1]], s=90, c="tab:green", marker="*", label="start")
    ax.scatter([waypoints[-1, 0]], [waypoints[-1, 1]], s=90, c="tab:purple", marker="*", label="end")
    ax.set_title("Selected Path (XY)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()


def get_object_vertices(scene, object_id, dtype=np.float32, round_decimals=None):
    """Extract object vertices (N,3) from Mitsuba scene by object id."""
    if not hasattr(scene, "mi_scene"):
        raise ValueError("scene has no mi_scene.")

    target = None
    for shape in scene.mi_scene.shapes():
        if shape.id() == object_id:
            target = shape
            break
    if target is None:
        raise ValueError(f"Object '{object_id}' not found in scene.mi_scene.")

    params = mi.traverse(target)
    if "vertex_positions" not in params:
        raise ValueError(f"Object '{object_id}' does not expose vertex_positions.")

    vertices = np.array(params["vertex_positions"]).reshape(-1, 3).astype(dtype)
    if round_decimals is not None:
        vertices = np.round(vertices.astype(np.float32), int(round_decimals)).astype(dtype)
    return vertices


def prepare_path_data(road_positions, path_indices, z_offset=1.5, speed_ms=25.0, delta_t=0.5):
    """Build frame/time sampling data from selected path indices."""
    vertices = np.asarray(road_positions, dtype=np.float32)
    idx = np.asarray(path_indices, dtype=np.int32)
    if idx.size < 2:
        raise ValueError("path_indices must contain at least 2 indices.")

    waypoints = vertices[idx].copy()
    waypoints[:, 2] += float(z_offset)

    seg = waypoints[1:] - waypoints[:-1]
    seg_dist = np.linalg.norm(seg, axis=1)
    cum_dist = np.concatenate([[0.0], np.cumsum(seg_dist)])
    total_dist = float(cum_dist[-1])
    total_time = 0.0 if speed_ms <= 0 else total_dist / float(speed_ms)
    frame_times = np.arange(0.0, total_time + delta_t, delta_t, dtype=np.float32)

    return {
        "waypoints": waypoints,
        "speed_ms": float(speed_ms),
        "delta_t": float(delta_t),
        "segment_dists": seg_dist,
        "cumulative_dists": cum_dist,
        "total_distance": total_dist,
        "total_time": total_time,
        "frame_times": frame_times,
    }


def make_fixed_path_data(position, delta_t=0.5):
    """Build path_data for a device fixed at a specific XYZ position."""
    pos = np.asarray(position, dtype=np.float32).reshape(3)
    waypoints = np.stack([pos, pos], axis=0).astype(np.float32)

    return {
        "waypoints": waypoints,
        "speed_ms": 0.0,
        "delta_t": float(delta_t),
        "segment_dists": np.array([0.0], dtype=np.float32),
        "cumulative_dists": np.array([0.0, 0.0], dtype=np.float32),
        "total_distance": 0.0,
        "total_time": 0.0,
        "frame_times": np.array([0.0], dtype=np.float32),
    }


def get_state_at_time(path_data, t):
    """Return position and velocity at time t (seconds)."""
    waypoints = path_data["waypoints"]
    seg_dist = path_data["segment_dists"]
    cum_dist = path_data["cumulative_dists"]
    speed_ms = path_data["speed_ms"]
    total_dist = path_data["total_distance"]

    target_dist = np.clip(float(t) * speed_ms, 0.0, total_dist)
    idx = np.searchsorted(cum_dist, target_dist) - 1
    idx = int(np.clip(idx, 0, len(waypoints) - 2))

    seg_len = seg_dist[idx]
    if seg_len <= 1e-9:
        pos = waypoints[idx]
        vel = np.zeros(3, dtype=np.float32)
        return pos.astype(np.float32), vel

    ratio = (target_dist - cum_dist[idx]) / seg_len
    pos = waypoints[idx] + ratio * (waypoints[idx + 1] - waypoints[idx])
    direction = (waypoints[idx + 1] - waypoints[idx]) / seg_len
    vel = direction * speed_ms
    return pos.astype(np.float32), vel.astype(np.float32)


def get_state_at_frame(path_data, frame_idx):
    """Return position and velocity at frame index."""
    frame_times = path_data["frame_times"]
    if frame_idx < 0 or frame_idx >= len(frame_times):
        raise IndexError(f"frame_idx out of range: {frame_idx}")
    return get_state_at_time(path_data, float(frame_times[frame_idx]))


def get_state_at_frame_clamped(path_data, frame_idx):
    """Return state at frame index, clamped to last valid frame."""
    frame_times = path_data["frame_times"]
    if len(frame_times) == 0:
        raise ValueError("path_data['frame_times'] is empty.")
    idx = int(np.clip(frame_idx, 0, len(frame_times) - 1))
    return get_state_at_time(path_data, float(frame_times[idx]))


def setup_tx_rx(
    scene,
    tx_positions,
    rx_positions,
    tx_power_dbm=43.0,
    tx_names=None,
    rx_names=None,
    tx_velocities=None,
    rx_velocities=None,
):
    """Reset devices and add Tx/Rx for path simulation.

    Single/multi Rx are both list-based inputs.
    Returns tx_names, rx_names, solver.
    """
    if hasattr(scene, "transmitters"):
        for name in list(scene.transmitters.keys()):
            scene.remove(name)
    if hasattr(scene, "receivers"):
        for name in list(scene.receivers.keys()):
            scene.remove(name)

    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="VH")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="VH")

    if tx_velocities is None:
        tx_velocities = [[0.0, 0.0, 0.0] for _ in tx_positions]
    if len(tx_velocities) != len(tx_positions):
        raise ValueError("len(tx_velocities) must match len(tx_positions).")

    rx_positions = list(rx_positions)
    if len(rx_positions) == 0:
        raise ValueError("rx_positions must be non-empty.")

    if rx_velocities is None:
        rx_velocities = [[0.0, 0.0, 0.0] for _ in rx_positions]
    if len(rx_velocities) != len(rx_positions):
        raise ValueError("len(rx_velocities) must match number of Rx positions.")

    if tx_names is None:
        tx_names = [f"tx_{i}" for i in range(len(tx_positions))]
    if len(tx_names) != len(tx_positions):
        raise ValueError("len(tx_names) must match number of Tx positions.")

    if rx_names is None:
        rx_names = [f"rx_{i}" for i in range(len(rx_positions))]
    if len(rx_names) != len(rx_positions):
        raise ValueError("len(rx_names) must match number of Rx positions.")

    rx_ref = np.asarray(rx_positions[0], dtype=np.float32)

    created_tx_names = []
    for tx_name, pos, vel in zip(tx_names, tx_positions, tx_velocities):
        tx = Transmitter(
            name=tx_name,
            position=np.asarray(pos, dtype=np.float32),
            power_dbm=float(tx_power_dbm),
            velocity=np.asarray(vel, dtype=np.float32),
        )
        tx.transmit_antenna = scene.tx_array
        tx.look_at(rx_ref)
        scene.add(tx)
        created_tx_names.append(tx_name)

    rx_list = []
    for name, pos, vel in zip(rx_names, rx_positions, rx_velocities):
        rx = Receiver(
            name=name,
            position=np.asarray(pos, dtype=np.float32),
            velocity=np.asarray(vel, dtype=np.float32),
        )
        rx.receive_antenna = scene.rx_array
        scene.add(rx)
        rx_list.append(rx)

    return created_tx_names, list(rx_names), PathSolver()


def solve_paths_for_frame(
    scene,
    solver,
    rx_names,
    rx_path_data,
    tx_names,
    tx_path_data,
    frame_idx,
    tx_look_at_rx=True,
    tx_look_at_rx_idx=0,
    **solver_kwargs,
):
    """Update frame state and solve paths.

    rx_path_data can be a single path_data or list of path_data (for multi-Rx).
    tx_path_data can be a single path_data or list of path_data (for multi-Tx).
    """
    if rx_path_data is None:
        raise ValueError("rx_path_data is required.")
    if tx_path_data is None:
        raise ValueError("tx_path_data is required.")

    rx_name_list = list(rx_names) if isinstance(rx_names, (list, tuple)) else [rx_names]
    if isinstance(rx_path_data, (list, tuple)):
        rx_path_list = list(rx_path_data)
    else:
        rx_path_list = [rx_path_data for _ in rx_name_list]
    if len(rx_name_list) != len(rx_path_list):
        raise ValueError("len(rx_names) must match len(rx_path_data).")

    rx_pos_list = []
    rx_vel_list = []
    for rx_name, rx_pd in zip(rx_name_list, rx_path_list):
        rx_pos, rx_vel = get_state_at_frame_clamped(rx_pd, frame_idx)
        rx_obj = scene.receivers[rx_name]
        rx_obj.position = rx_pos
        rx_obj.velocity = rx_vel
        rx_pos_list.append(rx_pos)
        rx_vel_list.append(rx_vel)

    tx_name_list = list(tx_names) if isinstance(tx_names, (list, tuple)) else [tx_names]
    if isinstance(tx_path_data, (list, tuple)):
        tx_path_list = list(tx_path_data)
    else:
        tx_path_list = [tx_path_data for _ in tx_name_list]
    if len(tx_name_list) != len(tx_path_list):
        raise ValueError("len(tx_names) must match len(tx_path_data).")

    tx_pos_list = []
    tx_vel_list = []
    for tx_name, tx_pd in zip(tx_name_list, tx_path_list):
        tx_pos, tx_vel = get_state_at_frame_clamped(tx_pd, frame_idx)
        tx_obj = scene.transmitters[tx_name]
        tx_obj.position = tx_pos
        tx_obj.velocity = tx_vel
        tx_pos_list.append(tx_pos)
        tx_vel_list.append(tx_vel)

    if tx_look_at_rx:
        if tx_look_at_rx_idx < 0 or tx_look_at_rx_idx >= len(rx_pos_list):
            raise IndexError("tx_look_at_rx_idx out of range.")
        look_at_pos = rx_pos_list[tx_look_at_rx_idx]
        for name in tx_name_list:
            scene.transmitters[name].look_at(look_at_pos)

    paths = solver(scene, **solver_kwargs)
    return paths, np.asarray(rx_pos_list), np.asarray(rx_vel_list), np.asarray(tx_pos_list), np.asarray(tx_vel_list)


def _extract_tau_ns(tau):
    tau_np = tau.numpy() if hasattr(tau, "numpy") else np.asarray(tau)
    if tau_np.size == 0:
        return np.array([], dtype=np.float32)
    if tau_np.ndim == 3:
        return (tau_np[0, 0, :] / 1e-9).astype(np.float32)
    if tau_np.ndim == 4:
        return (tau_np[0, 0, 0, :] / 1e-9).astype(np.float32)
    if tau_np.ndim == 5:
        return (tau_np[0, 0, 0, 0, :] / 1e-9).astype(np.float32)
    return (np.ravel(tau_np) / 1e-9).astype(np.float32)


def _extract_cir_abs(a):
    a_np = a.numpy() if hasattr(a, "numpy") else np.asarray(a)
    if a_np.size == 0:
        return np.array([], dtype=np.float32)
    mag = np.abs(a_np)
    if mag.ndim == 1:
        return mag.astype(np.float32)
    if mag.ndim == 2:
        return mag[:, 0].astype(np.float32)

    idx = [0] * (mag.ndim - 2) + [slice(None), 0]
    return mag[tuple(idx)].astype(np.float32)


def get_cir_for_frame(
    scene,
    solver,
    rx_names,
    rx_path_data,
    tx_names,
    tx_path_data,
    frame_idx,
    cir_kwargs=None,
    solver_kwargs=None,
    rx_idx=0,
):
    """Compute CIR at specific frame index."""
    cir_kwargs = cir_kwargs or {}
    solver_kwargs = solver_kwargs or {}

    paths, rx_pos, rx_vel, tx_pos, tx_vel = solve_paths_for_frame(
        scene=scene,
        solver=solver,
        rx_names=rx_names,
        rx_path_data=rx_path_data,
        tx_names=tx_names,
        tx_path_data=tx_path_data,
        frame_idx=frame_idx,
        **solver_kwargs,
    )
    rx_name_list = list(rx_names) if isinstance(rx_names, (list, tuple)) else [rx_names]
    if isinstance(rx_path_data, (list, tuple)):
        rx_path_list = list(rx_path_data)
    else:
        rx_path_list = [rx_path_data for _ in rx_name_list]
    if rx_idx < 0 or rx_idx >= len(rx_name_list):
        raise IndexError("rx_idx out of range.")
    a, tau = paths.cir(out_type="tf", normalize_delays=False, **cir_kwargs)
    return {
        "frame_idx": int(frame_idx),
        "time_s": float(rx_path_list[rx_idx]["frame_times"][frame_idx]),
        "rx_pos": rx_pos,
        "rx_vel": rx_vel,
        "tx_pos": tx_pos,
        "tx_vel": tx_vel,
        "paths": paths,
        "a": a,
        "tau": tau,
        "a_abs": _extract_cir_abs(a),
        "tau_ns": _extract_tau_ns(tau),
    }


def get_cfr_for_frame(
    scene,
    solver,
    rx_names,
    rx_path_data,
    tx_names,
    tx_path_data,
    frame_idx,
    frequencies,
    cfr_kwargs=None,
    solver_kwargs=None,
    rx_idx=0,
):
    """Compute CFR at specific frame index."""
    cfr_kwargs = cfr_kwargs or {}
    solver_kwargs = solver_kwargs or {}

    freqs = np.asarray(frequencies, dtype=np.float32).reshape(-1)
    if freqs.size == 0:
        raise ValueError("frequencies is empty.")

    paths, rx_pos, rx_vel, tx_pos, tx_vel = solve_paths_for_frame(
        scene=scene,
        solver=solver,
        rx_names=rx_names,
        rx_path_data=rx_path_data,
        tx_names=tx_names,
        tx_path_data=tx_path_data,
        frame_idx=frame_idx,
        **solver_kwargs,
    )
    rx_name_list = list(rx_names) if isinstance(rx_names, (list, tuple)) else [rx_names]
    if isinstance(rx_path_data, (list, tuple)):
        rx_path_list = list(rx_path_data)
    else:
        rx_path_list = [rx_path_data for _ in rx_name_list]
    if rx_idx < 0 or rx_idx >= len(rx_name_list):
        raise IndexError("rx_idx out of range.")
    h = paths.cfr(frequencies=freqs, out_type="numpy", **cfr_kwargs)
    h_np = np.asarray(h)
    h_sel = h_np[tuple([0] * (h_np.ndim - 1) + [slice(None)])]

    return {
        "frame_idx": int(frame_idx),
        "time_s": float(rx_path_list[rx_idx]["frame_times"][frame_idx]),
        "rx_pos": rx_pos,
        "rx_vel": rx_vel,
        "tx_pos": tx_pos,
        "tx_vel": tx_vel,
        "paths": paths,
        "frequencies": freqs,
        "h_freq": h,
        "h_abs": np.abs(h_sel),
    }
