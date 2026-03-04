"""jinsup 시뮬레이션 설정을 재사용해 프레임별 CIR/PDP 데이터를 저장하는 스크립트."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SimConfig:
    """CIR 추출에 필요한 주요 설정 묶음."""

    xml_path: str = "/home/mh/kmh/sionna-rt/jinsup/scene/blender_tmp/HSV_260227_flat.xml"
    road_object_id: str = "elm__19"

    tx_positions: tuple[tuple[float, float, float], ...] = ((0.0, -264.0, 11.0),)
    tx_power_dbm: float = 43.0

    path_indices: tuple[int, int] = (4, 16)
    z_offset: float = 2.7
    rx_z: float = 4.0
    speed_ms: float = 100.0 / 3.6
    target_frames: int = 281

    sim_profile: str = "balanced"  # safe | balanced | high
    synthetic_array: bool = True

    bandwidth: float = 500e6
    l_min: int = 0
    l_max: int = 4094

    add_measurement_noise: bool = True
    noise_floor_rel_db: float = -45.0
    noise_random_seed: int = 1234


def _resolve_profile(sim_profile: str) -> dict[str, object]:
    """프로파일 이름을 solver 파라미터로 변환한다."""
    if sim_profile == "safe":
        return {
            "max_depth": 2,
            "samples_per_src": 20_000,
            "diffuse_reflection": False,
            "diffraction": False,
        }
    if sim_profile == "balanced":
        return {
            "max_depth": 3,
            "samples_per_src": 50_000,
            "diffuse_reflection": True,
            "diffraction": False,
        }
    if sim_profile == "high":
        return {
            "max_depth": 4,
            "samples_per_src": 100_000,
            "diffuse_reflection": True,
            "diffraction": True,
        }
    raise ValueError(f"알 수 없는 sim_profile: {sim_profile}")


def get_road_positions_from_object(
    scene,
    road_object_id: str,
    dtype=np.float32,
    round_decimals: int | None = None,
):
    """Mitsuba shape에서 vertex_positions를 읽어 도로 정점 좌표를 반환한다."""
    import mitsuba as mi

    road_positions = []

    if not hasattr(scene, "mi_scene"):
        return road_positions

    mi_scene = scene.mi_scene
    target_shape = None
    for shape in mi_scene.shapes():
        if shape.id() == road_object_id:
            target_shape = shape
            break

    if target_shape is None:
        return road_positions

    params = mi.traverse(target_shape)
    if "vertex_positions" not in params:
        return road_positions

    vertices_flat = np.array(params["vertex_positions"])
    if len(vertices_flat) == 0:
        return road_positions

    road_positions = vertices_flat.reshape(-1, 3).astype(dtype)

    if round_decimals is not None:
        # 한글 주석: 반올림은 float32에서 수행해 수치 오차를 줄인다.
        rounded = np.round(road_positions.astype(np.float32), round_decimals)
        road_positions = rounded.astype(dtype)

    return road_positions


def prepare_vehicle_path(
    road_positions,
    path_indices,
    z_offset: float = 1.5,
    speed_ms: float = 90.0 / 3.6,
    delta_t: float = 0.5,
):
    """수동 인덱스(path_indices) 기반으로 이동 경로와 시간축을 만든다."""
    if road_positions is None or len(road_positions) == 0:
        raise ValueError("road_positions 데이터가 없습니다.")
    if path_indices is None or len(path_indices) < 2:
        raise ValueError("path_indices는 최소 2개 이상 필요합니다.")

    waypoints = np.asarray(road_positions[path_indices], dtype=np.float32).copy()
    waypoints[:, 2] += float(z_offset)

    diffs = waypoints[1:] - waypoints[:-1]
    segment_dists = np.linalg.norm(diffs, axis=1)
    cumulative_dists = np.concatenate(([0.0], np.cumsum(segment_dists)))
    total_distance = float(cumulative_dists[-1])
    total_time = total_distance / float(speed_ms) if speed_ms > 0 else 0.0
    time_steps = np.arange(0.0, total_time + delta_t, delta_t, dtype=np.float32)

    return {
        "waypoints": waypoints,
        "speed_ms": float(speed_ms),
        "delta_t": float(delta_t),
        "segment_dists": segment_dists,
        "cumulative_dists": cumulative_dists,
        "total_distance": total_distance,
        "total_time": total_time,
        "time_steps": time_steps,
    }


def get_state_at_time(path_data, t: float):
    """시각 t에서 차량 위치(pos)와 속도벡터(vel)를 반환한다."""
    waypoints = path_data["waypoints"]
    speed_ms = path_data["speed_ms"]
    segment_dists = path_data["segment_dists"]
    cumulative_dists = path_data["cumulative_dists"]
    total_distance = path_data["total_distance"]

    if len(waypoints) < 2:
        return waypoints[0], np.array([0.0, 0.0, 0.0], dtype=np.float32)

    target_dist = np.clip(speed_ms * float(t), 0.0, total_distance)
    idx = np.searchsorted(cumulative_dists, target_dist) - 1
    idx = max(0, min(idx, len(waypoints) - 2))

    seg_start = cumulative_dists[idx]
    seg_len = segment_dists[idx]
    if seg_len <= 1e-9:
        return waypoints[idx], np.array([0.0, 0.0, 0.0], dtype=np.float32)

    ratio = (target_dist - seg_start) / seg_len
    pos = waypoints[idx] + ratio * (waypoints[idx + 1] - waypoints[idx])

    # 한글 주석: 진행 방향 단위벡터에 속도 크기를 곱해 속도벡터를 만든다.
    direction = (waypoints[idx + 1] - waypoints[idx]) / seg_len
    vel = direction * speed_ms
    return pos.astype(np.float32), vel.astype(np.float32)


def _extract_1d_paths_from_cir(a, tau):
    """paths.cir() 출력에서 단일 링크의 path 계수/지연 1D를 추출한다."""
    a_np = a.numpy() if hasattr(a, "numpy") else np.asarray(a)
    tau_np = tau.numpy() if hasattr(tau, "numpy") else np.asarray(tau)

    # 한글 주석: tau는 항상 1D path 축으로 평탄화한다(경로 1개면 길이 1).
    tau_1d = np.ravel(np.squeeze(tau_np)).astype(np.float64)
    if tau_1d.ndim == 0:
        tau_1d = tau_1d.reshape(1)
    p = int(tau_1d.shape[0])

    # 한글 주석: a에서 path 길이(p)와 일치하는 축을 찾아 첫 링크/안테나 조합을 선택한다.
    a_sq = np.squeeze(a_np)
    if p == 0:
        return np.zeros((0,), dtype=np.complex128), tau_1d

    if np.ndim(a_sq) == 0:
        if p != 1:
            raise ValueError(f"a/tau path size mismatch: a={np.shape(a_sq)}, tau={tau_1d.shape}")
        a_path = np.asarray([a_sq], dtype=np.complex128)
        return a_path, tau_1d

    shape = np.shape(a_sq)
    candidate_axes = [ax for ax, size in enumerate(shape) if size == p]

    if candidate_axes:
        path_axis = candidate_axes[-1]
        moved = np.moveaxis(a_sq, path_axis, 0).reshape(p, -1)
        a_path = moved[:, 0]
    else:
        flat = np.ravel(a_sq)
        if p == 1:
            a_path = flat[:1]
        elif flat.size == p:
            a_path = flat
        elif flat.size > p:
            a_path = flat[:p]
        else:
            raise ValueError(f"cannot align a with tau: a={shape}, tau={tau_1d.shape}")

    return np.asarray(a_path, dtype=np.complex128), tau_1d


def _map_paths_to_cir_bins(paths, bandwidth: float, l_min: int, l_max: int):
    """연속 지연 경로를 칩 bin(CIR)으로 변환한다."""
    a, tau = paths.cir(out_type="tf", normalize_delays=False)
    a_path, tau_s = _extract_1d_paths_from_cir(a, tau)

    n_bins = int(l_max - l_min + 1)
    dt = 1.0 / float(bandwidth)

    # 한글 주석: delay(초)를 칩 인덱스로 반올림해 bin에 누적한다.
    chip_idx = np.rint(tau_s / dt).astype(np.int64)
    idx = chip_idx - int(l_min)
    valid = (idx >= 0) & (idx < n_bins)

    h_cir = np.zeros(n_bins, dtype=np.complex128)
    np.add.at(h_cir, idx[valid], a_path[valid])

    return h_cir, tau_s, a_path


def _inject_noise_on_cir(
    h_cir: np.ndarray,
    frame_idx: int,
    add_noise: bool,
    noise_floor_rel_db: float,
    noise_random_seed: int,
) -> np.ndarray:
    """복소 CIR에만 AWGN을 추가한다.

    노이즈 전력은 프레임 내 CIR 피크 전력 대비 `noise_floor_rel_db`로 설정한다.
    예) -45 dB면 noise_p = peak_p * 10^(-45/10)
    """
    if not add_noise:
        return h_cir

    peak_p = float(np.max(np.abs(h_cir) ** 2)) if h_cir.size else 0.0
    if peak_p <= 0.0:
        return h_cir

    rng = np.random.default_rng(noise_random_seed + int(frame_idx))

    noise_p = peak_p * (10.0 ** (float(noise_floor_rel_db) / 10.0))
    sigma = np.sqrt(max(noise_p, 1e-30) / 2.0)

    n_re = rng.normal(0.0, sigma, size=h_cir.shape)
    n_im = rng.normal(0.0, sigma, size=h_cir.shape)
    return h_cir + (n_re + 1j * n_im)


def run_export(config: SimConfig, out_dir: Path) -> None:
    """프레임별 CIR/PDP(npz)와 요약(csv)을 저장한다."""
    # 한글 주석: --help에서도 죽지 않도록 무거운 의존성 import를 지연한다.
    from sionna.rt import PathSolver, Receiver, Transmitter, PlanarArray, load_scene

    solver_opt = _resolve_profile(config.sim_profile)

    scene = load_scene(config.xml_path, merge_shapes=False)
    scene.frequency = 28e9

    road_positions = get_road_positions_from_object(
        scene, config.road_object_id, dtype=np.float32, round_decimals=3
    )
    if len(road_positions) == 0:
        raise RuntimeError("road_positions is empty")

    warmup = prepare_vehicle_path(
        road_positions=road_positions,
        path_indices=list(config.path_indices),
        z_offset=config.z_offset,
        speed_ms=config.speed_ms,
        delta_t=0.5,
    )
    delta_t = float(warmup["total_time"]) / float(config.target_frames - 1)

    path_data = prepare_vehicle_path(
        road_positions=road_positions,
        path_indices=list(config.path_indices),
        z_offset=config.z_offset,
        speed_ms=config.speed_ms,
        delta_t=delta_t,
    )

    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="VH")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="VH")

    tx_names = []
    for i, pos in enumerate(config.tx_positions):
        tx_name = f"Tx_{i+1}"
        tx = Transmitter(name=tx_name, position=list(pos), power_dbm=config.tx_power_dbm)
        tx.transmit_antenna = scene.tx_array
        tx.look_at([0, 0, 0])
        scene.add(tx)
        tx_names.append(tx_name)

    start_pos, start_vel = get_state_at_time(path_data, 0.0)
    rx = Receiver(
        name="rx_car",
        position=[float(start_pos[0]), float(start_pos[1]), float(config.rx_z)],
        velocity=start_vel,
        display_radius=2.0,
    )
    rx.receive_antenna = scene.rx_array
    scene.add(rx)

    solver = PathSolver()

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    n_frames = len(path_data["time_steps"])
    print(f"export start: frames={n_frames}, out={out_dir.resolve()}")

    for frame_idx, t in enumerate(path_data["time_steps"]):
        pos, vel = get_state_at_time(path_data, float(t))
        rx.position = np.array([float(pos[0]), float(pos[1]), float(config.rx_z)], dtype=np.float32)
        rx.velocity = vel

        for name in tx_names:
            scene.transmitters[name].look_at(pos)

        paths = solver(
            scene,
            max_depth=int(solver_opt["max_depth"]),
            samples_per_src=int(solver_opt["samples_per_src"]),
            diffuse_reflection=bool(solver_opt["diffuse_reflection"]),
            diffraction=bool(solver_opt["diffraction"]),
            synthetic_array=config.synthetic_array,
        )

        h_cir_clean, tau_s, a_path = _map_paths_to_cir_bins(
            paths,
            bandwidth=config.bandwidth,
            l_min=config.l_min,
            l_max=config.l_max,
        )
        h_cir_noisy = _inject_noise_on_cir(
            h_cir_clean,
            frame_idx=frame_idx,
            add_noise=config.add_measurement_noise,
            noise_floor_rel_db=config.noise_floor_rel_db,
            noise_random_seed=config.noise_random_seed,
        )

        pdp_clean = np.abs(h_cir_clean) ** 2
        pdp_noisy = np.abs(h_cir_noisy) ** 2

        frame_out = out_dir / f"cir_frame_{frame_idx:04d}.npz"
        np.savez_compressed(
            frame_out,
            frame_idx=np.int32(frame_idx),
            time_s=np.float64(float(t)),
            position_xyz=np.asarray([float(pos[0]), float(pos[1]), float(config.rx_z)], dtype=np.float64),
            velocity_xyz=np.asarray(vel, dtype=np.float64),
            h_cir_clean=h_cir_clean,
            h_cir_noisy=h_cir_noisy,
            pdp_clean=pdp_clean,
            pdp_noisy=pdp_noisy,
            tau_s=tau_s,
            a_path=a_path,
            bandwidth_hz=np.float64(config.bandwidth),
            l_min=np.int32(config.l_min),
            l_max=np.int32(config.l_max),
            noise_floor_rel_db=np.float64(config.noise_floor_rel_db),
            add_measurement_noise=np.int8(1 if config.add_measurement_noise else 0),
        )

        peak_clean_db = 10.0 * np.log10(max(float(np.max(pdp_clean)), 1e-30))
        peak_noisy_db = 10.0 * np.log10(max(float(np.max(pdp_noisy)), 1e-30))
        summary_rows.append((frame_idx, float(t), peak_clean_db, peak_noisy_db, frame_out.name))

        if frame_idx % 20 == 0 or frame_idx == n_frames - 1:
            print(f"  frame={frame_idx:04d} saved")

    summary_path = out_dir / "summary.csv"
    np.savetxt(
        summary_path,
        np.asarray(summary_rows, dtype=object),
        fmt="%s",
        delimiter=",",
        header="frame_idx,time_s,peak_clean_db,peak_noisy_db,file_name",
        comments="",
    )
    print(f"done: {summary_path}")


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="jinsup 기반 CIR 데이터 추출")
    parser.add_argument("--xml-path", type=str, default="/home/mh/kmh/sionna-rt/jinsup/scene/blender_tmp/HSV_260227_flat.xml")
    parser.add_argument("--road-object-id", type=str, default="elm__19")
    parser.add_argument("--out-dir", type=str, default="jinsup/cir_exports")
    parser.add_argument("--target-frames", type=int, default=281)
    parser.add_argument("--sim-profile", type=str, default="balanced", choices=["safe", "balanced", "high"])
    parser.add_argument("--noise-floor-rel-db", type=float, default=-45.0)
    parser.add_argument("--disable-noise", action="store_true")
    return parser


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()

    config = SimConfig(
        xml_path=args.xml_path,
        road_object_id=args.road_object_id,
        target_frames=args.target_frames,
        sim_profile=args.sim_profile,
        noise_floor_rel_db=args.noise_floor_rel_db,
        add_measurement_noise=not args.disable_noise,
    )
    run_export(config=config, out_dir=Path(args.out_dir))


if __name__ == "__main__":
    main()
