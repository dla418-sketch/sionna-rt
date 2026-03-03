"""Mobility_Analysis.ipynb용 재질/정점/이동 시뮬레이션 유틸."""

import numpy as np
import mitsuba as mi
import tensorflow as tf
import ipywidgets as widgets
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import plotly.graph_objects as go
from sionna.rt import PlanarArray, Transmitter, Receiver, PathSolver

def plot_vertices_index(road_positions,
                            width: int = 1000,
                            height: int = 800,
                            marker_size: int = 5):
    """Plotly로 XY 평면 정점 인덱스를 간단히 확인한다."""
    vertices = np.asarray(road_positions, dtype=np.float32)
    if len(vertices) == 0:
        raise ValueError("road_positions가 비어 있습니다.")

    x_vals = vertices[:, 0]
    y_vals = vertices[:, 1]
    indices = list(range(len(vertices)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="markers",
        marker=dict(size=marker_size, color="blue"),
        text=indices,
        hovertemplate="<b>Index: %{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
    ))

    # 한글 주석: hover로 인덱스를 바로 확인할 수 있게 closest 모드 사용
    fig.update_layout(
        title="도로 점 확인용 지도 (X-Y 평면)",
        xaxis_title="X Axis (East/West)",
        yaxis_title="Y Axis (North/South)",
        width=width,
        height=height,
        hovermode="closest",
    )
    fig.show()

def plot_vertices_xy(road_positions,
                     waypoints=None,
                     start_point=None,
                     end_point=None,
                     figsize=(7, 6)):
    """정점과 중심선 후보를 XY 평면에서 시각화한다."""
    vertices = np.asarray(road_positions, dtype=np.float32)
    if len(vertices) == 0:
        raise ValueError("road_positions가 비어 있습니다.")

    fig, ax = plt.subplots(figsize=figsize)

    # 한글 주석: 도로 정점 분포를 먼저 배경으로 표시
    ax.scatter(vertices[:, 0], vertices[:, 1], s=12, c="tab:blue", alpha=0.55, label="road vertices")

    if waypoints is not None and len(waypoints) > 0:
        wp = np.asarray(waypoints, dtype=np.float32)
        ax.plot(wp[:, 0], wp[:, 1], "-o", c="tab:orange", ms=3, lw=1.5, label="centerline waypoints")

    if start_point is not None:
        sp = np.asarray(start_point, dtype=np.float32)
        if sp.ndim == 0: # 정점 인덱스가 하나만 들어온 경우
            sp = vertices[int(sp)]
        ax.scatter([sp[0]], [sp[1]], s=150, c="tab:green", marker="*", label="start", zorder=3)

    if end_point is not None:
        ep = np.asarray(end_point, dtype=np.float32)
        if ep.ndim == 0: # 정점 인덱스가 하나만 들어온 경우
            ep = vertices[int(ep)]
        ax.scatter([ep[0]], [ep[1]], s=150, c="tab:red", marker="*", label="end", zorder=3)

    ax.set_title("Road vertices / selected path (X-Y)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()

def print_scene_materials(scene) -> None:
    """씬의 객체별 재질과 정의된 재질 목록을 출력한다."""
    if scene is None:
        raise ValueError("scene이 None입니다.")

    print("\n" + "=" * 60)
    print(f"{'Object Name':<30} | {'Assigned Material':<20}")
    print("=" * 60)

    # 한글 주석: 씬의 모든 객체를 순회하며 할당된 재질 이름을 출력
    for name, obj in scene.objects.items():
        mat_name = obj.radio_material.name if obj.radio_material else "None"
        print(f"{name:<30} | {mat_name:<20}")

    print("=" * 60)
    print("\n[정의된 재질 목록]")

    # 한글 주석: scene에 등록된 라디오 재질의 핵심 정보만 출력
    for mat_name, mat in scene.radio_materials.items():
        itu_type = getattr(mat, "itu_type", "Unknown")
        thickness = getattr(mat, "thickness", "Unknown")
        print(f" - 이름: {mat_name:<15} (Type: {itu_type}, Thickness: {thickness})")

def get_road_positions_from_object(scene,
                                   road_object_id: str,
                                   dtype=np.float32,
                                   round_decimals: int | None = None):
    """Mitsuba shape에서 vertex_positions를 읽어 도로 정점 좌표를 반환한다."""
    road_positions = []

    print(f"[탐색] Mitsuba Scene 내부에서 '{road_object_id}' 형상을 찾습니다...")

    if not hasattr(scene, "mi_scene"):
        print("[오류] scene 객체에서 'mi_scene' 속성을 찾을 수 없습니다.")
        return road_positions

    mi_scene = scene.mi_scene

    # 한글 주석: Mitsuba Scene의 모든 Shape를 순회하며 ID를 매칭
    target_shape = None
    for shape in mi_scene.shapes():
        if shape.id() == road_object_id:
            target_shape = shape
            break

    if target_shape is None:
        print(f"[실패] ID가 '{road_object_id}'인 Shape를 mi_scene에서 찾을 수 없습니다.")
        return road_positions

    try:
        params = mi.traverse(target_shape)

        if "vertex_positions" not in params:
            print(f"[오류] '{road_object_id}' 객체에 'vertex_positions' 속성이 없습니다.")
            return road_positions

        vertex_buffer = params["vertex_positions"]

        # 한글 주석: 버퍼를 NumPy로 변환한 뒤 (N, 3) 좌표로 변환
        vertices_flat = np.array(vertex_buffer)
        if len(vertices_flat) == 0:
            print("  [경고] 버퍼가 비어있습니다.")
            return road_positions

        road_positions = vertices_flat.reshape(-1, 3).astype(dtype)

        # 한글 주석: 소수점 자릿수 반올림은 float32에서 수행해 오버플로우를 줄인다
        if round_decimals is not None:
            rounded = np.round(road_positions.astype(np.float32), round_decimals)
            road_positions = rounded.astype(dtype)

        print(f"  -> 추출된 도로 좌표 수: {len(road_positions)}개 (dtype={road_positions.dtype})")
        return road_positions

    except Exception as exc:
        print(f"[오류] Vertex 추출 중 에러 발생: {exc}")
        return road_positions

def prepare_vehicle_path(road_positions,
                         path_indices,
                         z_offset: float = 1.5,
                         speed_ms: float = 90.0 / 3.6,
                         delta_t: float = 0.5):
    """수동 인덱스(path_indices) 기반으로 이동 경로와 시간축을 만든다."""
    if road_positions is None or len(road_positions) == 0:
        raise ValueError("❌ 'road_positions' 데이터가 없습니다.")
    if path_indices is None or len(path_indices) < 2:
        raise ValueError("❌ path_indices는 최소 2개 이상 필요합니다.")

    waypoints = np.asarray(road_positions[path_indices], dtype=np.float32).copy()
    waypoints[:, 2] += float(z_offset)

    # 한글 주석: 구간 거리/누적 거리/시간축 계산
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

    # 한글 주석: 진행 방향 단위벡터 * 속도크기 = 속도벡터
    direction = (waypoints[idx + 1] - waypoints[idx]) / seg_len
    vel = direction * speed_ms
    return pos.astype(np.float32), vel.astype(np.float32)


def setup_scene_for_vehicle_path(scene,
                                 path_data,
                                 tx_positions=None,
                                 tx_names=None,
                                 tx_power_dbm: float = 43.0,
                                 display_radius: float = 10.0,
                                 add_vehicle: bool = False,
                                 vehicle_name: str = "rx_vehicle",
                                 vehicle_scale=(2.5, 2.5, 2.5),
                                 vehicle_material_name: str = "car_material_rt",
                                 vehicle_z: float = 2.5,
                                 rx_z: float = 3.0,
                                 vehicle_rot_offset: float = 90.0):
    """이동 경로 시뮬레이션용 Tx/Rx(+옵션 자동차 Mesh)와 solver를 초기화한다."""
    if hasattr(scene, "transmitters"):
        for name in list(scene.transmitters.keys()):
            scene.remove(name)
    if hasattr(scene, "receivers"):
        for name in list(scene.receivers.keys()):
            scene.remove(name)

    # 한글 주석: 기존 차량 Mesh가 있으면 먼저 제거
    if add_vehicle and hasattr(scene, "objects") and vehicle_name in scene.objects:
        scene.edit(remove=vehicle_name)

    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="VH")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="VH")

    if tx_positions is None:
        tx_positions = [[-12.176, 25.0, 84.599]]
    if tx_names is None:
        tx_names = [f"Tx_{i+1}" for i in range(len(tx_positions))]

    for i, pos in enumerate(tx_positions):
        tx = Transmitter(
            name=tx_names[i],
            position=pos,
            power_dbm=tx_power_dbm,
            velocity=[0.0, 0.0, 0.0],
        )
        tx.transmit_antenna = scene.tx_array
        tx.look_at([0, 0, 0])
        scene.add(tx)

    start_pos, start_vel = get_state_at_time(path_data, 0.0)
    rx_start = np.array([float(start_pos[0]), float(start_pos[1]), float(rx_z)], dtype=np.float32)

    rx = Receiver(name="rx_car", position=rx_start, velocity=start_vel, display_radius=display_radius)
    rx.receive_antenna = scene.rx_array
    scene.add(rx)

    vehicle = None
    if add_vehicle:
        import sionna.rt
        from sionna.rt import ITURadioMaterial, SceneObject

        # 한글 주석: 차량 재질이 없으면 생성, 있으면 재사용
        if vehicle_material_name in scene.radio_materials:
            car_material = scene.radio_materials[vehicle_material_name]
        else:
            car_material = ITURadioMaterial(
                name=vehicle_material_name,
                itu_type="metal",
                thickness=0.01,
                color=(0.85, 0.15, 0.15),
            )
            scene.add(car_material)

        vehicle = SceneObject(
            fname=sionna.rt.scene.low_poly_car,
            name=vehicle_name,
            radio_material=car_material,
        )
        scene.edit(add=[vehicle])
        vehicle.scaling = mi.Vector3f(float(vehicle_scale[0]), float(vehicle_scale[1]), float(vehicle_scale[2]))
        vehicle.position = mi.Vector3f(float(start_pos[0]), float(start_pos[1]), float(vehicle_z))
        vehicle.orientation = mi.Point3f(np.pi/2, 0.0, 0.0) # 90도 회전 (np.pi/2)

    solver = PathSolver()
    return tx_names, rx, solver, vehicle


def create_vehicle_simulation_widgets(scene,
                                      path_data,
                                      tx_names,
                                      rx,
                                      solver,
                                      vehicle=None,
                                      vehicle_z: float = 2.5,
                                      rx_z: float = 3.0,
                                      max_depth: int = 3,
                                      max_num_paths_per_src: int = 10,
                                      samples_per_src: int = 100000,
                                      diffuse_reflection: bool = True,
                                      diffraction: bool = True,
                                      synthetic_array: bool = True,
                                      resolution=(800, 600)):
    """슬라이더 기반 인터랙티브 시뮬레이션 위젯을 생성한다."""
    output_widget = widgets.Output()
    time_steps = path_data["time_steps"]

    def update_simulation(frame_idx):
        t = float(time_steps[frame_idx])
        current_pos, current_vel = get_state_at_time(path_data, t)

        # 한글 주석: Rx와 차량 Mesh를 같은 XY로 동기 이동
        rx.position = np.array([float(current_pos[0]), float(current_pos[1]), float(rx_z)], dtype=np.float32)
        rx.velocity = current_vel
        if vehicle is not None:
            vehicle.position = mi.Vector3f(float(current_pos[0]), float(current_pos[1]), float(vehicle_z))

        for name in tx_names:
            scene.transmitters[name].look_at(current_pos)

        paths = solver(
            scene,
            max_depth=max_depth,
            max_num_paths_per_src=max_num_paths_per_src,
            samples_per_src=samples_per_src,
            diffuse_reflection=diffuse_reflection,
            diffraction=diffraction,
            synthetic_array=synthetic_array,
        )

        with output_widget:
            output_widget.clear_output(wait=True)
            scene.preview(paths=paths, show_devices=True, resolution=list(resolution))

            a0, _ = paths.cir(out_type="tf", normalize_delays=False)
            p_val = tf.reduce_sum(tf.abs(a0) ** 2) if tf.size(a0) > 0 else 0.0
            db_val = 10 * np.log10(float(p_val) + 1e-30)

            print(
                f"⏱ Time: {t:.1f}s | 📍 Pos: {np.round(current_pos, 3)} "
                f"| 🚗 Vel: {np.round(current_vel, 3)} m/s | 📶 Power: {db_val:.2f} dB"
            )

    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(time_steps) - 1,
        step=1,
        description="Time Step:",
        layout=widgets.Layout(width="600px"),
    )

    widgets.interactive_output(update_simulation, {"frame_idx": slider})
    return slider, output_widget


def plot_cir_for_frame(scene,
                       rx,
                       solver,
                       path_data,
                       tx_names,
                       frame_idx: int,
                       max_depth: int = 3,
                       samples_per_src: int = 100000,
                       diffuse_reflection: bool = True,
                       diffraction: bool = True,
                       synthetic_array: bool = True,
                       show_plot: bool = True,
                       figsize=(8, 4)):
    """프레임 인덱스(frame_idx) 기준으로 CIR을 계산하고(선택적으로) 시각화한다."""
    time_steps = path_data["time_steps"]
    if frame_idx < 0 or frame_idx >= len(time_steps):
        raise IndexError(f"frame_idx 범위 오류: {frame_idx} (0 ~ {len(time_steps)-1})")

    t = float(time_steps[frame_idx])
    current_pos, current_vel = get_state_at_time(path_data, t)

    # 한글 주석: 위치/속도를 매 프레임 업데이트해 도플러를 반영
    rx.position = current_pos
    rx.velocity = current_vel
    for name in tx_names:
        scene.transmitters[name].look_at(current_pos)

    # 한글 주석: 현재 프레임 기준 경로를 계산
    paths = solver(
        scene,
        max_depth=max_depth,
        samples_per_src=samples_per_src,
        diffuse_reflection=diffuse_reflection,
        diffraction=diffraction,
        synthetic_array=synthetic_array
    )

    # 한글 주석: CIR 텐서를 가져오고 지연축/진폭축을 플롯용으로 변환
    a, tau = paths.cir(out_type="tf", normalize_delays=False)

    a_np = a.numpy() if hasattr(a, "numpy") else np.asarray(a)
    tau_np = tau.numpy() if hasattr(tau, "numpy") else np.asarray(tau)

    if a_np.size == 0 or tau_np.size == 0:
        raise ValueError("CIR 결과가 비어 있습니다. (a 또는 tau가 empty)")

    # 한글 주석: 버전에 따라 tau 차원이 다를 수 있어 분기 처리
    if tau_np.ndim == 3:
        tau_ns = tau_np[0, 0, :] / 1e-9
    elif tau_np.ndim == 5:
        tau_ns = tau_np[0, 0, 0, 0, :] / 1e-9
    else:
        raise ValueError(f"지원하지 않는 tau 차원: {tau_np.shape}")

    # 한글 주석: 기존 코드 인덱싱과 동일하게 기본 링크(Rx0/RxAnt0/Tx0/TxAnt0) 사용
    if a_np.ndim == 6:
        a_abs = np.abs(a_np)[0, 0, 0, 0, :, 0]
    elif a_np.ndim == 5:
        a_abs = np.abs(a_np)[0, 0, 0, 0, :]
    else:
        raise ValueError(f"지원하지 않는 a 차원: {a_np.shape}")

    if show_plot:
        plt.figure(figsize=figsize)
        plt.title(f"Channel Impulse Response (frame={frame_idx})")
        plt.stem(tau_ns, a_abs, basefmt=" ")
        plt.xlim(left=0)
        plt.xlabel("Delay τ (ns)")
        plt.ylabel("|a|")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

    print("frame_idx =", frame_idx)
    print("t =", t)
    print("a shape:", a_np.shape)
    print("tau shape:", tau_np.shape)

    return {
        "frame_idx": frame_idx,
        "time_s": t,
        "pos": current_pos,
        "vel": current_vel,
        "paths": paths,
        "a": a,
        "tau": tau,
        "tau_ns": tau_ns,
        "a_abs": a_abs,
    }

def create_cir_slider_widget(scene,
                             rx,
                             solver,
                             path_data,
                             tx_names,
                             max_depth: int = 3,
                             samples_per_src: int = 100000,
                             diffuse_reflection: bool = True,
                             diffraction: bool = True,
                             synthetic_array: bool = True,
                             figsize=(8, 4),
                             continuous_update: bool = False):
    """프레임 인덱스 슬라이더로 CIR 변화를 확인하는 위젯을 생성한다."""
    time_steps = path_data["time_steps"]
    out = widgets.Output()

    frame_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(time_steps) - 1,
        step=1,
        description="Frame:",
        continuous_update=continuous_update,
        layout=widgets.Layout(width="600px"),
    )

    def _update(frame_idx):
        with out:
            out.clear_output(wait=True)

            # 한글 주석: 기존 단일 프레임 함수 호출로 계산/플롯 로직을 재사용
            result = plot_cir_for_frame(
                scene=scene,
                rx=rx,
                solver=solver,
                path_data=path_data,
                tx_names=tx_names,
                frame_idx=int(frame_idx),
                max_depth=max_depth,
                samples_per_src=samples_per_src,
                diffuse_reflection=diffuse_reflection,
                diffraction=diffraction,
                synthetic_array=synthetic_array,
                show_plot=True,
                figsize=figsize,
            )

            # 한글 주석: 슬라이더 상태에서 핵심 정보만 추가 출력
            print(
                f"⏱ Time: {result['time_s']:.2f}s | frame={result['frame_idx']} "
                f"| 📍 Pos: {np.round(result['pos'], 3)} | 🚗 Vel: {np.round(result['vel'], 3)}"
            )

    widgets.interactive_output(_update, {"frame_idx": frame_slider})

    # 한글 주석: 초기 프레임을 먼저  번 렌더링
    _update(0)

    return widgets.VBox([frame_slider, out])


def plot_cfr_for_frame(scene,
                       rx,
                       solver,
                       path_data,
                       tx_names,
                       frame_idx: int,
                       frequencies,
                       normalize: bool = True,
                       normalize_delays: bool = True,
                       max_depth: int = 3,
                       samples_per_src: int = 100000,
                       diffuse_reflection: bool = True,
                       diffraction: bool = True,
                       synthetic_array: bool = True,
                       show_plot: bool = True,
                       figsize=(8, 3.2),
                       rx_idx: int = 0,
                       rx_ant_idx: int = 0,
                       tx_idx: int = 0,
                       tx_ant_idx: int = 0,
                       time_idx: int = 0):
    """프레임 인덱스(frame_idx) 기준으로 CFR을 계산하고(선택적으로) 시각화한다."""
    time_steps = path_data["time_steps"]
    if frame_idx < 0 or frame_idx >= len(time_steps):
        raise IndexError(f"frame_idx 범위 오류: {frame_idx} (0 ~ {len(time_steps)-1})")

    freqs = np.asarray(frequencies, dtype=np.float32).reshape(-1)
    if freqs.size == 0:
        raise ValueError("frequencies가 비어 있습니다.")

    t = float(time_steps[frame_idx])
    current_pos, current_vel = get_state_at_time(path_data, t)

    # 한글 주석: 프레임별 위치/속도를 갱신해 도플러를 반영한다.
    rx.position = current_pos
    rx.velocity = current_vel
    for name in tx_names:
        scene.transmitters[name].look_at(current_pos)

    # 한글 주석: 현재 프레임의 경로를 다시 계산한다.
    paths = solver(
        scene,
        max_depth=max_depth,
        samples_per_src=samples_per_src,
        diffuse_reflection=diffuse_reflection,
        diffraction=diffraction,
        synthetic_array=synthetic_array
    )

    h_freq = paths.cfr(
        frequencies=freqs,
        normalize=normalize,
        normalize_delays=normalize_delays,
        out_type="numpy"
    )

    h_np = np.asarray(h_freq)
    if h_np.size == 0:
        raise ValueError("CFR 결과가 비어 있습니다.")

    # 한글 주석: 기본 링크 하나를 선택해 주파수 축 벡터를 뽑는다.
    h_sel = h_np[rx_idx, rx_ant_idx, tx_idx, tx_ant_idx, time_idx, :]
    h_abs = np.abs(h_sel)

    if show_plot:
        plt.figure(figsize=figsize)

        if freqs.size == 1:
            # 한글 주석: 단일 주파수는 막대 하나로 표시한다.
            plt.stem([0.0], [h_abs[0]], basefmt=" ")
            plt.xlabel("Frequency offset from center [MHz]")
            plt.ylabel("|H(f)|")
            plt.title(f"CFR (single frequency) | frame={frame_idx}")
            plt.grid(True)
        else:
            f0 = float(np.mean(freqs))
            f_off_mhz = (freqs - f0) / 1e6
            plt.plot(f_off_mhz, h_abs)
            plt.xlabel(f"Frequency offset from {f0/1e9:.3f}GHz [MHz]")
            plt.ylabel("|H(f)|")
            plt.title(f"CFR (frequency sweep) | frame={frame_idx}")
            plt.grid(True)

        plt.tight_layout()
        plt.show()

    print("frame_idx =", frame_idx)
    print("t =", t)
    print("h_freq shape:", h_np.shape)
    print("selected |H(f)| first:", float(h_abs[0]))

    return {
        "frame_idx": frame_idx,
        "time_s": t,
        "pos": current_pos,
        "vel": current_vel,
        "paths": paths,
        "frequencies": freqs,
        "h_freq": h_freq,
        "h_abs": h_abs,
    }


def create_cfr_slider_widget(scene,
                             rx,
                             solver,
                             path_data,
                             tx_names,
                             frequencies,
                             normalize: bool = True,
                             normalize_delays: bool = True,
                             max_depth: int = 3,
                             samples_per_src: int = 100000,
                             diffuse_reflection: bool = True,
                             diffraction: bool = True,
                             synthetic_array: bool = True,
                             figsize=(8, 3.2),
                             continuous_update: bool = False,
                             rx_idx: int = 0,
                             rx_ant_idx: int = 0,
                             tx_idx: int = 0,
                             tx_ant_idx: int = 0,
                             time_idx: int = 0):
    """프레임 슬라이더로 CFR 변화를 확인하는 위젯을 생성한다."""
    time_steps = path_data["time_steps"]
    out = widgets.Output()

    frame_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(time_steps) - 1,
        step=1,
        description="Frame:",
        continuous_update=continuous_update,
        layout=widgets.Layout(width="600px")
    )

    def _update(frame_idx):
        with out:
            out.clear_output(wait=True)

            # 한글 주석: 단일 프레임 CFR 함수를 재사용해 중복 코드를 줄인다.
            result = plot_cfr_for_frame(
                scene=scene,
                rx=rx,
                solver=solver,
                path_data=path_data,
                tx_names=tx_names,
                frame_idx=int(frame_idx),
                frequencies=frequencies,
                normalize=normalize,
                normalize_delays=normalize_delays,
                max_depth=max_depth,
                samples_per_src=samples_per_src,
                diffuse_reflection=diffuse_reflection,
                diffraction=diffraction,
                synthetic_array=synthetic_array,
                show_plot=True,
                figsize=figsize,
                rx_idx=rx_idx,
                rx_ant_idx=rx_ant_idx,
                tx_idx=tx_idx,
                tx_ant_idx=tx_ant_idx,
                time_idx=time_idx
            )

            print(
                f"⏱ Time: {result['time_s']:.2f}s | frame={result['frame_idx']} "
                f"| 📍 Pos: {np.round(result['pos'], 3)} | 🚗 Vel: {np.round(result['vel'], 3)}"
            )

    widgets.interactive_output(_update, {"frame_idx": frame_slider})

    # 한글 주석: 위젯 생성 직후 첫 프레임을 한 번 렌더링한다.
    _update(0)

    return widgets.VBox([frame_slider, out])


def create_taps_pdp_widgets(scene, rx, solver, path_data, tx_names,
                            bandwidth, l_min, l_max, sampling_frequency, num_time_steps,
                            bandwidth_min: float = 1e6,
                            bandwidth_max: float = 2e9,
                            sampling_frequency_min: float = 1e3,
                            sampling_frequency_max: float = 1e7,
                            num_time_steps_min: int = 1,
                            num_time_steps_max: int = 200,
                            l_step_min: int = 0,
                            l_step_max: int = 2500):
    """
    Taps 기반 PDP를 프레임/파라미터 슬라이더로 시각화하는 인터랙티브 위젯을 생성합니다.
    """
    time_steps = path_data["time_steps"]

    # 캐시: (frame_idx, tx_idx, rel_delay, bw, lmin, lmax, fs, nts) -> (t, pos, df)
    _taps_cache = {}

    out = widgets.Output()

    tx_dropdown = widgets.Dropdown(
        options=[(name, i) for i, name in enumerate(tx_names)],
        value=0,
        description="TX:",
        layout=widgets.Layout(width="220px")
    )

    frame_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(time_steps) - 1,
        step=1,
        description="Time:",
        continuous_update=False,
        layout=widgets.Layout(width="600px")
    )

    rel_delay_chk = widgets.Checkbox(
        value=False,
        description="Relative delay (min τ = 0)",
        indent=False
    )

    # 한글 주석: taps 파라미터를 슬라이더로 조절할 수 있게 구성
    bandwidth_slider = widgets.FloatLogSlider(
        value=float(bandwidth),
        base=10,
        min=np.log10(max(bandwidth_min, 1.0)),
        max=np.log10(max(bandwidth_max, bandwidth_min * 10.0)),
        step=0.01,
        description="BW [Hz]:",
        continuous_update=False,
        readout_format=".3e",
        layout=widgets.Layout(width="600px")
    )

    l_min_slider = widgets.IntSlider(
        value=int(l_min),
        min=int(l_step_min),
        max=int(l_step_max),
        step=1,
        description="l_min:",
        continuous_update=False,
        layout=widgets.Layout(width="600px")
    )

    l_max_slider = widgets.IntSlider(
        value=int(l_max),
        min=int(l_step_min),
        max=int(l_step_max),
        step=1,
        description="l_max:",
        continuous_update=False,
        layout=widgets.Layout(width="600px")
    )

    sampling_frequency_slider = widgets.FloatLogSlider(
        value=float(sampling_frequency),
        base=10,
        min=np.log10(max(sampling_frequency_min, 1.0)),
        max=np.log10(max(sampling_frequency_max, sampling_frequency_min * 10.0)),
        step=0.01,
        description="Fs [Hz]:",
        continuous_update=False,
        readout_format=".3e",
        layout=widgets.Layout(width="600px")
    )

    num_time_steps_slider = widgets.IntSlider(
        value=int(num_time_steps),
        min=int(num_time_steps_min),
        max=int(num_time_steps_max),
        step=1,
        description="N_time:",
        continuous_update=False,
        layout=widgets.Layout(width="600px")
    )

    def _compute_taps_for_frame(frame_idx: int,
                                tx_idx: int,
                                rel_delay: bool,
                                bw: float,
                                lmin: int,
                                lmax: int,
                                fs: float,
                                nts: int):
        key = (frame_idx, tx_idx, rel_delay, float(bw), int(lmin), int(lmax), float(fs), int(nts))
        if key in _taps_cache:
            return _taps_cache[key]

        t = float(time_steps[frame_idx])
        pos, vel = get_state_at_time(path_data, t)

        # 한글 주석: Rx 위치/속도를 프레임별로 업데이트
        rx.position = pos
        rx.velocity = vel
        for name in tx_names:
            scene.transmitters[name].look_at(pos)

        # 경로 계산
        paths = solver(
            scene,
            max_depth=3,
            samples_per_src=100000,
            diffuse_reflection=True,
            diffraction=True,
            synthetic_array=True
        )

        # taps 계산 (TF 텐서)
        h = paths.taps(
            bandwidth=float(bw),
            l_min=int(lmin),
            l_max=int(lmax),
            sampling_frequency=float(fs),
            num_time_steps=int(nts),
            normalize=False,
            normalize_delays=False,
            out_type="tf"
        )

        # Rx=0, 선택 Tx, time=0에서 안테나 축 합산해 PDP 생성
        h_sel = h[0, :, tx_idx, :, 0, :]
        pdp = tf.reduce_sum(tf.abs(h_sel) ** 2, axis=[0, 1])
        pdp_np = pdp.numpy()

        tap_idx = np.arange(int(lmin), int(lmax) + 1, dtype=int)
        tau_ns = (tap_idx / float(bw)) * 1e9

        valid = np.isfinite(pdp_np) & (pdp_np > 0)
        if np.sum(valid) == 0:
            df = pd.DataFrame(columns=["tap_idx", "tau_ns", "pdp_db"])
            _taps_cache[key] = (t, pos, df)
            return _taps_cache[key]

        tap_v = tap_idx[valid]
        tau_v = tau_ns[valid]
        pdp_v = pdp_np[valid]

        if rel_delay:
            tau_v = tau_v - np.min(tau_v)

        order = np.argsort(tau_v)
        tap_s = tap_v[order]
        tau_s = tau_v[order]
        pdp_db = 10 * np.log10(pdp_v[order] + 1e-30)

        df = pd.DataFrame({
            "tap_idx": tap_s.astype(int),
            "tau_ns": tau_s,
            "pdp_db": pdp_db
        })

        _taps_cache[key] = (t, pos, df)
        return _taps_cache[key]

    def _update_plot(_=None):
        frame_idx = frame_slider.value
        tx_idx = tx_dropdown.value
        rel_delay = rel_delay_chk.value

        bw = float(bandwidth_slider.value)
        lmin = int(l_min_slider.value)
        lmax = int(l_max_slider.value)
        fs = float(sampling_frequency_slider.value)
        nts = int(num_time_steps_slider.value)

        # 한글 주석: 슬라이더 조작 중 l_min/l_max 순서가 바뀌면 자동 보정
        if lmin > lmax:
            lmin, lmax = lmax, lmin

        with out:
            out.clear_output(wait=True)

            t, pos, df = _compute_taps_for_frame(
                frame_idx=frame_idx,
                tx_idx=tx_idx,
                rel_delay=rel_delay,
                bw=bw,
                lmin=lmin,
                lmax=lmax,
                fs=fs,
                nts=nts
            )

            if df.empty:
                print(f"t={t:.2f}s | frame={frame_idx} | {tx_names[tx_idx]} : 유효 탭이 없습니다.")
                print(f"pos={pos}")
                print(f"BW={bw:.3e}, l_min={lmin}, l_max={lmax}, Fs={fs:.3e}, N_time={nts}")
                return

            plt.figure(figsize=(8, 3.6))
            plt.stem(df["tau_ns"].values, df["pdp_db"].values, basefmt=" ")
            for x, y, tap in zip(df["tau_ns"].values, df["pdp_db"].values, df["tap_idx"].values):
                plt.text(x, y, str(tap), fontsize=9, ha="center", va="bottom")

            plt.xlabel("Delay τ (ns)")
            plt.ylabel("Power (dB)")
            plt.title(
                f"{tx_names[tx_idx]} taps-PDP at t={t:.2f}s | frame={frame_idx}\n"
                f"BW={bw:.3e}, l_min={lmin}, l_max={lmax}, Fs={fs:.3e}, N_time={nts}\npos={pos}"
            )
            plt.grid(True)
            plt.show()

            print("\n[ Taps Data Table ]")
            display(df)

    # 위젯 이벤트 연결
    frame_slider.observe(_update_plot, names="value")
    tx_dropdown.observe(_update_plot, names="value")
    rel_delay_chk.observe(_update_plot, names="value")
    bandwidth_slider.observe(_update_plot, names="value")
    l_min_slider.observe(_update_plot, names="value")
    l_max_slider.observe(_update_plot, names="value")
    sampling_frequency_slider.observe(_update_plot, names="value")
    num_time_steps_slider.observe(_update_plot, names="value")

    _update_plot()
    return widgets.VBox([
        widgets.HBox([frame_slider, tx_dropdown]),
        rel_delay_chk,
        bandwidth_slider,
        widgets.HBox([l_min_slider, l_max_slider]),
        sampling_frequency_slider,
        num_time_steps_slider,
        out
    ])


def export_simulation_video(scene, rx, solver, path_data, tx_names, camera,
                            vehicle=None,
                            vehicle_z: float = 2.5,
                            rx_z: float = 3.0,
                            filename="simulation_output.mp4",
                            fps=10,
                            resolution=(800, 600),
                            max_depth=3,
                            samples_per_src=100000,
                            white_background: bool = True,
                            save_frames_dir: str = None):
    """
    모든 프레임에 대해 시뮬레이션을 돌리고 고정 카메라 화면을 mp4로 저장합니다.
    white_background=True이면 RGBA 프레임을 흰 배경으로 합성해 저장합니다.
    save_frames_dir가 주어지면 해당 폴더에 매 프레임을 .jpg 파일로 함께 저장합니다.
    """
    import imageio

    time_steps = path_data["time_steps"]
    total_frames = len(time_steps)
    print(f"🎥 애니메이션 렌더링 시작... (총 {total_frames} 프레임)")
    print(f"저장 경로: {filename}")

    writer = imageio.get_writer(filename, fps=fps)

    for frame_idx, t in enumerate(time_steps):
        t_float = float(t)
        current_pos, current_vel = get_state_at_time(path_data, t_float)

        # 한글 주석: Rx 위치/속도 갱신
        rx.position = np.array([float(current_pos[0]), float(current_pos[1]), float(rx_z)], dtype=np.float32)
        rx.velocity = current_vel

        # 한글 주석: 차량 Mesh가 있으면 Rx와 같은 XY로 동기 이동
        if vehicle is not None:
            vehicle.position = mi.Vector3f(float(current_pos[0]), float(current_pos[1]), float(vehicle_z))

        for name in tx_names:
            scene.transmitters[name].look_at(current_pos)

        paths = solver(
            scene,
            max_depth=max_depth,
            samples_per_src=samples_per_src,
            diffuse_reflection=True,
            diffraction=True,
            synthetic_array=True
        )

        try:
            # 한글 주석: 렌더링 결과를 numpy로 변환
            img_bitmap = scene.render(
                camera=camera,
                paths=paths,
                resolution=resolution,
                show_devices=True,
                return_bitmap=True
            )
            img = np.array(img_bitmap, dtype=np.float32)

            # 한글 주석: RGBA면 알파를 흰 배경에 합성해서 검은 바닥 문제를 제거
            if img.ndim == 3 and img.shape[-1] == 4:
                rgb = img[..., :3]
                alpha = img[..., 3:4]
                if white_background:
                    bg = np.ones_like(rgb, dtype=np.float32)  # 흰 배경
                    rgb = rgb * alpha + bg * (1.0 - alpha)
                img_out = rgb
            elif img.ndim == 3 and img.shape[-1] >= 3:
                img_out = img[..., :3]
            else:
                # 한글 주석: 비정상 shape 방어 처리
                img_out = np.stack([img, img, img], axis=-1) if img.ndim == 2 else img

            img_uint8 = np.clip(img_out * 255.0, 0, 255).astype(np.uint8)
            writer.append_data(img_uint8)

            if save_frames_dir is not None:
                import os
                os.makedirs(save_frames_dir, exist_ok=True)
                frame_path = os.path.join(save_frames_dir, f"frame_{frame_idx:05d}.jpg")
                imageio.imwrite(frame_path, img_uint8)

            print(f"  -> 프레임 렌더링 완료: {frame_idx + 1}/{total_frames} ({(frame_idx+1)/total_frames*100:.1f}%)")
        except Exception as e:
            print(f"프레임 {frame_idx} 렌더링 중 에러 발생: {e}")
            break

    writer.close()

    # 한글 주석: recorder_cam이 있을 때만 제거
    if hasattr(scene, "cameras") and ("recorder_cam" in scene.cameras):
        scene.remove("recorder_cam")

    print(f"🎬 렌더링 완료! '{filename}' 파일이 생성되었습니다.")