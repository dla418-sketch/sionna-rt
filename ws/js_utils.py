# File: jinsup/js_utils.py

"""path_loss.ipynb용 재질/정점/이동 시뮬레이션 유틸."""

import numpy as np
import mitsuba as mi
import tensorflow as tf
import ipywidgets as widgets
import matplotlib.pyplot as plt

from sionna.rt import PlanarArray, Transmitter, Receiver, PathSolver

def build_centerline_waypoints(road_positions,
                               num_points: int = 40,
                               start_index: int | None = None,
                               end_index: int | None = None):
    """도로 정점 클라우드에서 주행용 중심선 waypoints를 만든다."""
    vertices = np.asarray(road_positions, dtype=np.float32)
    if len(vertices) < 2:
        return vertices.copy()

    xy = vertices[:, [0, 1]]
    xy_centered = xy - np.mean(xy, axis=0, keepdims=True)

    # 한글 주석: 주행 방향 추정을 위해 XY 평면의 주성분 축을 사용
    _, _, vt = np.linalg.svd(xy_centered, full_matrices=False)
    forward_axis = vt[0]
    proj = xy_centered @ forward_axis

    lo, hi = float(np.min(proj)), float(np.max(proj))
    if abs(hi - lo) < 1e-6:
        waypoints = vertices.copy()
    else:
        bins = np.linspace(lo, hi, int(num_points) + 1, dtype=np.float32)
        centers = []

        # 한글 주석: 축 방향 구간별 평균점을 잡아 중심선 형태로 정렬
        for i in range(len(bins) - 1):
            mask = (proj >= bins[i]) & (proj < bins[i + 1])
            if not np.any(mask):
                continue
            centers.append(np.mean(vertices[mask], axis=0))

        waypoints = np.asarray(centers, dtype=np.float32)
        if len(waypoints) < 2:
            order = np.argsort(proj)
            waypoints = vertices[order]

    # 한글 주석: 시작/끝 기준점이 있으면 XY 기준으로 가장 가까운 구간만 잘라서 사용
    start_ref = None
    end_ref = None
    if start_index is not None and len(vertices) > 0:
        start_ref = vertices[int(np.clip(start_index, 0, len(vertices) - 1))]

    if end_index is not None and len(vertices) > 0:
        end_ref = vertices[int(np.clip(end_index, 0, len(vertices) - 1))]

    if start_ref is not None or end_ref is not None:
        wp_xy = waypoints[:, [0, 1]]

        if start_ref is None:
            i_start = 0
        else:
            i_start = int(np.argmin(np.linalg.norm(wp_xy - start_ref[[0, 1]], axis=1)))

        if end_ref is None:
            i_end = len(waypoints) - 1
        else:
            i_end = int(np.argmin(np.linalg.norm(wp_xy - end_ref[[0, 1]], axis=1)))

        if i_start <= i_end:
            waypoints = waypoints[i_start:i_end + 1]
        else:
            waypoints = waypoints[i_end:i_start + 1][::-1]

    return waypoints

def plot_vertices_xy_with_indices(road_positions,
                                  step: int = 10,
                                  figsize=(8, 7)):
    """XY 평면에서 정점 인덱스를 간격 기반으로 함께 표시한다."""
    vertices = np.asarray(road_positions, dtype=np.float32)
    if len(vertices) == 0:
        raise ValueError("road_positions가 비어 있습니다.")

    fig, ax = plt.subplots(figsize=figsize)

    # 한글 주석: 전체 정점 분포를 점으로 그리고 일부 인덱스만 라벨링해 가독성 유지
    ax.scatter(vertices[:, 0], vertices[:, 1], s=12, c="tab:blue", alpha=0.55)

    step = max(1, int(step))
    for idx in range(0, len(vertices), step):
        ax.text(vertices[idx, 0], vertices[idx, 1], str(idx), fontsize=8, color="black")

    ax.set_title("Road vertices with sampled indices (X-Y)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

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
        ax.scatter([sp[0]], [sp[1]], s=90, c="tab:green", marker="*", label="start")

    if end_point is not None:
        ep = np.asarray(end_point, dtype=np.float32)
        ax.scatter([ep[0]], [ep[1]], s=90, c="tab:red", marker="*", label="end")

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
                         path_indices=None,
                         y_offset: float = 1.5,
                         speed_ms: float = 60.0 / 3.6,
                         delta_t: float = 0.5,
                         use_centerline: bool = False,
                         centerline_points: int = 40,
                         start_index: int | None = None,
                         end_index: int | None = None):
    """도로 정점에서 이동 경로와 시간축 정보를 만든다."""
    if road_positions is None or len(road_positions) == 0:
        raise ValueError("❌ 'road_positions' 데이터가 없습니다.")

    if use_centerline or path_indices is None:
        # 한글 주석: 중심선 생성 시 start/end 인덱스를 직접 반영한다
        waypoints = build_centerline_waypoints(
            road_positions,
            num_points=centerline_points,
            start_index=start_index,
            end_index=end_index,
        )
    else:
        waypoints = np.asarray(road_positions[path_indices], dtype=np.float32).copy()

    waypoints[:, 1] += float(y_offset)

    # 한글 주석: 구간 거리/누적 거리/시간축을 한 번에 계산
    diffs = waypoints[1:] - waypoints[:-1]
    segment_dists = np.linalg.norm(diffs, axis=1)
    cumulative_dists = np.concatenate(([0.0], np.cumsum(segment_dists)))
    total_distance = float(cumulative_dists[-1])
    total_time = total_distance / float(speed_ms) if speed_ms > 0 else 0.0
    time_steps = np.arange(0.0, total_time + delta_t, delta_t, dtype=np.float32)

    return {
        "waypoints": waypoints,
        "use_centerline": bool(use_centerline or path_indices is None),
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
                                 display_radius: float = 10.0):
    """이동 경로 시뮬레이션용 Tx/Rx와 solver를 초기화한다."""
    if hasattr(scene, "transmitters"):
        for name in list(scene.transmitters.keys()):
            scene.remove(name)
    if hasattr(scene, "receivers"):
        for name in list(scene.receivers.keys()):
            scene.remove(name)

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
    rx = Receiver(name="rx_car", position=start_pos, velocity=start_vel, display_radius=display_radius)
    rx.receive_antenna = scene.rx_array
    scene.add(rx)

    solver = PathSolver()
    return tx_names, rx, solver


def create_vehicle_simulation_widgets(scene,
                                      path_data,
                                      tx_names,
                                      rx,
                                      solver,
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

        # 한글 주석: 위치/속도를 매 프레임 업데이트해 도플러를 반영
        rx.position = current_pos
        rx.velocity = current_vel
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
