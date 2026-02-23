# File: jinsup/js_utils.py
# mi_utils.py
import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt


def _find_shape_by_id(scene, object_id: str):
    """Mitsuba Scene에서 ID로 Shape를 찾습니다."""
    mi_scene = scene.mi_scene
    for shape in mi_scene.shapes():
        if shape.id() == object_id:
            return shape
    return None


def get_mesh_vertices(scene, object_id: str) -> np.ndarray:
    """
    특정 객체의 메쉬 정점 좌표를 NumPy 배열로 추출합니다.

    Args:
        scene: Sionna Scene 객체
        object_id (str): 대상 객체의 ID

    Returns:
        np.ndarray: (N, 3) 형태의 좌표 배열 (없으면 빈 배열 반환)
    """
    target_shape = _find_shape_by_id(scene, object_id)

    if target_shape is None:
        print(f"[mi_utils] 오류: ID가 '{object_id}'인 객체를 찾을 수 없습니다.")
        return np.array([])

    params = mi.traverse(target_shape)

    if "vertex_positions" in params:
        # 한글 주석: 정점 버퍼를 (N, 3) 형태로 변환
        v_flat = params["vertex_positions"].numpy()
        vertices = v_flat.reshape(-1, 3)
        print(f"[mi_utils] '{object_id}'에서 {len(vertices)}개의 정점을 추출했습니다.")
        return vertices

    print(f"[mi_utils] 경고: '{object_id}'는 메쉬가 아니거나 정점 정보가 없습니다.")
    return np.array([])


def get_shape_translation(scene, object_id: str) -> np.ndarray:
    """
    to_world가 있으면 원점 좌표를, 없으면 정점 중심 좌표를 반환합니다.

    Returns:
        np.ndarray: shape 원점의 월드 좌표 [x, y, z] (못 찾으면 빈 배열)
    """
    target_shape = _find_shape_by_id(scene, object_id)
    if target_shape is None:
        print(f"[mi_utils] 오류: ID가 '{object_id}'인 객체를 찾을 수 없습니다.")
        return np.array([])

    params = mi.traverse(target_shape)

    # 한글 주석: 환경마다 to_world 키 노출 방식이 달라 suffix도 함께 탐색
    to_world = params.get("to_world")
    if to_world is None:
        for key in params.keys():
            if key.endswith("to_world"):
                to_world = params[key]
                break

    if to_world is not None:
        p_world = to_world @ mi.Point3f(0.0, 0.0, 0.0)
        return np.array([float(p_world.x), float(p_world.y), float(p_world.z)])

    vertices = get_mesh_vertices(scene, object_id)
    if vertices.size == 0:
        return np.array([])

    print(f"[mi_utils] 경고: '{object_id}'에서 to_world를 찾지 못해 중심 좌표를 반환합니다.")
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    return (mins + maxs) * 0.5


def get_shape_center(scene, object_id: str) -> np.ndarray:
    """정점 기반 AABB 중심 좌표를 반환합니다."""
    vertices = get_mesh_vertices(scene, object_id)
    if vertices.size == 0:
        return np.array([])

    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    return (mins + maxs) * 0.5


def get_object_coordinates(scene, object_id: str) -> dict:
    """
    객체 좌표 관련 정보를 한 번에 반환합니다.

    Returns:
        dict:
            - id
            - origin_xyz: to_world 기준 원점 좌표
            - center_xyz: 정점 AABB 중심 좌표
            - n_vertices: 정점 수
    """
    vertices = get_mesh_vertices(scene, object_id)
    origin = get_shape_translation(scene, object_id)
    center = np.array([])
    if vertices.size > 0:
        center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5

    return {
        "id": object_id,
        "origin_xyz": origin,
        "center_xyz": center,
        "n_vertices": int(vertices.shape[0]) if vertices.size > 0 else 0,
    }


def plot_vertices_xy_xz(vertices: np.ndarray, object_id: str = "object"):
    """
    정점 좌표를 색상 기반 XY, XZ 2개 투영으로 시각화합니다.

    Args:
        vertices (np.ndarray): (N, 3) 정점 배열
        object_id (str): 플롯 제목에 사용할 객체 이름
    """
    if vertices is None or len(vertices) == 0:
        print("❌ 시각화할 정점이 없습니다.")
        return

    v = np.asarray(vertices)

    # 한글 주석: 포인트 인덱스를 색상으로 매핑해 정점 분포를 확인
    colors = np.linspace(0.0, 1.0, len(v))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    xy = axes[0].scatter(v[:, 0], v[:, 1], c=colors, cmap="viridis", s=10, alpha=0.85)
    axes[0].set_title(f"{object_id} - X-Y projection")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].grid(True)
    fig.colorbar(xy, ax=axes[0], label="vertex index")

    xz = axes[1].scatter(v[:, 0], v[:, 2], c=colors, cmap="plasma", s=10, alpha=0.85)
    axes[1].set_title(f"{object_id} - X-Z projection")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Z")
    axes[1].grid(True)
    fig.colorbar(xz, ax=axes[1], label="vertex index")

    plt.tight_layout()
    plt.show()


def show_vertices_on_scene_preview(scene,
                                   vertices: np.ndarray,
                                   object_id: str = "object",
                                   point_radius: float = 0.6,
                                   cmap: str = "viridis",
                                   preview_kwargs: dict | None = None):
    """
    scene.preview 화면 위에 정점 포인트를 색상으로 오버레이합니다.

    Args:
        scene: Sionna Scene 객체
        vertices (np.ndarray): (N, 3) 정점 배열
        object_id (str): 출력 로그용 객체 이름
        point_radius (float): 포인트 반지름
        cmap (str): matplotlib 컬러맵 이름
        preview_kwargs (dict | None): scene.preview에 전달할 키워드 인자
    """
    if vertices is None or len(vertices) == 0:
        print("❌ 표시할 정점이 없습니다.")
        return

    v = np.asarray(vertices, dtype=np.float32)
    if v.ndim != 2 or v.shape[1] != 3:
        print("❌ vertices는 (N, 3) 형태여야 합니다.")
        return

    # 한글 주석: preview를 먼저 띄운 뒤 내부 위젯에 포인트를 추가한다
    if preview_kwargs is None:
        preview_kwargs = {}
    scene.preview(**preview_kwargs)

    widget = getattr(scene, "_preview_widget", None)
    if widget is None:
        print("❌ preview 위젯을 찾지 못했습니다.")
        return

    # 한글 주석: 정점 인덱스를 컬러맵에 매핑해 포인트 색상을 만든다
    color_map = plt.get_cmap(cmap)
    color_positions = np.linspace(0.0, 1.0, len(v), dtype=np.float32)
    colors = color_map(color_positions)[:, :3].astype(np.float32)

    widget._plot_points(v, persist=False, colors=colors, radius=point_radius)
    widget.display()

    print(f"✅ '{object_id}' 정점 {len(v)}개를 preview에 표시했습니다.")
