# mi_utils.py
import mitsuba as mi
import drjit as dr
import numpy as np

def get_mesh_vertices(scene, object_id: str) -> np.ndarray:
    """
    특정 객체의 메쉬(Mesh) 정점(Vertex) 좌표들을 NumPy 배열로 추출합니다.
    이동체(Vehicle)의 경로(Trajectory)를 도로 위에 생성할 때 유용합니다.
    
    Args:
        scene: Sionna Scene 객체
        object_id (str): 대상 객체의 ID
        
    Returns:
        np.ndarray: (N, 3) 형태의 좌표 배열 (없으면 빈 배열 반환)
    """
    mi_scene = scene.mi_scene
    target_shape = None
    
    # 1. Mitsuba Scene에서 해당 ID를 가진 Shape 찾기
    for s in mi_scene.shapes():
        if s.id() == object_id:
            target_shape = s
            break
            
    if target_shape is None:
        print(f"[mi_utils] 오류: ID가 '{object_id}'인 객체를 찾을 수 없습니다.")
        return np.array([])
        
    try:
        # 2. 파라미터 접근 및 데이터 추출
        params = mi.traverse(target_shape)
        
        if 'vertex_positions' in params:
            # Dr.Jit 배열을 Numpy로 변환 (CPU 메모리로 가져옴)
            # 데이터는 1차원 플랫 배열로 들어있음 [x1, y1, z1, x2, y2, z2, ...]
            v_flat = params['vertex_positions'].numpy()
            
            # (N, 3) 형태로 Reshape
            vertices = v_flat.reshape(-1, 3)
            print(f"[mi_utils] '{object_id}'에서 {len(vertices)}개의 정점을 추출했습니다.")
            return vertices
        else:
            print(f"[mi_utils] 경고: '{object_id}'는 메쉬가 아니거나 정점 정보가 없습니다.")
            return np.array([])
            
    except Exception as e:
        print(f"[mi_utils] 정점 추출 중 예외 발생: {e}")
        return np.array([])
    
