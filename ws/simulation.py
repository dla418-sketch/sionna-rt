import numpy as np
import tensorflow as tf
import mitsuba as mi
import drjit as dr
import os
import argparse
from tqdm import tqdm
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver

print("\n" + "="*50)
print("✅ [버전 확인] 수정된 full_simulation_v2.py 가 실행되었습니다.")
print("   - Tuple 반환 에러 패치 적용됨")
print("="*50 + "\n")

# ==============================================================================
# 0. 실행 옵션 설정
# ==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='사용할 GPU 번호 (0~3)')
parser.add_argument('--part', type=int, default=0, help='파일 구분 번호')
args = parser.parse_args()

# GPU 설정
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        target_gpu = gpus[args.gpu]
        tf.config.set_visible_devices(target_gpu, 'GPU')
        tf.config.experimental.set_memory_growth(target_gpu, True)
        print(f"✅ [Part {args.part}] GPU {args.gpu}번을 사용하여 작업을 시작합니다.")
    except RuntimeError as e:
        print(e)

# ==============================================================================
# 1. XML 경로 수정 및 장면 로드
# ==============================================================================
xml_path = "/home/hw/sionna/ws/scenes/kookmin/kookmin_itu.xml"
scene_dir = os.path.dirname(xml_path)
meshes_dir = os.path.join(scene_dir, "meshes")
temp_xml_path = os.path.join(scene_dir, "kookmin_fixed_final_v2.xml")

if not os.path.exists(xml_path):
    raise FileNotFoundError(f"❌ 원본 XML 파일을 찾을 수 없습니다: {xml_path}")

with open(xml_path, 'r', encoding='utf-8') as f:
    xml_content = f.read()

if os.path.exists(meshes_dir):
    abs_mesh_path = meshes_dir + "/" if not meshes_dir.endswith("/") else meshes_dir
    xml_content_fixed = xml_content.replace('value="meshes/', f'value="{abs_mesh_path}')
    with open(temp_xml_path, 'w', encoding='utf-8') as f:
        f.write(xml_content_fixed)
else:
    raise FileNotFoundError(f"❌ meshes 폴더를 찾을 수 없습니다: {meshes_dir}")

try:
    scene = load_scene(temp_xml_path)
    print("[성공] 장면(Scene) 로드 완료.")
except Exception as e:
    print(f"[치명적 오류] 장면 로드 실패: {e}")
    raise e

# ==============================================================================
# 2. 도로 좌표(road_positions) 추출
# ==============================================================================
road_object_id = "elm__00"
road_positions = []

if hasattr(scene, 'mi_scene'):
    mi_scene = scene.mi_scene
    target_shape = None
    for s in mi_scene.shapes():
        if s.id() == road_object_id:
            target_shape = s
            break
    
    if target_shape: 
        try:           
            params = mi.traverse(target_shape)
            if 'vertex_positions' in params:
                vertex_buffer = params['vertex_positions']
                vertices_flat = np.array(vertex_buffer)
                if len(vertices_flat) > 0:
                    road_positions = vertices_flat.reshape(-1, 3)
                    print(f"[탐색] 도로 좌표 추출 성공: {len(road_positions)}개")
                else:
                    print(" [경고] 도로 버퍼가 비어있습니다.")
        except Exception as e:
            print(f"[오류] Vertex 추출 중 에러: {e}")

if len(road_positions) == 0:
    raise ValueError("❌ 도로 좌표 추출 실패. 시뮬레이션을 중단합니다.")

# ==============================================================================
# 3. 시뮬레이션 파라미터
# ==============================================================================
CARRIER_FREQUENCY = 5.9e9   
TX_POWER_DBM = 26.0         

BURST_RATE = 20.0               
BURST_INTERVAL = 1.0 / BURST_RATE  
SNAPSHOTS_PER_BURST = 30        
BURST_DURATION = 640e-6         
SNAPSHOT_INTERVAL = BURST_DURATION / SNAPSHOTS_PER_BURST 

antenna_config = PlanarArray(num_rows=2, num_cols=4, pattern="iso", polarization="V")

scene.frequency = CARRIER_FREQUENCY
scene.tx_array = antenna_config
scene.rx_array = antenna_config

path_indices = [
    412, 410, 408, 406, 404, 401, 400, 72, 69, 68, 342, 340, 338, 335, 334, 
    88, 86, 84, 81, 464, 462, 459, 295, 370, 294, 368, 366, 363, 308, 306, 
    303, 302, 514, 511, 510, 488, 483, 482, 172, 269, 292, 288, 286, 284, 
    281, 604, 601, 507, 289, 506, 268, 522, 517, 516, 122, 120, 117, 116, 
    525, 582
]
valid_indices = [idx for idx in path_indices if idx < len(road_positions)]
waypoints = road_positions[valid_indices].copy()
waypoints[:, 1] += 1.5 

diffs = waypoints[1:] - waypoints[:-1]
segment_dists = np.linalg.norm(diffs, axis=1)
cumulative_dists = np.concatenate(([0], np.cumsum(segment_dists)))
total_distance = cumulative_dists[-1]
speed_ms = 60.0 / 3.6 

def get_pos_at_time(t):
    target_dist = np.clip(speed_ms * t, 0, total_distance)
    idx = np.searchsorted(cumulative_dists, target_dist) - 1
    idx = max(0, min(idx, len(waypoints) - 2))
    seg_start, seg_len = cumulative_dists[idx], segment_dists[idx]
    if seg_len == 0: return waypoints[idx]
    ratio = (target_dist - seg_start) / seg_len
    return waypoints[idx] + ratio * (waypoints[idx+1] - waypoints[idx])

# ==============================================================================
# 4. 시뮬레이션 실행 (Tuple Error 완전 해결)
# ==============================================================================
scene.transmitters.clear()
scene.receivers.clear()

tx_positions = [[-125.66, 56.36, -181.45], [323.47, 36.87, -204.31], [0.66, 56.36, -181.45]]
tx_names = ["BS1", "BS2", "BS3"]

for i, pos in enumerate(tx_positions):
    tx = Transmitter(name=tx_names[i], position=pos, power_dbm=TX_POWER_DBM)
    tx.transmit_antenna = scene.tx_array
    tx.look_at([0,0,0])
    scene.add(tx)

rx = Receiver(name="RxCar", position=[0,0,0])
rx.receive_antenna = scene.rx_array
scene.add(rx)

solver = PathSolver()

SIM_DURATION = 12.5 # GPU 분할용
num_bursts = int(SIM_DURATION * BURST_RATE) 

print(f"🚀 [Part {args.part}] Burst 시뮬레이션 시작 (목표: {num_bursts} Bursts)")

burst_data_list = []
start_time_offset = args.part * SIM_DURATION 

for b_idx in tqdm(range(num_bursts), desc=f"Part {args.part}"):
    absolute_b_idx = (args.part * num_bursts) + b_idx
    burst_start_time = absolute_b_idx * BURST_INTERVAL
    
    b_a, b_tau, b_pos, b_time = [], [], [], []
    
    for s_idx in range(SNAPSHOTS_PER_BURST):
        current_time = burst_start_time + (s_idx * SNAPSHOT_INTERVAL)
        pos = get_pos_at_time(current_time)
        
        rx.position = pos
        paths = solver(scene, max_depth=3, samples_per_src=100000, 
                       diffuse_reflection=True, diffraction=True)
        
        # 🔥 [Tuple 처리 로직: 중요]
        if paths.a is not None:
            # 1. 'a' (Channel Gain) 처리
            if isinstance(paths.a, tuple):
                real_part = paths.a[0]
                imag_part = paths.a[1]
                # tuple 내부 요소가 Tensor인지 확인하고 numpy 변환
                if hasattr(real_part, 'numpy'): real_part = real_part.numpy()
                if hasattr(imag_part, 'numpy'): imag_part = imag_part.numpy()
                b_a.append(real_part + 1j * imag_part)
            else:
                try:
                    b_a.append(paths.a.numpy())
                except:
                    b_a.append(np.array(paths.a))
            
            # 2. 'tau' (Delay) 처리
            if isinstance(paths.tau, tuple):
                val = paths.tau[0]
                if hasattr(val, 'numpy'): val = val.numpy()
                b_tau.append(val)
            else:
                try:
                    b_tau.append(paths.tau.numpy())
                except:
                    b_tau.append(np.array(paths.tau))
        else:
            b_a.append(np.array([]))
            b_tau.append(np.array([]))
            
        b_pos.append(pos)
        b_time.append(current_time)
        
    burst_data_list.append({
        'burst_idx': absolute_b_idx,
        'a': b_a,       
        'tau': b_tau,   
        'pos': b_pos,
        'time': b_time
    })

save_name = f'channel_history_burst_mimo_3bs_part{args.part}.npy'
np.save(save_name, burst_data_list)
print(f"💾 Part {args.part} 저장 완료: {save_name}")