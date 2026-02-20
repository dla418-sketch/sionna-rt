# 파일명: run_manual.py
import os
import argparse
import numpy as np
import time

# ==========================================
# 실행 인자 설정 (터미널에서 입력받음)
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, required=True, help="사용할 GPU 번호 (0 또는 1)")
parser.add_argument("--ues", type=str, required=True, help="처리할 UE 인덱스 (콤마로 구분, 예: 0,1,2)")
args = parser.parse_args()

# GPU 설정 (가장 먼저 해야 함)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import tensorflow as tf
# 메모리 증가 허용
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

import drjit as dr
import gc
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver

# ==========================================
# 설정
# ==========================================
# UE 리스트 파싱 ("0,1,2" -> [0, 1, 2])
target_ues = [int(x) for x in args.ues.split(',')]

print(f"🚀 [GPU {args.gpu}] 시뮬레이션 시작!")
print(f"   - 담당 UE: {target_ues}")

output_dir = "/data1/mh/sionna/tutorials/rt/build code/paper/L1_data"
if not os.path.exists(output_dir): os.makedirs(output_dir)

# 맵 및 경로 설정
xml_path = "/data1/mh/sionna/src/sionna/rt/scenes/kookmin/kookmin_itu.xml"
scene_dir = os.path.dirname(xml_path)
temp_xml_path = os.path.join(scene_dir, "kookmin_fixed_absolute.xml")

simulation_time = 60.0
dt = 0.5e-3
total_steps = int(simulation_time / dt)
frequencies = tf.cast([0], tf.float32)

speeds_kmh = [20, 40, 60, 80, 100, 120]
speeds_ms = [v / 3.6 for v in speeds_kmh]

# Road Positions 로드 (파일에서 로드한다고 가정하거나, 기존 로직 사용)
# 주의: 이 파일 실행 전 road_positions를 얻는 로직이 필요함.
# 편의상 여기서는 기존에 메모리에 있던 값을 사용하지 못하므로, 
# 만약 road_positions 파일이 없다면 아래 코드가 에러날 수 있음.
# (이전에 실행한 노트북에서 road_positions를 .npy로 저장해두는 것을 추천)

# [임시] road_positions가 없으면 에러가 나므로, 
# 기존 노트북에서 np.save('road_positions.npy', road_positions)를 먼저 수행하세요.
try:
    road_positions = np.load("road_positions.npy")
except FileNotFoundError:
    print("❌ 'road_positions.npy' 파일을 찾을 수 없습니다.")
    print("   이전 코드에서 np.save('road_positions.npy', road_positions)를 먼저 실행해주세요.")
    exit()

path_indices = [
    412, 410, 408, 406, 404, 401, 400, 72, 69, 68, 342, 340, 338, 335, 334, 
    88, 86, 84, 81, 464, 462, 459, 295, 370, 294, 368, 366, 363, 308, 306, 
    303, 302, 514, 511, 510, 488, 483, 482, 172, 269, 292, 288, 286, 284, 
    281, 604, 601, 507, 289, 506, 268, 522, 517, 516, 122, 120, 117, 116, 
    525, 582
]
trajectory_points = road_positions[path_indices]

class PolylineWalker:
    def __init__(self, points):
        self.points = points
        diffs = points[1:] - points[:-1]
        self.seg_lengths = np.linalg.norm(diffs, axis=1)
        self.cum_dist = np.insert(np.cumsum(self.seg_lengths), 0, 0.0)
        self.total_length = self.cum_dist[-1]
    def get_position(self, distance):
        if distance >= self.total_length: return self.points[-1]
        if distance <= 0: return self.points[0]
        idx = np.searchsorted(self.cum_dist, distance) - 1
        idx = max(0, idx)
        p_start = self.points[idx]
        p_end = self.points[idx+1]
        ratio = (distance - self.cum_dist[idx]) / self.seg_lengths[idx] if self.seg_lengths[idx] > 0 else 0
        return p_start + (p_end - p_start) * ratio

walker = PolylineWalker(trajectory_points)

# Scene 로드
scene = load_scene(temp_xml_path)
bs_array = PlanarArray(num_rows=8, num_cols=8, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso", polarization="V")
ue_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
scene.tx_array = bs_array
scene.rx_array = ue_array

tx_positions = [[-125.663, 56.367, -181.453], [323.472, 36.869, -204.315], [0.663, 56.367, -181.453]]
tx_names = ["Tx_1", "Tx_2", "Tx_3"]

dummy_rx_name = "Active_Rx"
scene.add(Receiver(name=dummy_rx_name, position=trajectory_points[0], orientation=[0,0,0]))
active_rx = scene.receivers[dummy_rx_name]
solver = PathSolver()

final_data_container = [] 

# ==========================================
# 시뮬레이션 루프
# ==========================================
for ue_real_idx in target_ues:
    ue_speed = speeds_ms[ue_real_idx]
    print(f"\n[GPU {args.gpu}] UE {ue_real_idx} (속도 {speeds_kmh[ue_real_idx]}km/h) 처리 시작...")
    
    ue_bs_data_list = [] 
    
    for tx_idx, (tx_name, tx_pos) in enumerate(zip(tx_names, tx_positions)):
        print(f"  ├─ BS {tx_idx+1} 계산 중...", end="", flush=True) # flush=True 중요
        
        active_tx_name = "Active_Tx"
        if active_tx_name in scene.transmitters: scene.remove(active_tx_name)
        scene.add(Transmitter(name=active_tx_name, position=tx_pos, orientation=[0,0,0]))
        
        current_tx_data = []
        
        for step in range(total_steps):
            current_time = step * dt
            new_pos = walker.get_position(ue_speed * current_time)
            active_rx.position = new_pos
            
            paths = solver(scene, max_depth=3, samples_per_src=20000)
            cfr = paths.cfr(frequencies=frequencies)
            if isinstance(cfr, tuple): cfr = cfr[0]
            
            val = np.squeeze(cfr.numpy())
            current_tx_data.append(val)
            
            del paths, cfr
            if step % 2000 == 0:
                dr.flush_malloc_cache()
                gc.collect()
        
        ue_bs_data_list.append(np.array(current_tx_data))
        scene.remove(active_tx_name)
        dr.flush_malloc_cache()
        gc.collect()
        print(" 완료.")

    # 저장 (UE 1명 끝날 때마다 저장해버림 - 안전하게)
    ue_combined = np.stack(ue_bs_data_list, axis=0) # (BS, Time, 64)
    ue_combined = np.transpose(ue_combined, (1, 0, 2)) # (Time, BS, 64)
    
    # 부분 파일 저장 (Part_UE_0.npy)
    save_name = f"part_data_UE_{ue_real_idx}.npy"
    np.save(os.path.join(output_dir, save_name), ue_combined)
    print(f"  💾 UE {ue_real_idx} 데이터 저장 완료: {save_name}")

print(f"🏁 [GPU {args.gpu}] 모든 작업 완료.")