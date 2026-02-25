import os
import tensorflow as tf
import numpy as np
import sionna

# [버전 호환성] Import 경로 처리
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver
try:
    from sionna.phy.ofdm import ResourceGrid
except ImportError:
    from sionna.ofdm import ResourceGrid

# ==========================================
# 1. 파일 저장 경로 설정 (요청 사항 반영)
# ==========================================
# 저장할 디렉토리 경로
output_dir = "/data/hw/sionna/ws/output"

# 디렉토리가 없으면 생성
if not os.path.exists(output_dir):
    try:
        os.makedirs(output_dir)
        print(f"디렉토리 생성됨: {output_dir}")
    except OSError as e:
        print(f"[오류] 디렉토리를 생성할 수 없습니다: {e}")
        # 실패 시 현재 디렉토리에 저장하도록 fallback
        output_dir = "."

# Scene 로딩을 위한 경로 설정 (기존 유지)
xml_path = "/data1/mh/sionna/src/sionna/rt/scenes/kookmin/kookmin_itu.xml"
scene_dir = os.path.dirname(xml_path)
temp_xml_path = os.path.join(scene_dir, "kookmin_fixed_absolute.xml")

# ==========================================
# 2. 시스템 및 시뮬레이션 파라미터 (정밀 설정)
# ==========================================
carrier_frequency = 3.5e9
subcarrier_spacing = 30e3 
fft_size = 72
num_ofdm_symbols = 14

# [수정됨] 정밀 시뮬레이션 설정
simulation_time = 10.0 # 10초
dt = 0.5e-3 # 0.5 ms (30kHz SCS의 1 Slot 길이)
total_steps = int(simulation_time / dt)

print(f"--- 시뮬레이션 설정 ---")
print(f"총 시간: {simulation_time}초")
print(f"시간 간격(dt): {dt}초 (0.5ms)")
print(f"총 스텝 수: {total_steps} (메모리 주의)")

# ==========================================
# 3. 경로 및 기지국 설정
# ==========================================
# 기지국(BS) 위치
bs_position = [323.47, 36.87, -204.31]

# 이동성 설정
num_ues = 8
speeds = [10, 15, 20, 25, 30, 40, 50, 60]

# [필수] 이전 셀에서 생성된 road_positions 확인
if 'road_positions' not in globals() or len(road_positions) == 0:
    raise ValueError("메모리에 'road_positions' 변수가 없습니다. 도로 좌표 추출 코드를 먼저 실행해주세요.")

path_indices = [
    412, 410, 408, 406, 404, 401, 400, 72, 69, 68, 342, 340, 338, 335, 334, 
    88, 86, 84, 81, 464, 462, 459, 295, 370, 294, 368, 366, 363, 308, 306, 
    303, 302, 514, 511, 510, 488, 483, 482, 172, 269, 292, 288, 286, 284, 
    281, 604, 601, 507, 289, 506, 268, 522, 517, 516, 122, 120, 117, 116, 
    525, 582
]
trajectory_points = road_positions[path_indices]

# PolylineWalker 클래스 정의
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
        seg_len = self.seg_lengths[idx]
        ratio = (distance - self.cum_dist[idx]) / seg_len if seg_len > 0 else 0
        return p_start + (p_end - p_start) * ratio

walker = PolylineWalker(trajectory_points)

# ==========================================
# 4. Scene 구성
# ==========================================
scene = load_scene(temp_xml_path)

# 안테나: BS(8x8), UE(1x1)
bs_array = PlanarArray(num_rows=8, num_cols=8, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso", polarization="V")
ue_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")

scene.tx_array = bs_array
scene.rx_array = ue_array

# 기기 배치
tx = Transmitter(name="BS", position=bs_position, orientation=[0,0,0])
scene.add(tx)

ues = []
start_pos = trajectory_points[0]
for i in range(num_ues):
    rx = Receiver(name=f"UE_{i}", position=start_pos, orientation=[0,0,0])
    scene.add(rx)
    ues.append(rx)

# ==========================================
# 5. 시뮬레이션 루프 (대용량 데이터 생성)
# ==========================================
solver = PathSolver()
dataset_h = [] # 여기에 20,000개의 데이터가 쌓입니다.

print("시뮬레이션 시작...")

for step in range(total_steps):
    current_time = step * dt
    
    # 이동
    for i, ue in enumerate(ues):
        speed = speeds[i]
        new_pos = walker.get_position(speed * current_time)
        ue.position = new_pos
    
    # Ray Tracing
    paths = solver(scene, max_depth=3)
    
    # CFR 계산
    frequencies = subcarrier_spacing * tf.range(fft_size, dtype=tf.float32)
    cfr_output = paths.cfr(frequencies=frequencies)
    
    if isinstance(cfr_output, tuple):
        h_freq = cfr_output[0]
    else:
        h_freq = cfr_output
        
    dataset_h.append(h_freq.numpy())
    
    # 진행 상황 출력 (너무 자주 출력하지 않도록 1000스텝마다)
    if step % 1000 == 0:
        print(f"Progress: {step}/{total_steps} ({(step/total_steps)*100:.1f}%)")

# ==========================================
# 6. 저장
# ==========================================
#print("데이터 변환 중... (시간이 소요될 수 있습니다)")
#dataset_h = np.array(dataset_h)
#dataset_h = np.squeeze(dataset_h)

#print(f"최종 데이터 형태: {dataset_h.shape}")
# 예상: (20000, 8, 64, 72)

#file_name = "vit_channel_dataset_precise_10s.npy"
#save_path = os.path.join(output_dir, file_name)

#np.save(save_path, dataset_h)
#print(f"저장 완료: {save_path}")