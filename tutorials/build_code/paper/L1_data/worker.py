# 파일명: worker.py
import os
import numpy as np
import time

def run_simulation_chunk(gpu_id, ue_indices, speeds_kmh, speeds_ms, 
                         xml_path, temp_xml_path, road_positions, 
                         tx_positions, tx_names, 
                         total_steps, dt, frequencies_arg): # 인자 이름 변경 (무시용)
    
    # 1. GPU 설정 (Sionna/TF 임포트 전 필수)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            
    # 2. 무거운 라이브러리 로드
    import drjit as dr
    import gc
    from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver

    print(f"🚀 [GPU {gpu_id}] 시작! 담당 UE: {ue_indices}")
    
    # [!!! 핵심 수정 !!!] 
    # 메인에서 None으로 넘어왔으므로, 여기서 다시 텐서를 생성해야 합니다.
    # 단일 주파수 (Central Frequency) 설정
    frequencies = tf.cast([0], tf.float32)
    
    # 3. 경로 인덱스 및 Walker 클래스 정의
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

    # 4. Scene 로드
    scene = load_scene(temp_xml_path)
    
    bs_array = PlanarArray(num_rows=8, num_cols=8, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso", polarization="V")
    ue_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.tx_array = bs_array
    scene.rx_array = ue_array
    
    # Dummy Receiver
    dummy_rx_name = "Active_Rx"
    scene.add(Receiver(name=dummy_rx_name, position=trajectory_points[0], orientation=[0,0,0]))
    active_rx = scene.receivers[dummy_rx_name]
    
    solver = PathSolver()
    local_results = {} 

    # 5. 시뮬레이션 루프
    for ue_real_idx in ue_indices:
        ue_speed = speeds_ms[ue_real_idx]
        ue_bs_data_list = [] 
        
        # Tx 루프
        for tx_name, tx_pos in zip(tx_names, tx_positions):
            active_tx_name = "Active_Tx"
            if active_tx_name in scene.transmitters: scene.remove(active_tx_name)
            scene.add(Transmitter(name=active_tx_name, position=tx_pos, orientation=[0,0,0]))
            
            current_tx_data = []
            
            # Time 루프
            for step in range(total_steps):
                current_time = step * dt
                new_pos = walker.get_position(ue_speed * current_time)
                active_rx.position = new_pos
                
                # Ray Tracing (샘플 2만개)
                paths = solver(scene, max_depth=3, samples_per_src=20000)
                
                # [수정] 위에서 새로 생성한 frequencies 변수 사용
                cfr = paths.cfr(frequencies=frequencies)
                
                if isinstance(cfr, tuple): cfr = cfr[0]
                
                # 데이터 추출
                val = np.squeeze(cfr.numpy())
                current_tx_data.append(val)
                
                # 메모리 정리
                del paths, cfr
                if step % 2000 == 0:
                    dr.flush_malloc_cache()
                    gc.collect()

            # Tx 하나 완료
            ue_bs_data_list.append(np.array(current_tx_data)) # (Time, 64)
            
            scene.remove(active_tx_name)
            dr.flush_malloc_cache()
            gc.collect()
        
        # UE 하나 완료 -> (BS, Time, 64) -> Transpose (Time, BS, 64)
        ue_combined = np.stack(ue_bs_data_list, axis=0)
        ue_combined = np.transpose(ue_combined, (1, 0, 2))
        
        local_results[ue_real_idx] = ue_combined
        print(f"  ✅ [GPU {gpu_id}] UE {ue_real_idx} 완료!")

    return local_results