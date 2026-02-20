import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import os
import glob
from tqdm import tqdm

# ==============================================================================
# 1. 설정 파라미터
# ==============================================================================
# 데이터 파일 경로 패턴 (part0.npy, part1.npy ... 모두 읽음)
DATA_FILE_PATTERN = "channel_history_burst_mimo_3bs_part*.npy"

# 물리 계층 파라미터
BANDWIDTH = 15e6
NUM_SUBCARRIERS = 64
FREQUENCIES = np.linspace(-BANDWIDTH/2, BANDWIDTH/2, NUM_SUBCARRIERS)

# 학습 파라미터
INPUT_SEQ_LEN = 10   # 과거 10개 보고
PRED_SEQ_LEN = 10    # 미래 10개 예측
BATCH_SIZE = 64      # GPU 메모리에 따라 조절 (32 ~ 128)
EPOCHS = 100         # 학습 반복 횟수
VALIDATION_SPLIT = 0.2 # 검증 데이터 비율

# ==============================================================================
# 2. 모델 정의 (보내주신 코드 + 수정)
# ==============================================================================
def build_paper_cnn_lstm(input_steps=10, pred_steps=10):
    # 입력: (Time, Rx=8, Tx=24, Freq_RealImag=128)
    inputs = Input(shape=(input_steps, 8, 24, 128))
    
    # 1. Spatial Feature Extraction (CNN)
    x = layers.TimeDistributed(layers.Conv2D(64, (3,3), padding='same', activation='relu'))(inputs)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.Conv2D(128, (3,3), padding='same', activation='relu'))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    
    # Pooling: (8, 24) -> (4, 12)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    
    # Flatten: (Time, 4*12*128)
    x = layers.TimeDistributed(layers.Flatten())(x)
    
    # 2. Temporal Feature Extraction (LSTM)
    x = layers.LSTM(512, return_sequences=False)(x) # Encoder
    
    x = layers.RepeatVector(pred_steps)(x)            # Future placeholder
    x = layers.LSTM(512, return_sequences=True)(x)    # Decoder
    
    # 3. Output Reconstruction
    # 원래 차원인 8*24*128 (Rx*Tx*Freq*2)로 복원
    output_dim = 8 * 24 * 128
    x = layers.TimeDistributed(layers.Dense(output_dim))(x)
    outputs = layers.Reshape((pred_steps, 8, 24, 128))(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Paper_CNN_LSTM")
    return model

# ==============================================================================
# 3. 데이터 로드 및 전처리 (Physics -> AI Data)
# ==============================================================================
def load_and_preprocess_data():
    file_list = glob.glob(DATA_FILE_PATTERN)
    if not file_list:
        raise FileNotFoundError("❌ 데이터 파일(*.npy)을 찾을 수 없습니다. 시뮬레이션을 먼저 실행하세요.")
    
    print(f"📂 발견된 데이터 파일: {len(file_list)}개")
    
    all_bursts_H = [] # 변환된 H 행렬들을 담을 리스트

    for fname in file_list:
        print(f"Reading {fname}...")
        data = np.load(fname, allow_pickle=True)
        
        for burst in tqdm(data, desc=f"Processing {fname}"):
            a_list = burst['a']     # (30, 1, 8, 3, 8, Paths)
            tau_list = burst['tau'] # (30, 1, 3, Paths) -> Sionna 구조에 따라 다름
            
            # Burst 내 30개 스냅샷 처리
            h_seq = []
            for i in range(len(a_list)):
                # 1. Path Gain (a) 처리
                # 예상 Shape: (1, Rx=8, Tx=3, TxAnt=8, Paths)
                # 목표 Shape: (Rx=8, TxTotal=24, Paths)
                if a_list[i].size == 0:
                    h_seq.append(np.zeros((8, 24, 64), dtype=np.complex64))
                    continue
                    
                a_raw = a_list[i] 
                # 차원 축소 및 병합: (1, 8, 3, 8, Paths) -> (8, 3, 8, Paths) -> (8, 24, Paths)
                # squeeze로 불필요한 차원 제거 (Batch 등)
                try:
                    a_sq = a_raw.reshape(8, 3, 8, -1) 
                    a_flat = a_sq.reshape(8, 24, -1) # (Rx, Tx*TxAnt, Paths)
                except:
                    # Shape이 안 맞을 경우 예외 처리 (빈 데이터 등)
                    h_seq.append(np.zeros((8, 24, 64), dtype=np.complex64))
                    continue

                # 2. Delay (tau) 처리
                tau_raw = tau_list[i]
                # Broadcasting을 위해 차원 맞추기
                # tau는 보통 (Batch, Tx, Paths) 형태임 -> (1, 3, Paths)
                # 이를 (8, 24, Paths)로 확장해야 함
                try:
                    tau_sq = tau_raw.flatten() # 일단 펼침
                    # Paths 개수가 a와 맞는지 확인 필요하지만, 여기선 Broadcasting 이용
                    # 가장 간단한 방법: tau를 (1, 1, Paths)로 보고 확장
                    # 하지만 정확히는 Tx별로 다르므로, (3, Paths) -> (24, Paths) -> (8, 24, Paths)
                    
                    # 간단화: tau shape의 마지막 차원이 Paths라고 가정
                    num_paths = a_flat.shape[-1]
                    tau_flat = tau_raw.reshape(-1) # 전체 다 펼치고
                    # 경로 개수에 맞춰 자르거나 확장 (Sionna 구조 특성상 복잡하므로 단순화)
                    
                    # [중요] Sionna RT 출력 구조상, tau는 (Tx, Paths) 혹은 (1, Paths)일 수 있음.
                    # 여기서는 계산 효율을 위해 a_flat에 맞는 차원으로 강제 확장
                    tau_broadcast = tau_raw.reshape(1, 1, -1) if tau_raw.ndim == 1 else tau_raw
                    # (Broadcasting은 numpy가 알아서 처리하도록 유도)
                    
                except:
                    h_seq.append(np.zeros((8, 24, 64), dtype=np.complex64))
                    continue

                # 3. Frequency Response 계산: H = sum( a * exp(-j2pi * tau * f) )
                # Frequencies: (1, 1, 1, 64)
                # a_flat: (8, 24, Paths, 1)
                # tau: (..., Paths, 1)
                
                f_reshaped = FREQUENCIES.reshape(1, 1, 1, -1)
                a_input = a_flat[..., np.newaxis] 
                
                # tau 차원 맞추기 (약식: 경로 수만 맞으면 작동)
                if tau_raw.size > 0:
                    # tau의 마지막 차원이 Paths라고 가정
                    tau_input = tau_raw.flatten()[:a_flat.shape[-1]] # 경로 수 맞춤
                    tau_input = tau_input.reshape(1, 1, -1, 1)
                    
                    phase = -1j * 2 * np.pi * tau_input * f_reshaped
                    h_val = np.sum(a_input * np.exp(phase), axis=-2) # Sum over paths
                else:
                    h_val = np.zeros((8, 24, 64), dtype=np.complex64)

                h_seq.append(h_val) # (8, 24, 64)

            all_bursts_H.append(np.array(h_seq)) # (30, 8, 24, 64)

    return all_bursts_H

def create_dataset(bursts_data):
    X, Y = [], []
    print("🔄 데이터셋(X, Y) 생성 중...")
    
    for burst in bursts_data:
        # burst shape: (30, 8, 24, 64) (Complex)
        if len(burst) < INPUT_SEQ_LEN + PRED_SEQ_LEN:
            continue
            
        # 복소수 -> 실수/허수 채널 분리 (8, 24, 64) -> (8, 24, 128)
        # Real: [..., 0:64], Imag: [..., 64:128]
        burst_real = burst.real
        burst_imag = burst.imag
        burst_concat = np.concatenate([burst_real, burst_imag], axis=-1) # (30, 8, 24, 128)
        
        # 슬라이딩 윈도우
        for i in range(len(burst) - INPUT_SEQ_LEN - PRED_SEQ_LEN + 1):
            X.append(burst_concat[i : i+INPUT_SEQ_LEN])
            Y.append(burst_concat[i+INPUT_SEQ_LEN : i+INPUT_SEQ_LEN+PRED_SEQ_LEN])
            
    return np.array(X), np.array(Y)

# ==============================================================================
# 4. 메인 실행 (학습)
# ==============================================================================
if __name__ == "__main__":
    # 1. GPU 확인
    strategy = tf.distribute.MirroredStrategy()
    print(f"✅ 사용 가능한 GPU 수: {strategy.num_replicas_in_sync}")

    # 2. 데이터 로드 및 변환
    bursts = load_and_preprocess_data()
    if len(bursts) == 0:
        print("❌ 데이터가 비어있습니다.")
        exit()
        
    # 3. 학습 데이터셋 만들기
    X_train, Y_train = create_dataset(bursts)
    print(f"📊 학습 데이터 준비 완료:")
    print(f"   - X shape: {X_train.shape} (Samples, Time, Rx, Tx, Feat)")
    print(f"   - Y shape: {Y_train.shape}")
    
    # 4. 모델 생성 및 학습
    with strategy.scope():
        model = build_paper_cnn_lstm(INPUT_SEQ_LEN, PRED_SEQ_LEN)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        print("\n🚀 학습 시작 (Training)...")
        history = model.fit(
            X_train, Y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            verbose=1
        )
        
    # 5. 모델 저장
    model.save("cnn_lstm_mimo_model.h5")
    print("💾 모델 저장 완료: cnn_lstm_mimo_model.h5")