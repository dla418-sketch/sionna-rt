import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input, mixed_precision
import os
import glob
from tqdm import tqdm
import gc

print("\n" + "="*60)
print("🚀 [논문 구현] GIE & Meta-Learning 학습 코드 (V3: 메모리 최적화)")
print("   - 수정사항: tf.data.Dataset 파이프라인 전면 적용 (OOM 방지)")
print("="*60 + "\n")

# ==============================================================================
# 0. GPU 및 메모리 설정
# ==============================================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Memory Growth 설정 완료 (GPU {len(gpus)}대)")
    except RuntimeError as e:
        print(e)

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# ==============================================================================
# 1. 설정 파라미터
# ==============================================================================
DATA_FILE_PATTERN = "channel_history_burst_mimo_3bs_final.npy"
BANDWIDTH = 15e6
NUM_SUBCARRIERS = 64
FREQUENCIES = np.linspace(-BANDWIDTH/2, BANDWIDTH/2, NUM_SUBCARRIERS)

INPUT_SEQ_LEN = 10   
PRED_SEQ_LEN = 10    
BATCH_SIZE = 16       # 메모리 안전을 위해 작게 유지
EPOCHS = 50           
META_ITERATIONS = 3   

# ==============================================================================
# 2. 데이터 전처리
# ==============================================================================
def load_and_preprocess_data():
    file_list = glob.glob(DATA_FILE_PATTERN)
    if not file_list:
        raise FileNotFoundError("❌ 데이터 파일을 찾을 수 없습니다.")
    
    print(f"📂 데이터 파일 로드 중: {len(file_list)}개")
    all_bursts_H = [] 

    for fname in file_list:
        data = np.load(fname, allow_pickle=True)
        for burst in tqdm(data, desc=f"Processing {fname}"):
            a_list = burst['a']
            tau_list = burst['tau']
            
            h_seq = []
            for i in range(len(a_list)):
                if a_list[i].size == 0:
                    h_seq.append(np.zeros((8, 24, 64), dtype=np.complex64))
                    continue
                try:
                    a_raw = a_list[i]
                    tau_raw = tau_list[i]
                    a_sq = a_raw.reshape(8, 24, -1)
                    
                    f_reshaped = FREQUENCIES.reshape(1, 1, 1, -1).astype(np.float32)
                    tau_flat = tau_raw.flatten()[:a_sq.shape[-1]]
                    tau_input = tau_flat.reshape(1, 1, -1, 1).astype(np.float32)
                    a_input = a_sq[..., np.newaxis]
                    
                    phase = -1j * 2 * np.pi * tau_input * f_reshaped
                    h_val = np.sum(a_input * np.exp(phase), axis=-2)
                except:
                    h_val = np.zeros((8, 24, 64), dtype=np.complex64)

                h_seq.append(h_val)
            all_bursts_H.append(np.array(h_seq))
        del data
        gc.collect()
    return all_bursts_H

def create_dataset(bursts_data):
    X, Y = [], []
    for burst in bursts_data:
        if len(burst) < INPUT_SEQ_LEN + PRED_SEQ_LEN: continue
        burst_concat = np.concatenate([burst.real, burst.imag], axis=-1)
        for i in range(len(burst) - INPUT_SEQ_LEN - PRED_SEQ_LEN + 1):
            X.append(burst_concat[i : i+INPUT_SEQ_LEN])
            Y.append(burst_concat[i+INPUT_SEQ_LEN : i+INPUT_SEQ_LEN+PRED_SEQ_LEN])
            
    X_np = np.array(X, dtype=np.float32)
    Y_np = np.array(Y, dtype=np.float32)
    del X, Y
    gc.collect()
    return X_np, Y_np

# [NEW] 데이터셋 생성 헬퍼 함수
def make_tf_dataset(X, Y, batch_size, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if shuffle:
        ds = ds.shuffle(buffer_size=1024)
    # prefetch를 통해 GPU가 놀지 않게 데이터 미리 준비
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# ==============================================================================
# 3. GIE 모듈 (Lambda Layer - Fixed)
# ==============================================================================
def GIE_Module(inputs):
    def shift_time_axis(x):
        x_sliced = x[:, :-1, :, :, :]
        x_shifted = tf.pad(x_sliced, [[0,0], [1,0], [0,0], [0,0], [0,0]])
        return x_shifted

    shifted_inputs = layers.Lambda(shift_time_axis, name="Time_Shift_Op")(inputs)
    gradients = layers.Subtract(name="Gradient_Calc")([inputs, shifted_inputs])

    gie_feat = layers.TimeDistributed(
        layers.Conv2D(64, (3,3), padding='same', activation='relu'), 
        name='GIE_Conv'
    )(gradients)
    gie_feat = layers.TimeDistributed(layers.BatchNormalization())(gie_feat)
    
    return gie_feat

# ==============================================================================
# 4. 모델 정의
# ==============================================================================
def build_gie_model(input_steps=10, pred_steps=10, name="Student_Model"):
    inputs = Input(shape=(input_steps, 8, 24, 128), name='CSI_Input')
    
    gie_features = GIE_Module(inputs)
    
    x = layers.TimeDistributed(layers.Conv2D(64, (3,3), padding='same', activation='relu'))(inputs)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    
    x = layers.Concatenate(axis=-1)([x, gie_features])
    
    x = layers.TimeDistributed(layers.Conv2D(128, (3,3), padding='same', activation='relu'))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = layers.TimeDistributed(layers.Flatten())(x)
    
    x = layers.LSTM(512, return_sequences=False)(x)
    x = layers.RepeatVector(pred_steps)(x)
    x = layers.LSTM(512, return_sequences=True)(x)
    
    output_dim = 8 * 24 * 128
    x = layers.TimeDistributed(layers.Dense(output_dim, dtype='float32'))(x)
    outputs = layers.Reshape((pred_steps, 8, 24, 128))(x)
    
    return models.Model(inputs=inputs, outputs=outputs, name=name)

# ==============================================================================
# 5. 메타 러닝 실행 (Dataset 적용)
# ==============================================================================
if __name__ == "__main__":
    # 1. 데이터 로드 및 분할
    bursts = load_and_preprocess_data()
    X_full, Y_full = create_dataset(bursts)
    
    # 원본 데이터 삭제하여 메모리 확보 (Slice는 View일 수 있으므로 copy본 생성 후 삭제)
    split_idx = int(len(X_full) * 0.5)
    
    # 메모리 절약을 위해 deep copy 후 원본 즉시 삭제
    X_labeled = np.array(X_full[:split_idx])
    Y_labeled = np.array(Y_full[:split_idx])
    X_unlabeled = np.array(X_full[split_idx:])
    Y_unlabeled_true = np.array(Y_full[split_idx:]) # 평가용
    
    del X_full, Y_full, bursts
    gc.collect()
    
    print(f"📊 데이터 분할 완료: Labeled {len(X_labeled)} / Unlabeled {len(X_unlabeled)}")

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        teacher_model = build_gie_model(name="Teacher")
        student_model = build_gie_model(name="Student")
        
        teacher_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        student_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # 메타 러닝 루프
    for iteration in range(META_ITERATIONS):
        print("\n" + "#"*60)
        print(f"🔄 Meta-Learning Iteration {iteration+1}/{META_ITERATIONS}")
        print("#"*60)
        
        # [Step 1] Teacher 학습 (Labeled Data)
        print("👨‍🏫 [Teacher] 학습 중...")
        # 🔥 tf.data.Dataset으로 변환해서 fit
        teacher_ds = make_tf_dataset(X_labeled, Y_labeled, BATCH_SIZE)
        teacher_model.fit(teacher_ds, epochs=5, verbose=1)
        del teacher_ds # 사용 후 데이터셋 객체 정리
        gc.collect()
        
        # [Step 2] Pseudo-Label 생성
        print("🔮 [Teacher] Pseudo-Label 생성 중...")
        # predict도 배치 단위로 실행되지만 결과값이 크므로 주의
        pseudo_labels = teacher_model.predict(X_unlabeled, batch_size=BATCH_SIZE, verbose=1)
        
        # [Step 3] Student 학습 (Combined Data)
        print("🧑‍🎓 [Student] 데이터 병합 및 학습 준비...")
        # numpy concat은 메모리를 많이 쓰므로 주의. 
        # (X_labeled + X_unlabeled) -> Dataset 생성 -> 원본 삭제 패턴 사용
        
        X_combined = np.concatenate([X_labeled, X_unlabeled], axis=0)
        Y_combined = np.concatenate([Y_labeled, pseudo_labels], axis=0)
        
        # 🔥 여기서 Dataset으로 바로 변환하고 numpy array는 즉시 삭제!
        student_ds = make_tf_dataset(X_combined, Y_combined, BATCH_SIZE)
        
        del X_combined, Y_combined
        gc.collect() # 메모리 청소
        
        print("🧑‍🎓 [Student] 학습 시작...")
        student_model.fit(student_ds, epochs=5, verbose=1)
        
        del student_ds, pseudo_labels
        gc.collect()
        
        # [Step 4] Teacher 업데이트
        print("💡 Evolving Teacher with Student's knowledge...")
        alpha = 0.5
        for t_var, s_var in zip(teacher_model.trainable_variables, student_model.trainable_variables):
            t_var.assign(alpha * t_var + (1 - alpha) * s_var)

    print("\n💾 최종 Student 모델 저장 중...")
    student_model.save("gie_meta_student_model.h5")
    
    print("\n🏆 최종 성능 평가")
    # 평가용 데이터셋 생성
    test_ds = make_tf_dataset(X_unlabeled, Y_unlabeled_true, BATCH_SIZE, shuffle=False)
    loss, mae = student_model.evaluate(test_ds)
    print(f"   - Final MSE: {loss:.6f}")
    print(f"   - Final MAE: {mae:.6f}")