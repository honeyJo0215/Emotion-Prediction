import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# =============================================================================
# 결과 저장 경로 설정
RESULT_DIR = "/home/bcml1/sigenv/_4월/eeg_signal_diffusion/eeg_diffusion1"
os.makedirs(RESULT_DIR, exist_ok=True)

# =============================================================================
# GPU 메모리 제한 (필요시 사용)
def limit_gpu_memory(memory_limit_mib=10000):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit_mib)]
            )
            print(f"GPU memory limited to {memory_limit_mib} MiB.")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU available, using CPU.")
limit_gpu_memory(10000)

# =============================================================================
# DiffuSSMLayer: Dense와 LayerNormalization을 이용한 노이즈 제거 모듈
class DiffuSSMLayer(layers.Layer):
    def __init__(self, hidden_dim=64, output_units=None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_units = output_units
        self.dense1 = layers.Dense(hidden_dim, activation='relu')
        self.norm1 = layers.LayerNormalization()
        self.dense2 = layers.Dense(hidden_dim, activation='relu')
        self.norm2 = layers.LayerNormalization()
        final_units = output_units if output_units is not None else hidden_dim
        self.out_dense = layers.Dense(final_units, activation=None)
        self.norm_out = layers.LayerNormalization()

    def call(self, x, training=False):
        h = self.dense1(x)
        h = self.norm1(h)
        h = self.dense2(h)
        h = self.norm2(h)
        out = self.out_dense(h)
        out = self.norm_out(out)
        return out

# =============================================================================
# 가우시안 노이즈 추가 함수
def add_diffusion_noise(x, stddev=0.05):
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=stddev)
    return x + noise

# =============================================================================
# Frequency branch CNN block with pre- and post-diffusion
def build_freq_branch_with_diffusion(input_shape=(200,8,4), noise_std=0.05):
    inp = layers.Input(shape=input_shape)
    # Pre-CNN: 노이즈 추가 (Residual 연결 없음)
    pre_processed = layers.Lambda(lambda x: add_diffusion_noise(x, stddev=noise_std))(inp)
    # CNN block
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(pre_processed)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    freq_feat = x
    # Post-diffusion block
    noisy_freq_feat = layers.Lambda(lambda x: add_diffusion_noise(x, stddev=noise_std))(freq_feat)
    diffu_freq = DiffuSSMLayer(hidden_dim=64)(noisy_freq_feat)
    freq_res = layers.Add()([freq_feat, diffu_freq])
    branch = models.Model(inputs=inp, outputs=freq_res, name="FreqBranchDiff")
    return branch

# =============================================================================
# Channel branch CNN block with pre- and post-diffusion
def build_chan_branch_with_diffusion(input_shape=(200,4,8), noise_std=0.05):
    inp = layers.Input(shape=input_shape)
    # Pre-CNN: 노이즈 추가 (Residual 연결 없음)
    pre_processed = layers.Lambda(lambda x: add_diffusion_noise(x, stddev=noise_std))(inp)
    # CNN block
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(pre_processed)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    chan_feat = x
    # Post-diffusion block
    noisy_chan_feat = layers.Lambda(lambda x: add_diffusion_noise(x, stddev=noise_std))(chan_feat)
    diffu_chan = DiffuSSMLayer(hidden_dim=64)(noisy_chan_feat)
    chan_res = layers.Add()([chan_feat, diffu_chan])
    branch = models.Model(inputs=inp, outputs=chan_res, name="ChanBranchDiff")
    return branch

# =============================================================================
# 전체 복원 모델 구성: 두 branch 결합 후 원본 EEG 신호 복원 (입출력 shape: (4,200,8))
def build_diffusion_restoration_model(noise_std=0.05):
    input_signal = layers.Input(shape=(4,200,8), name="EEG_Input")
    # Frequency branch: (4,200,8) -> (200,8,4)
    freq_input = layers.Permute((2,3,1), name="Freq_Permute")(input_signal)
    freq_branch = build_freq_branch_with_diffusion(input_shape=(200,8,4), noise_std=noise_std)
    freq_features = freq_branch(freq_input)
    # Channel branch: (4,200,8) -> (200,4,8)
    chan_input = layers.Permute((2,1,3), name="Chan_Permute")(input_signal)
    chan_branch = build_chan_branch_with_diffusion(input_shape=(200,4,8), noise_std=noise_std)
    chan_features = chan_branch(chan_input)
    # 두 branch의 feature 결합 (각각 64 차원 → 128 차원)
    combined_features = layers.Concatenate()([freq_features, chan_features])
    
    # 복원기: Dense 레이어를 통해 최종 (4,200,8) 형태의 신호 복원
    x = layers.Dense(256, activation='relu')(combined_features)
    x = layers.Dense(6400, activation=None)(x)  # 4*200*8 = 6400
    output_signal = layers.Reshape((4,200,8))(x)
    
    model = models.Model(inputs=input_signal, outputs=output_signal, name="Diffusion_Restoration_Model")
    return model

# =============================================================================
# CSP feature 로드 및 1초 단위 분할 (원본 신호는 (4,200,8))
def load_csp_features(csp_feature_dir="/home/bcml1/2025_EMOTION/DEAP_eeg_new_label_CSP"):
    X_list, Y_list, subjects_list, files_list = [], [], [], []
    for file_name in os.listdir(csp_feature_dir):
        if not file_name.endswith(".npy"):
            continue
        file_path = os.path.join(csp_feature_dir, file_name)
        try:
            data = np.load(file_path)  # 예상 shape: (4, T, 8)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
        if data.ndim != 3 or data.shape[0] != 4 or data.shape[2] != 8:
            print(f"Unexpected shape in {file_path}: {data.shape}")
            continue

        # 디버깅용 label, subject 추출
        label_match = re.search(r'label(\d+)', file_name, re.IGNORECASE)
        if label_match is None:
            print(f"No label found in {file_name}, skipping file.")
            continue
        label = int(label_match.group(1))
        if label not in [0, 1, 2, 3]:
            print(f"Label {label} in file {file_name} is not in [0,1,2,3], skipping file.")
            continue

        subject_match = re.search(r'subject(\d+)', file_name, re.IGNORECASE)
        subject = subject_match.group(1) if subject_match else 'unknown'
        T = data.shape[1]
        n_windows = T // 200  # 1초 = 200 샘플
        if n_windows < 1:
            continue
        for i in range(n_windows):
            window = data[:, i*200:(i+1)*200, :]  # (4,200,8)
            X_list.append(window)
            Y_list.append(label)  # 복원 모델에서는 사용하지 않음
            files_list.append(f"{file_name}_win{i}")
            subjects_list.append(subject)
    X = np.array(X_list)
    Y = np.array(Y_list)
    subjects = np.array(subjects_list)
    file_ids = np.array(files_list)

    print(f"Loaded {X.shape[0]} samples, each shape: {X.shape[1:]}")
    print(f"Unique labels found (for debug): {np.unique(Y)}")
    print(f"Unique subjects found: {np.unique(subjects)}")
    return X, Y, subjects, file_ids

# =============================================================================
# Intra-subject 복원 학습, 평가 및 1D 파형 시각화 (신호 파형 방식)
if __name__ == "__main__":
    CSP_FEATURE_DIR = "/home/bcml1/2025_EMOTION/DEAP_eeg_new_label_CSP"
    X, Y, subjects, file_ids = load_csp_features(CSP_FEATURE_DIR)
    
    unique_subjects = np.unique(subjects)
    for subj in unique_subjects:
        # 예제: subject 번호 12~32번에 대해 학습 수행
        if not (12 <= int(subj) < 33):
            continue
        print(f"\n========== Intra-subject Restoration: Subject = {subj} ==========")
        subj_mask = (subjects == subj)
        X_subj = X[subj_mask]
        print(f"Subject {subj}: Total samples = {X_subj.shape[0]}")
        
        # train/validation/test 분할 (랜덤 분할)
        X_train, X_temp = train_test_split(X_subj, test_size=0.3, random_state=42)
        X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)
        print(f"Subject {subj}: Train {X_train.shape[0]} samples, Val {X_val.shape[0]} samples, Test {X_test.shape[0]} samples.")
        
        subj_folder = os.path.join(RESULT_DIR, f"s{subj}")
        os.makedirs(subj_folder, exist_ok=True)
        
        # 복원 문제: 입력과 타겟 모두 원본 신호
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, X_train)).shuffle(1000).batch(16)
        val_dataset   = tf.data.Dataset.from_tensor_slices((X_val, X_val)).batch(16)
        test_dataset  = tf.data.Dataset.from_tensor_slices((X_test, X_test)).batch(16)
        
        # 모델 생성 및 학습
        model = build_diffusion_restoration_model(noise_std=0.02)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=1e-4, decay_steps=100000, decay_rate=0.9, staircase=True),
            clipnorm=1.0
        )
        early_stopping = EarlyStopping(monitor='val_loss', patience=2000, mode='min', restore_best_weights=True)
        model.compile(optimizer=optimizer,
                      loss="mse",
                      metrics=["mse"])
        model.summary()
        
        history = model.fit(train_dataset, epochs=500, validation_data=val_dataset, callbacks=[early_stopping])
        
        # 학습 곡선 시각화
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.title('Loss Curve')
        
        plt.subplot(1,2,2)
        plt.plot(history.history['mse'], label='Train MSE')
        plt.plot(history.history['val_mse'], label='Val MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        plt.title('MSE Curve')
        train_curve_path = os.path.join(subj_folder, "training_curves.png")
        plt.savefig(train_curve_path)
        plt.close()
        print(f"Subject {subj}: Training curves saved to {train_curve_path}")
        
        # 테스트: 모델 복원 성능 평가
        test_loss, test_mse = model.evaluate(test_dataset)
        print(f"Subject {subj}: Test Loss (MSE): {test_loss:.6f}")
        
        # 예측: 복원 결과 산출
        X_test_pred = model.predict(test_dataset)
        
        # 유사도 측정: 각 샘플별 MSE 및 피어슨 상관계수 계산
        mse_list = []
        corr_list = []
        for original, restored in zip(X_test, X_test_pred):
            mse_val = np.mean((original - restored) ** 2)
            mse_list.append(mse_val)
            orig_flat = original.flatten()
            restored_flat = restored.flatten()
            corr_val = np.corrcoef(orig_flat, restored_flat)[0, 1]
            corr_list.append(corr_val)
        avg_mse = np.mean(mse_list)
        avg_corr = np.mean(corr_list)
        
        similarity_report = f"Subject {subj} Restoration Similarity Report\n"
        similarity_report += f"Test Samples: {len(mse_list)}\n"
        similarity_report += f"Average MSE: {avg_mse:.6f}\n"
        similarity_report += f"Average Pearson Correlation: {avg_corr:.6f}\n"
        print(similarity_report)
        sim_report_path = os.path.join(subj_folder, "restoration_similarity.txt")
        with open(sim_report_path, "w") as f:
            f.write(similarity_report)
        print(f"Subject {subj}: Restoration similarity report saved to {sim_report_path}")
        
        # -------------------------------------------
        # 시각화: test 데이터 샘플 중 일부에 대해 원본과 복원 신호의 1D 파형 시각화
        # 각 샘플의 shape: (4,200,8) → 4 밴드 (알파, 베타, 감마, 세타) × 8 채널의 1D 파형 (200 포인트)
        # 각 테스트 샘플마다 하나의 figure를 생성하여, 각 행은 하나의 주파수 밴드에 해당하며,
        # 좌측 subplot: 원본 파형, 우측 subplot: 복원 파형 (각 subplot에는 8채널의 파형이 overlay 됩니다)
        freq_labels = ["Alpha", "Beta", "Gamma", "Theta"]
        num_viz = min(5, X_test.shape[0])
        for idx in range(num_viz):
            original_sample = X_test[idx]   # shape: (4,200,8)
            restored_sample = X_test_pred[idx]  # shape: (4,200,8)
            fig, axs = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
            time_axis = np.arange(200)
            for j in range(4):
                # 좌측: 원본 신호, 우측: 복원 신호, 해당 주파수 밴드 j에 대해 8채널 파형을 개별 plot
                ax_orig = axs[j, 0]
                ax_rest = axs[j, 1]
                for ch in range(8):
                    ax_orig.plot(time_axis, original_sample[j, :, ch], label=f"Ch {ch+1}" if ch==0 else "")
                    ax_rest.plot(time_axis, restored_sample[j, :, ch], label=f"Ch {ch+1}" if ch==0 else "")
                ax_orig.set_title(f"{freq_labels[j]} - Original")
                ax_rest.set_title(f"{freq_labels[j]} - Restored")
                if j == 3:
                    ax_orig.set_xlabel("Time (points)")
                    ax_rest.set_xlabel("Time (points)")
                ax_orig.set_ylabel("Amplitude")
                ax_rest.set_ylabel("Amplitude")
                if j == 0:
                    ax_orig.legend(loc="upper right", fontsize="small")
                    ax_rest.legend(loc="upper right", fontsize="small")
            plt.tight_layout()
            viz_path = os.path.join(subj_folder, f"test_signal_waveform_sample{idx}.png")
            plt.savefig(viz_path)
            plt.close()
            print(f"Subject {subj}: Test signal waveform visualization for sample {idx} saved to {viz_path}")
        # -------------------------------------------
        
        # 모델 저장
        model_save_path = os.path.join(subj_folder, "model_eeg_cnn_diffussm_restoration.keras")
        model.save(model_save_path)
        print(f"Subject {subj}: Model saved to {model_save_path}")
