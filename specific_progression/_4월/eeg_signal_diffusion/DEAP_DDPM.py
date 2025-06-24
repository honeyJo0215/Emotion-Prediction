import os
import re
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# =============================================================================
# 결과 저장 경로 설정
RESULT_DIR = "/home/bcml1/sigenv/_4월/eeg_signal_diffusion/eeg_DDPM1"
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
# 데이터 로드 함수
def load_csp_features(csp_feature_dir="/home/bcml1/2025_EMOTION/DEAP_eeg_new_label_CSP"):
    """
    지정된 디렉토리 내의 .npy 파일들을 모두 로드하여,
    각 파일을 (4, T, 8)로 읽고 1초(200 샘플) 단위로 분할하여 (4,200,8) 형태의 샘플들을 생성합니다.
    """
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

        # label 및 subject 정보 (필요시 사용)
        label_match = re.search(r'label(\d+)', file_name, re.IGNORECASE)
        if label_match is None:
            continue
        label = int(label_match.group(1))
        if label not in [0, 1, 2, 3]:
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
            Y_list.append(label)  # 복원에서는 사용하지 않음
            files_list.append(f"{file_name}_win{i}")
            subjects_list.append(subject)
    X = np.array(X_list)
    Y = np.array(Y_list)
    subjects = np.array(subjects_list)
    file_ids = np.array(files_list)
    print(f"Loaded {X.shape[0]} samples, each shape: {X.shape[1:]}")
    return X, Y, subjects, file_ids

# =============================================================================
# 모델 구성
# Encoder: 입력 (200,8,4) → latent representation (50,2,64)
def build_encoder():
    inp = layers.Input(shape=(200,8,4))
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inp)
    x = layers.MaxPooling2D((2,2))(x)  # (100,4,32)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)  # (50,2,64)
    model = models.Model(inputs=inp, outputs=x, name="Encoder")
    return model

# DDPM Block: latent 공간에서 노이즈 예측 (입/출력 shape: (50,2,64))
def build_ddpm_block():
    inp = layers.Input(shape=(50,2,64))
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(inp)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    out = layers.Conv2D(64, (3,3), activation=None, padding='same')(x)
    model = models.Model(inputs=inp, outputs=out, name="DDPM_Block")
    return model

# Decoder: latent (50,2,64) → 복원 신호 (200,8,4)
def build_decoder():
    inp = layers.Input(shape=(50,2,64))
    x = layers.UpSampling2D(size=(2,2))(inp)  # (100,4,64)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D(size=(2,2))(x)       # (200,8,32)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    out = layers.Conv2D(4, (3,3), activation=None, padding='same')(x)
    model = models.Model(inputs=inp, outputs=out, name="Decoder")
    return model

# =============================================================================
# Custom Model: Encoder–DDPM–Decoder 통합
class DiffusionRestorationModel(tf.keras.Model):
    def __init__(self, encoder, ddpm_block, decoder, alpha=0.9, lambda_diff=1.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.ddpm_block = ddpm_block
        self.decoder = decoder
        self.alpha = alpha
        self.lambda_diff = lambda_diff
        self.sqrt_alpha = math.sqrt(alpha)
        self.sqrt_one_minus_alpha = math.sqrt(1 - alpha)
    
    def call(self, x, training=False):
        # x: (batch, 4,200,8)
        # 우선 (200,8,4)로 변환하여 원본 신호 x0를 얻음
        x_perm = tf.transpose(x, perm=[0, 2, 3, 1])
        # 원본 latent 표현
        latent_x0 = self.encoder(x_perm)  # (batch,50,2,64)
        # 원본 공간에서 노이즈 생성
        epsilon = tf.random.normal(tf.shape(x_perm))
        # noisy input: x_t = √α*x₀ + √(1–α)*ε
        x_t = self.sqrt_alpha * x_perm + self.sqrt_one_minus_alpha * epsilon
        # noisy latent: latent_t = encoder(x_t)
        latent_t = self.encoder(x_t)  # (batch,50,2,64)
        # DDPM block에서 latent_t의 노이즈 예측: ε̂
        epsilon_hat = self.ddpm_block(latent_t)  # (batch,50,2,64)
        # latent 공간 target noise: ε_target = (latent_t – √α * latent_x0)/√(1–α)
        epsilon_target = (latent_t - self.sqrt_alpha * latent_x0) / self.sqrt_one_minus_alpha
        # 복원된 latent: latent₀_hat = (latent_t – √(1–α)*ε̂)/√α
        latent0_hat = (latent_t - self.sqrt_one_minus_alpha * epsilon_hat) / self.sqrt_alpha
        # Decoder: latent₀_hat → 복원 신호, shape: (batch,200,8,4)
        x0_hat = self.decoder(latent0_hat)
        # 최종 출력: (batch,4,200,8)
        x0_hat_perm = tf.transpose(x0_hat, perm=[0, 3, 1, 2])
        return x0_hat_perm, epsilon_hat, epsilon_target

    def train_step(self, data):
        x = data
        with tf.GradientTape() as tape:
            x0_hat, epsilon_hat, epsilon_target = self(x, training=True)
            # 복원 손실: 원본과 복원 신호 간 MSE
            loss_recon = tf.reduce_mean(tf.keras.losses.mse(x, x0_hat))
            # DDPM 손실: latent 공간에서 실제 노이즈와 예측 노이즈 간 MSE
            loss_diff = tf.reduce_mean(tf.keras.losses.mse(epsilon_target, epsilon_hat))
            loss = loss_recon + self.lambda_diff * loss_diff
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss, "loss_recon": loss_recon, "loss_diff": loss_diff}
    
    def test_step(self, data):
        x = data
        x0_hat, epsilon_hat, epsilon_target = self(x, training=False)
        loss_recon = tf.reduce_mean(tf.keras.losses.mse(x, x0_hat))
        loss_diff = tf.reduce_mean(tf.keras.losses.mse(epsilon_target, epsilon_hat))
        loss = loss_recon + self.lambda_diff * loss_diff
        return {"loss": loss, "loss_recon": loss_recon, "loss_diff": loss_diff}

# =============================================================================
# Main Script: 학습, 평가, 1D 파형 시각화, 모델 저장
if __name__ == "__main__":
    # 데이터 로드
    CSP_FEATURE_DIR = "/home/bcml1/2025_EMOTION/DEAP_eeg_new_label_CSP"
    X, Y, subjects, file_ids = load_csp_features(CSP_FEATURE_DIR)
    
    unique_subjects = np.unique(subjects)
    # 예제: subject 번호 12~32번에 대해 학습 (소규모 샘플)
    for subj in unique_subjects:
        if not (12 <= int(subj) < 33):
            continue
        print(f"\n========== Intra-subject Restoration (DDPM) for Subject = {subj} ==========")
        subj_mask = (subjects == subj)
        X_subj = X[subj_mask]
        print(f"Subject {subj}: Total samples = {X_subj.shape[0]}")
        
        # train/validation/test 분할 (랜덤)
        X_train, X_temp = train_test_split(X_subj, test_size=0.3, random_state=42)
        X_val, X_test = train_test_split(X_temp, test_size=0.5, random_state=42)
        print(f"Subject {subj}: Train {X_train.shape[0]} samples, Val {X_val.shape[0]} samples, Test {X_test.shape[0]} samples.")
        
        subj_folder = os.path.join(RESULT_DIR, f"s{subj}")
        os.makedirs(subj_folder, exist_ok=True)
        
        # 데이터셋 생성 (입력과 타겟은 동일)
        train_dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(1000).batch(16)
        val_dataset = tf.data.Dataset.from_tensor_slices(X_val).batch(16)
        test_dataset = tf.data.Dataset.from_tensor_slices(X_test).batch(16)
        
        # 모델 구성: encoder, DDPM block, decoder
        encoder = build_encoder()
        ddpm_block = build_ddpm_block()
        decoder = build_decoder()
        model = DiffusionRestorationModel(encoder, ddpm_block, decoder, alpha=0.9, lambda_diff=1.0)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer=optimizer)
        model.summary()
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=200, mode='min', restore_best_weights=True)
        history = model.fit(train_dataset, epochs=500, validation_data=val_dataset, callbacks=[early_stopping])
        
        # 학습 곡선 시각화
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Overall Loss')
        plt.legend()
        plt.title('Overall Loss Curve')
        plt.subplot(1,2,2)
        plt.plot(history.history['loss_recon'], label='Recon Loss')
        plt.plot(history.history['loss_diff'], label='Diffusion Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Components')
        train_curve_path = os.path.join(subj_folder, "training_curves.png")
        plt.savefig(train_curve_path)
        plt.close()
        print(f"Subject {subj}: Training curves saved to {train_curve_path}")
        
        # 테스트 평가
        test_metrics = model.evaluate(test_dataset)
        print(f"Subject {subj}: Test Loss: {test_metrics[0]:.6f}")

        # 예측: 모델.predict는 (복원신호, 예측노이즈, target노이즈)를 튜플로 반환
        X_test_pred_tuple = model.predict(test_dataset)
        X_test_hat = X_test_pred_tuple[0]  # 복원된 신호, shape: (batch,4,200,8)
        
        # -------------------------------------------
        # 1D 파형 시각화: 각 테스트 샘플에 대해 각 주파수 밴드별 8채널 1D 파형을 그림
        freq_labels = ["Alpha", "Beta", "Gamma", "Theta"]
        num_viz = min(5, X_test.shape[0])
        for idx in range(num_viz):
            original_sample = X_test[idx]    # (4,200,8)
            restored_sample = X_test_hat[idx]  # (4,200,8)
            fig, axs = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
            time_axis = np.arange(200)
            for j in range(4):
                ax_orig = axs[j, 0]
                ax_rest = axs[j, 1]
                for ch in range(8):
                    ax_orig.plot(time_axis, original_sample[j, :, ch],
                                 label=f"Ch {ch+1}" if ch==0 else "")
                    ax_rest.plot(time_axis, restored_sample[j, :, ch],
                                 label=f"Ch {ch+1}" if ch==0 else "")
                ax_orig.set_title(f"{freq_labels[j]} - Original")
                ax_rest.set_title(f"{freq_labels[j]} - Restored")
                ax_orig.set_ylabel("Amplitude")
                ax_rest.set_ylabel("Amplitude")
                if j == 3:
                    ax_orig.set_xlabel("Time (points)")
                    ax_rest.set_xlabel("Time (points)")
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
        model_save_path = os.path.join(subj_folder, "model_eeg_ddpm_restoration.keras")
        model.save(model_save_path)
        print(f"Subject {subj}: Model saved to {model_save_path}")
