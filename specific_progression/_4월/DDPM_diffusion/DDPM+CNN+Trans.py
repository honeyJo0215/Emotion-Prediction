import os
import re
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 결과 저장 경로 설정
RESULT_DIR = "/home/bcml1/sigenv/_4월/eeg_signal_diffusion/eeg_DDPM_Transformer_Emotion"
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

        # label 추출: "label{num}" 형식, 감정 분류의 경우 0,1,2,3 등의 라벨만 사용
        label_match = re.search(r'label(\d+)', file_name, re.IGNORECASE)
        if label_match is None:
            print(f"No label found in {file_name}, skipping file.")
            continue
        label = int(label_match.group(1))
        if label not in [0, 1, 2, 3]:
            print(f"Label {label} in file {file_name} is not in [0,1,2,3], skipping file.")
            continue

        # subject 추출
        subject_match = re.search(r'subject(\d+)', file_name, re.IGNORECASE)
        subject = subject_match.group(1) if subject_match else 'unknown'
        T = data.shape[1]
        n_windows = T // 200  # 1초 = 200 샘플
        if n_windows < 1:
            continue
        for i in range(n_windows):
            window = data[:, i*200:(i+1)*200, :]  # (4,200,8)
            X_list.append(window)
            Y_list.append(label)
            files_list.append(f"{file_name}_win{i}")
            subjects_list.append(subject)
    X = np.array(X_list)
    Y = np.array(Y_list)
    subjects = np.array(subjects_list)
    file_ids = np.array(files_list)
    print(f"Loaded {X.shape[0]} samples, each shape: {X.shape[1:]}")
    print(f"Unique labels found: {np.unique(Y)}")
    print(f"Unique subjects found: {np.unique(subjects)}")
    return X, Y, subjects, file_ids

# =============================================================================
# DDPM 복원 파트를 위한 모델 구성
def build_encoder():
    # 입력: (200,8,4) → 출력 latent: (50,2,64)
    inp = layers.Input(shape=(200,8,4))
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inp)
    x = layers.MaxPooling2D((2,2))(x)  # (100,4,32)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)  # (50,2,64)
    model = models.Model(inputs=inp, outputs=x, name="Encoder")
    return model

def build_ddpm_block():
    # 입력 및 출력: (50,2,64)
    inp = layers.Input(shape=(50,2,64))
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(inp)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    out = layers.Conv2D(64, (3,3), activation=None, padding='same')(x)
    model = models.Model(inputs=inp, outputs=out, name="DDPM_Block")
    return model

def build_decoder():
    # 입력: (50,2,64) → 출력: (200,8,4)
    inp = layers.Input(shape=(50,2,64))
    x = layers.UpSampling2D(size=(2,2))(inp)  # (100,4,64)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D(size=(2,2))(x)       # (200,8,32)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    out = layers.Conv2D(4, (3,3), activation=None, padding='same')(x)
    model = models.Model(inputs=inp, outputs=out, name="Decoder")
    return model

# =============================================================================
# Transformer 기반 감정 분류를 위한 분류 블록을 포함한 모델 (DDPM 복원+CNN+Transformer)
class EmotionDDPMTransformerModel(tf.keras.Model):
    def __init__(self, encoder, ddpm_block, decoder, num_classes,
                 alpha=0.9, lambda_diff=1.0, noise_std=0.02, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.lambda_diff = lambda_diff
        self.sqrt_alpha = math.sqrt(alpha)
        self.sqrt_one_minus_alpha = math.sqrt(1 - alpha)
        # DDPM 복원 파트
        self.encoder = encoder
        self.ddpm_block = ddpm_block
        self.decoder = decoder
        
        # 분류를 위한 Transformer 기반 블록
        # 복원된 신호는 (batch,4,200,8) → 이를 (batch,200,32) 형태로 재구성
        self.cnn_conv1 = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')
        self.cnn_conv2 = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')
        
        # 간단한 Transformer Encoder Block
        self.mha = layers.MultiHeadAttention(num_heads=4, key_dim=64)
        self.dropout_transformer = layers.Dropout(0.1)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = models.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dense(64)
        ])
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.global_avg_pool = layers.GlobalAveragePooling1D()
        self.classifier_out = layers.Dense(num_classes, activation='softmax')
    
    def call(self, x, training=False):
        # x: (batch,4,200,8)
        # ========= DDPM 복원 파트 =========
        # (batch,4,200,8) → (batch,200,8,4)
        x_perm = tf.transpose(x, perm=[0,2,3,1])
        # 원본 latent
        latent_x0 = self.encoder(x_perm)   # (batch,50,2,64)
        # 노이즈 생성 및 noisy input
        epsilon = tf.random.normal(tf.shape(x_perm))
        x_t = self.sqrt_alpha * x_perm + self.sqrt_one_minus_alpha * epsilon
        latent_t = self.encoder(x_t)         # (batch,50,2,64)
        # DDPM Block: 노이즈 예측
        epsilon_hat = self.ddpm_block(latent_t)  # (batch,50,2,64)
        # target noise in latent: (latent_t - √α*latent_x0)/√(1-α)
        epsilon_target = (latent_t - self.sqrt_alpha * latent_x0) / self.sqrt_one_minus_alpha
        # 복원된 latent
        latent0_hat = (latent_t - self.sqrt_one_minus_alpha * epsilon_hat) / self.sqrt_alpha
        # Decoder를 통해 복원 (batch,200,8,4)
        x0_hat = self.decoder(latent0_hat)
        # (batch,200,8,4) → (batch,4,200,8)
        restored = tf.transpose(x0_hat, perm=[0,3,1,2])
        # ========= Classification (CNN + Transformer) =========
        # restored: (batch,4,200,8)
        # Rearrange: (batch, 200, 4, 8) → reshape to (batch, 200, 32)
        x_cls = tf.transpose(restored, perm=[0,2,1,3])
        x_cls = tf.reshape(x_cls, (-1, 200, 32))
        # 1D CNN layers
        x_cls = self.cnn_conv1(x_cls)  # (batch,200,64)
        x_cls = self.cnn_conv2(x_cls)  # (batch,200,64)
        # Transformer Encoder Block
        attn_output = self.mha(x_cls, x_cls)
        attn_output = self.dropout_transformer(attn_output, training=training)
        out1 = self.layernorm1(x_cls + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout_transformer(ffn_output, training=training)
        x_transformer = self.layernorm2(out1 + ffn_output)
        x_transformer = self.global_avg_pool(x_transformer)  # (batch, feature_dim)
        class_output = self.classifier_out(x_transformer)    # (batch, num_classes)
        return class_output, restored, epsilon_hat, epsilon_target

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            class_output, restored, epsilon_hat, epsilon_target = self(x, training=True)
            loss_cls = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, class_output))
            loss_recon = tf.reduce_mean(tf.keras.losses.mse(x, restored))
            loss_diff = tf.reduce_mean(tf.keras.losses.mse(epsilon_target, epsilon_hat))
            loss = loss_cls + loss_recon + self.lambda_diff * loss_diff
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        acc = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(y, class_output))
        return {"loss": loss, "loss_cls": loss_cls, "loss_recon": loss_recon, "loss_diff": loss_diff, "accuracy": acc}

    def test_step(self, data):
        x, y = data
        class_output, restored, epsilon_hat, epsilon_target = self(x, training=False)
        loss_cls = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, class_output))
        loss_recon = tf.reduce_mean(tf.keras.losses.mse(x, restored))
        loss_diff = tf.reduce_mean(tf.keras.losses.mse(epsilon_target, epsilon_hat))
        loss = loss_cls + loss_recon + self.lambda_diff * loss_diff
        acc = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(y, class_output))
        return {"loss": loss, "loss_cls": loss_cls, "loss_recon": loss_recon, "loss_diff": loss_diff, "accuracy": acc}

# =============================================================================
# Main Script: subject별 데이터 분할 후 학습, 평가, 시각화, 저장
if __name__ == "__main__":
    CSP_FEATURE_DIR = "/home/bcml1/2025_EMOTION/DEAP_eeg_new_label_CSP"
    X, Y, subjects, file_ids = load_csp_features(CSP_FEATURE_DIR)
    
    unique_subjects = np.unique(subjects)
    for subj in unique_subjects:
        # 예제: subject 번호 12 ~ 32번 사용
        if not (12 <= int(subj) < 33):
            continue
        print(f"\n========== Intra-subject Emotion Classification: Subject = {subj} ==========")
        subj_mask = (subjects == subj)
        X_subj = X[subj_mask]
        Y_subj = Y[subj_mask]
        # 만약 해당 subject의 라벨이 일부만 있다면 재매핑
        unique_labels_subj = np.sort(np.unique(Y_subj))
        label_map = {old: new for new, old in enumerate(unique_labels_subj)}
        Y_subj = np.array([label_map[y] for y in Y_subj])
        num_classes = len(unique_labels_subj)
        print(f"Using labels: {unique_labels_subj} remapped to 0 ~ {num_classes-1}")
        
        # stratified split
        X_train, X_temp, Y_train, Y_temp = train_test_split(X_subj, Y_subj, test_size=0.3, random_state=42, stratify=Y_subj)
        X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42, stratify=Y_temp)
        print(f"Subject {subj}: Train {X_train.shape[0]} samples, Val {X_val.shape[0]} samples, Test {X_test.shape[0]} samples.")
        print("Train label distribution:", np.unique(Y_train, return_counts=True))
        print("Val label distribution:", np.unique(Y_val, return_counts=True))
        print("Test label distribution:", np.unique(Y_test, return_counts=True))
        
        # 간단한 증강 (training 데이터에만 적용)
        unique_labels_train, counts_train = np.unique(Y_train, return_counts=True)
        max_count_train = counts_train.max()
        print(f"Balancing training classes to {max_count_train} samples per label using random noise augmentation.")
        augmented_train_X, augmented_train_Y = [], []
        for label in unique_labels_train:
            label_mask = (Y_train == label)
            current_count = np.sum(label_mask)
            num_to_augment = max_count_train - current_count
            if num_to_augment > 0:
                indices = np.where(label_mask)[0]
                for i in range(num_to_augment):
                    idx = np.random.choice(indices)
                    sample = X_train[idx]
                    sample_tensor = tf.convert_to_tensor(sample, dtype=tf.float32)
                    augmented = sample_tensor + tf.random.normal(tf.shape(sample_tensor), mean=0.0, stddev=0.05)
                    augmented_train_X.append(augmented.numpy())
                    augmented_train_Y.append(label)
        if augmented_train_X:
            X_train = np.concatenate([X_train, np.array(augmented_train_X)], axis=0)
            Y_train = np.concatenate([Y_train, np.array(augmented_train_Y)], axis=0)
            print(f"After augmentation (train): {X_train.shape[0]} samples.")
        
        subj_folder = os.path.join(RESULT_DIR, f"s{subj}")
        os.makedirs(subj_folder, exist_ok=True)
        
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(1000).batch(16)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(16)
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(16)
        
        # 모델 구성: DDPM 복원 + CNN + Transformer 기반 감정 분류
        encoder = build_encoder()
        ddpm_block = build_ddpm_block()
        decoder = build_decoder()
        model = EmotionDDPMTransformerModel(encoder, ddpm_block, decoder, num_classes=num_classes,
                                             alpha=0.9, lambda_diff=1.0, noise_std=0.02)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=1e-4, decay_steps=100000, decay_rate=0.9, staircase=True),
            clipnorm=1.0
        )
        model.compile(optimizer=optimizer)
        model.build(input_shape=(None,4,200,8))
        model.summary()
        
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=2000, mode='max', restore_best_weights=True)
        history = model.fit(train_dataset, epochs=500, validation_data=val_dataset, callbacks=[early_stopping])
        
        # 학습 곡선 시각화
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curve')
        plt.subplot(1,2,2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy Curve')
        train_curve_path = os.path.join(subj_folder, "training_curves.png")
        plt.savefig(train_curve_path)
        plt.close()
        print(f"Subject {subj}: Training curves saved to {train_curve_path}")
        
        # 테스트 평가
        test_metrics = model.evaluate(test_dataset, return_dict=True)
        print(f"Subject {subj}: Test Loss: {test_metrics['loss']:.4f}, Test Accuracy: {test_metrics['accuracy']:.4f}")
        
        # 예측 및 Confusion Matrix, Classification Report
        y_pred_prob = model.predict(test_dataset)[0]
        y_pred = np.argmax(y_pred_prob, axis=1)
        cm = confusion_matrix(Y_test, y_pred)
        report = classification_report(Y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(report)
        
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        cm_path = os.path.join(subj_folder, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        print(f"Subject {subj}: Confusion matrix saved to {cm_path}")
        
        report_path = os.path.join(subj_folder, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Subject {subj}: Classification report saved to {report_path}")
        
        # 모델 저장
        model_save_path = os.path.join(subj_folder, "model_eeg_emotion_ddpm_transformer.keras")
        model.save(model_save_path)
        print(f"Subject {subj}: Model saved to {model_save_path}")
