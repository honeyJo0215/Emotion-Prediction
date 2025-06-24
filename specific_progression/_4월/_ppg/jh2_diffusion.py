import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import logging

# =============================================================================
# 결과 저장 경로 설정 (PPG용)
RESULT_DIR = "/home/bcml1/sigenv/_4월/_ppg/rx_diffnoise_CNN_diff2"
os.makedirs(RESULT_DIR, exist_ok=True)

# =============================================================================
# GPU 메모리 제한 (필요시 사용)
def limit_gpu_memory(memory_limit_mib=6000):
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
limit_gpu_memory(6000)

# =============================================================================
# DiffuSSMLayer: Dense와 LayerNormalization을 이용한 노이즈 복원 모듈  
# (flatten한 입력을 원래 shape로 복원할 때 사용)
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
# 가우시안 노이즈 추가 함수 (tf 기반, 모델 내 사용)
def add_diffusion_noise(x, stddev):
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=stddev)
    return x + noise

# =============================================================================
# [Data Augmentation Functions] - numpy 기반, 데이터 로드 시 augmentation 적용
# 1. 원본
def aug_identity(signal):
    return signal

# 2. Diffusion noise (유사한 방식으로 노이즈 추가)
def aug_diffusion(signal, noise_std=0.02):
    noise = np.random.normal(0, noise_std, size=signal.shape)
    return signal + noise

# 3. Gaussian noise (약간 더 강한 노이즈)
def aug_gaussian(signal, noise_std=0.05):
    noise = np.random.normal(0, noise_std, size=signal.shape)
    return signal + noise

# 4. Scaling (진폭 스케일 변경)
def aug_scaling(signal, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(*scale_range)
    return signal * scale

# 5. Time shift (시간 축 이동)
def aug_time_shift(signal, shift_max=10):
    shift = np.random.randint(-shift_max, shift_max + 1)
    if shift > 0:
        pad = np.zeros((shift, signal.shape[1]))
        return np.concatenate([pad, signal[:-shift, :]], axis=0)
    elif shift < 0:
        pad = np.zeros((-shift, signal.shape[1]))
        return np.concatenate([signal[-shift:, :], pad], axis=0)
    else:
        return signal

# 6. Inversion (신호 반전)
def aug_inversion(signal):
    return -signal

# 7. Jitter (작은 랜덤 잡음 추가)
def aug_jitter(signal, jitter_std=0.01):
    noise = np.random.normal(0, jitter_std, size=signal.shape)
    return signal + noise

# 8. Smoothing (이동평균 필터 적용)
def aug_smoothing(signal, kernel_size=5):
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.stack([np.convolve(signal[:, i], kernel, mode='same') for i in range(signal.shape[1])], axis=1)
    return smoothed

# 9. Dropout (일부 값 0 처리)
def aug_dropout(signal, dropout_rate=0.1):
    mask = np.random.binomial(1, 1 - dropout_rate, size=signal.shape)
    return signal * mask

# 10. Combined (scaling 후 diffusion noise 추가)
def aug_combined(signal):
    scaled = aug_scaling(signal, scale_range=(0.9, 1.1))
    return aug_diffusion(scaled, noise_std=0.02)

# List of augmentation functions (총 10가지)
augmentation_functions = [
    aug_identity,
    aug_diffusion,
    aug_gaussian,
    aug_scaling,
    aug_time_shift,
    aug_inversion,
    aug_jitter,
    aug_smoothing,
    aug_dropout,
    aug_combined
]

# =============================================================================
# 데이터 augmentation 함수 (각 샘플당 10가지 augmentation 적용)
def augment_data(subject_data, augmentation_funcs):
    for subject in subject_data:
        X = subject_data[subject]['X']  # shape: (n, L, 3)
        y = subject_data[subject]['y']  # shape: (n,)
        augmented_X = []
        augmented_y = []
        for i in range(X.shape[0]):
            sample = X[i]
            label = y[i]
            # 각 augmentation 기법 적용 (총 len(augmentation_funcs)개)
            for func in augmentation_funcs:
                augmented_sample = func(sample)
                augmented_X.append(augmented_sample)
                augmented_y.append(label)
        subject_data[subject]['X'] = np.array(augmented_X)
        subject_data[subject]['y'] = np.array(augmented_y)
    return subject_data

# -----------------------------
# 2. Feature Extractor 정의 (각 브랜치에 사용)
def build_feature_extractor(input_length=None):
    inputs = tf.keras.Input(shape=(input_length, 3))
    x = layers.Conv1D(filters=8, kernel_size=7, strides=1, padding='same')(inputs)
    x = layers.ReLU(max_value=6)(x)
    x = layers.SeparableConv1D(filters=16, kernel_size=7, strides=4, padding='same')(x)
    x = layers.ReLU(max_value=6)(x)
    x = layers.SeparableConv1D(filters=32, kernel_size=7, strides=2, padding='same')(x)
    x = layers.ReLU(max_value=6)(x)
    x = layers.SeparableConv1D(filters=64, kernel_size=7, strides=4, padding='same')(x)
    x = layers.ReLU(max_value=6)(x)
    x = layers.SeparableConv1D(filters=128, kernel_size=7, strides=2, padding='same')(x)
    x = layers.ReLU(max_value=6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    model = models.Model(inputs=inputs, outputs=x)
    return model

# -----------------------------
# 3. EMCNN 모델 정의 (PPG용)
class EMCNN(tf.keras.Model):
    def __init__(self, num_classes=4, smoothing_kernel_size=2, downsample_factor=2,
                 input_length=None, internal_noise_std=0.02):
        """
        num_classes: 분류 클래스 수  
        smoothing_kernel_size: 이동평균 필터 커널 크기  
        downsample_factor: 다운샘플링 비율  
        input_length: 원본 시퀀스 길이 (예: 1280 for 10s)  
        internal_noise_std: 기본 diffusion noise 강도 (PPG의 경우 0.5~5Hz에 맞춰 낮은 강도, 여기서는 0.02)
        """
        super(EMCNN, self).__init__()
        self.num_classes = num_classes
        self.smoothing_kernel_size = smoothing_kernel_size
        self.downsample_factor = downsample_factor
        self.internal_noise_std = internal_noise_std
        self.input_length = input_length

        self.feature_identity = build_feature_extractor(input_length=input_length)
        self.feature_smoothing = build_feature_extractor(input_length=input_length)
        self.feature_downsampling = build_feature_extractor(input_length=input_length // downsample_factor)

        self.smoothing_conv = layers.Conv1D(filters=3, kernel_size=smoothing_kernel_size,
                                            padding='same', use_bias=False, groups=3)
        smoothing_weight = np.ones((smoothing_kernel_size, 1, 3), dtype=np.float32) / smoothing_kernel_size
        self.smoothing_conv.build((None, input_length, 3))
        self.smoothing_conv.set_weights([smoothing_weight])
        self.smoothing_conv.trainable = False

        self.identity_restoration = DiffuSSMLayer(hidden_dim=256, output_units=input_length * 3)
        self.smoothing_restoration = DiffuSSMLayer(hidden_dim=256, output_units=input_length * 3)
        self.downsampling_restoration = DiffuSSMLayer(hidden_dim=256, 
                                                      output_units=(input_length // downsample_factor) * 3)

        self.diffusion_layer = DiffuSSMLayer(hidden_dim=384, output_units=384)

        self.fc1 = layers.Dense(128, activation=lambda x: tf.nn.relu6(x))
        self.fc2 = layers.Dense(64, activation=lambda x: tf.nn.relu6(x))
        self.fc3 = layers.Dense(num_classes)

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        L = self.input_length

        # diffusion noise 강도를 학습 시 무작위로 조절 (data augmentation 효과)
        if training:
            noise_std_identity = tf.random.uniform([], minval=0.015, maxval=0.025)
            noise_std_smooth = tf.random.uniform([], minval=0.015, maxval=0.025)
            noise_std_down = tf.random.uniform([], minval=0.015, maxval=0.025)
        else:
            noise_std_identity = self.internal_noise_std
            noise_std_smooth = self.internal_noise_std
            noise_std_down = self.internal_noise_std

        # Identity branch:
        noisy_identity = add_diffusion_noise(inputs, stddev=noise_std_identity)
        flat_identity = tf.reshape(noisy_identity, [batch_size, L * 3])
        restored_identity = self.identity_restoration(flat_identity)
        restored_identity_signal = tf.reshape(restored_identity, [batch_size, L, 3])
        feat_identity = self.feature_identity(restored_identity_signal)

        # Smoothing branch:
        smooth_input = self.smoothing_conv(inputs)
        noisy_smooth = add_diffusion_noise(smooth_input, stddev=noise_std_smooth)
        flat_smooth = tf.reshape(noisy_smooth, [batch_size, L * 3])
        restored_smooth = self.smoothing_restoration(flat_smooth)
        restored_smooth_signal = tf.reshape(restored_smooth, [batch_size, L, 3])
        feat_smoothing = self.feature_smoothing(restored_smooth_signal)

        # Downsampling branch:
        down_input = inputs[:, ::self.downsample_factor, :]
        noisy_down = add_diffusion_noise(down_input, stddev=noise_std_down)
        down_length = L // self.downsample_factor
        flat_down = tf.reshape(noisy_down, [batch_size, down_length * 3])
        restored_down = self.downsampling_restoration(flat_down)
        restored_down_signal = tf.reshape(restored_down, [batch_size, down_length, 3])
        feat_downsampling = self.feature_downsampling(restored_down_signal)

        features = tf.concat([feat_identity, feat_smoothing, feat_downsampling], axis=1)
        diffused_features = self.diffusion_layer(features)

        x = self.fc1(diffused_features)
        x = self.fc2(x)
        logits = self.fc3(x)
        return logits

# =============================================================================
# 4. 데이터 로딩 함수 (PPG, 1s 버전)
# 파일 shape: (60, 5, 128) → 선택된 채널: (60, 128, 3)
def load_data_1s(data_dir, ch_choice_pair1=1, ch_choice_pair2=3):
    subject_data = {}
    file_paths = glob.glob(os.path.join(data_dir, "*.npy"))
    logging.info(f"총 {len(file_paths)}개의 파일을 찾았습니다. (1s)")
    for file_path in file_paths:
        base = os.path.basename(file_path)
        try:
            subject_id = base.split('_')[0]
            label = int(base.split('_')[-1].split('.')[0])
        except Exception as e:
            logging.error(f"라벨/서브젝트 추출 오류: {file_path}, {e}")
            continue
        data = np.load(file_path)  # (60, 5, 128)
        data = data[:, [0, ch_choice_pair1, ch_choice_pair2], :]  # (60, 3, 128)
        data = np.transpose(data, (0, 2, 1))  # (60, 128, 3)
        if subject_id not in subject_data:
            subject_data[subject_id] = {'X': [], 'y': []}
        subject_data[subject_id]['X'].append(data)
        subject_data[subject_id]['y'].append(np.full((data.shape[0],), label, dtype=np.int32))
    for subject in subject_data:
        subject_data[subject]['X'] = np.concatenate(subject_data[subject]['X'], axis=0)
        subject_data[subject]['y'] = np.concatenate(subject_data[subject]['y'], axis=0)
        logging.info(f"{subject} - X shape: {subject_data[subject]['X'].shape}, y shape: {subject_data[subject]['y'].shape}")
    return subject_data

# =============================================================================
# 5. 1s 데이터를 non-overlapping하게 그룹화하여 10s 샘플로 변환 (예: 60s면 6개 샘플)
def group_1s_to_10s(subject_data, segment_length=10):
    for subject in subject_data:
        X = subject_data[subject]['X']  # shape: (n, 128, 3)
        n = X.shape[0]
        num_segments = n // segment_length  # non-overlapping 그룹 개수
        X = X[:num_segments * segment_length]
        # reshape: (num_segments, segment_length, 128, 3)
        X = X.reshape(num_segments, segment_length, X.shape[1], X.shape[2])
        # concatenate time axis: 최종 shape: (num_segments, segment_length*128, 3)
        X = X.reshape(num_segments, segment_length * X.shape[2], X.shape[3])
        subject_data[subject]['X'] = X
        y = subject_data[subject]['y']
        y = y[:num_segments * segment_length].reshape(num_segments, segment_length)
        subject_data[subject]['y'] = y[:, 0]  # 각 그룹의 첫번째 라벨 사용
    return subject_data

# =============================================================================
# 6. 데이터 증강 함수: 각 샘플당 10가지 augmentation 적용하여 데이터 10배 증가
def augment_data(subject_data, augmentation_funcs):
    for subject in subject_data:
        X = subject_data[subject]['X']  # shape: (n, L, 3)
        y = subject_data[subject]['y']  # shape: (n,)
        augmented_X = []
        augmented_y = []
        for i in range(X.shape[0]):
            sample = X[i]
            label = y[i]
            for func in augmentation_funcs:
                augmented_sample = func(sample)
                augmented_X.append(augmented_sample)
                augmented_y.append(label)
        subject_data[subject]['X'] = np.array(augmented_X)
        subject_data[subject]['y'] = np.array(augmented_y)
    return subject_data

# =============================================================================
# 7. Intra-Subject Cross Validation 학습 함수
def train_model_for_subject(subject_id, X, y, num_classes=4, epochs=3000, batch_size=64):
    print(f"Subject {subject_id} 학습 시작")
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1765, stratify=y_train_val, random_state=42)
    # (70% train, 15% val, 15% test)

    result_dir = os.path.join(result_dir_base, subject_id)
    os.makedirs(result_dir, exist_ok=True)

    input_length = X.shape[1]  # 10s 모드의 경우 10*128 = 1280
    model = EMCNN(num_classes=num_classes, smoothing_kernel_size=2,
                  downsample_factor=2, input_length=input_length, internal_noise_std=0.02)
    model.compile(optimizer=tf.keras.optimizers.Adam(
                    learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=1e-4, decay_steps=100000, decay_rate=0.9, staircase=True),
                    clipnorm=1.0),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    
    model.summary()
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val, y_val), verbose=1)
    
    model.save(os.path.join(result_dir, f"{subject_id}_model.keras"))
    np.save(os.path.join(result_dir, f"{subject_id}_history.npy"), history.history)

    y_pred = np.argmax(model.predict(X_test), axis=1)
    eval_path = os.path.join(result_dir, f"{subject_id}_evaluation.txt")
    with open(eval_path, 'w') as f:
        f.write(classification_report(y_test, y_pred))
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, y_pred)))
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title(f"{subject_id} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(os.path.join(result_dir, f"{subject_id}_confusion_matrix.png"))
    plt.close()
    
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(result_dir, f"{subject_id}_loss_output.png"))
    plt.close()
    
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(result_dir, f"{subject_id}_accuracy_output.png"))
    plt.close()

# =============================================================================
# 8. 메인: 모드 선택 후 전체 피실험자에 대해 intra-subject cross validation 실행
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mode = '10s'  # '1s' 또는 '10s'
    if mode == '1s':
        data_dir = '/home/bcml1/2025_EMOTION/DEAP_PPG_1s'
        subject_data = load_data_1s(data_dir, ch_choice_pair1=1, ch_choice_pair2=3)
        # 1s 모드에서는 증강 후 사용 (데이터 10배 증가)
        subject_data = augment_data(subject_data, augmentation_functions)
        result_dir_base = '/home/bcml1/sigenv/_4월/_ppg/rx_diffnoise_CNN_diff2_intra_1s'
    elif mode == '10s':
        # 10s 모드에서는 1s 데이터를 이어붙여 사용 (overlap 없이)
        data_dir = '/home/bcml1/2025_EMOTION/DEAP_PPG_1s'
        subject_data = load_data_1s(data_dir, ch_choice_pair1=1, ch_choice_pair2=3)
        subject_data = group_1s_to_10s(subject_data, segment_length=10)  # 10초 샘플 생성 (예: 60s -> 6개)
        # 데이터 증강: 10배 증강 (각 10s 샘플 당 10가지 augmentation 적용)
        subject_data = augment_data(subject_data, augmentation_functions)
        result_dir_base = '/home/bcml1/sigenv/_4월/_ppg/rx_diffnoise_CNN_diff2_intra_10s'
    else:
        raise ValueError("mode는 '1s' 또는 '10s'여야 합니다.")
    
    for subject_id in sorted(subject_data.keys(), key=lambda x: int(x[1:])):
        X = subject_data[subject_id]['X']
        y = subject_data[subject_id]['y']
        logging.info(f"--- {subject_id} 학습 시작 ---")
        train_model_for_subject(subject_id, X, y, num_classes=4, epochs=100, batch_size=64)
