import os
import glob
import numpy as np
import tensorflow as tf
import scipy.signal as signal
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Conv1D, DepthwiseConv1D, Dense, GlobalAveragePooling1D, concatenate, BatchNormalization, Activation, Lambda, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical

# =============================================================================
# GPU 메모리 제한 (필요 시)
# =============================================================================
def limit_gpu_memory(memory_limit_mib=8000):
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

limit_gpu_memory(8000)

# =============================================================================
# EMCNN 모델 구성 (입력 shape: (1280, 5))
# =============================================================================
input_shape = (1280, 5)
num_classes = 4  # 예: HVHA, HVLA, LVLA, LVHA

# ReLU6 활성화 함수 (Keras의 ReLU 레이어에서 max_value 인자를 이용)
def relu6():
    return lambda x: tf.nn.relu6(x)

# 이동평균 평활화 레이어: multi-channel 입력에 대해 고정 가중치 평균 필터를 적용합니다.
# smoothing은 이미 적해서 인풋데이터에 저장되어 있음
# class MovingAverage(layers.Layer):
#     def __init__(self, kernel_size=2):
#         super(MovingAverage, self).__init__()
#         self.kernel_size = kernel_size
#         # 필터 가중치: (kernel_size, in_channels=1, out_channels=1)
#         self.kernel = tf.constant(1.0 / kernel_size, shape=(kernel_size, 1, 1))

#     def call(self, inputs):
#         # inputs shape: (batch, length, channels)
#         channels = tf.shape(inputs)[-1]
#         # 입력을 4차원 텐서로 확장: (batch, 1, length, channels)
#         x = tf.expand_dims(inputs, axis=1)
#         # 동적으로 커널 생성: shape = [1, kernel_size, channels, 1]
#         kernel = tf.ones([1, self.kernel_size, channels, 1], dtype=inputs.dtype) / self.kernel_size
#         # "SAME" 패딩을 사용하여 출력 크기가 입력과 동일하게 유지되도록 함
#         x_conv = tf.nn.depthwise_conv2d(x, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
#         # 4차원 텐서를 원래 shape인 (batch, length, channels)로 축소
#         return tf.squeeze(x_conv, axis=1)


# !!!!!5채널 input CNN 구현!!!!!!!!!!!!!
# 하나의 CNN 분기를 구현한 레이어 (각 분기는 동일한 구조이지만 가중치는 독립적입니다)
# class CNNBranch(layers.Layer):
#     def __init__(self):
#         super(CNNBranch, self).__init__()
#         # 블록 1: 표준 1D 컨볼루션
#         self.conv1 = layers.Conv1D(filters=8, kernel_size=7, strides=1, padding='same', activation=relu6())
        
#         # 블록 2: depthwise conv (groups=8) 후 pointwise conv (1x1)
#         self.dwconv2 = layers.Conv1D(filters=8, kernel_size=7, strides=4, padding='same', groups=8, activation=relu6())
#         self.pwconv2 = layers.Conv1D(filters=16, kernel_size=1, strides=1, padding='same', activation=relu6())
        
#         # 블록 3
#         self.dwconv3 = layers.Conv1D(filters=16, kernel_size=7, strides=2, padding='same', groups=16, activation=relu6())
#         self.pwconv3 = layers.Conv1D(filters=32, kernel_size=1, strides=1, padding='same', activation=relu6())
        
#         # 블록 4
#         self.dwconv4 = layers.Conv1D(filters=32, kernel_size=7, strides=4, padding='same', groups=32, activation=relu6())
#         self.pwconv4 = layers.Conv1D(filters=64, kernel_size=1, strides=1, padding='same', activation=relu6())
        
#         # 블록 5
#         self.dwconv5 = layers.Conv1D(filters=64, kernel_size=7, strides=2, padding='same', groups=64, activation=relu6())
#         self.pwconv5 = layers.Conv1D(filters=128, kernel_size=1, strides=1, padding='same', activation=relu6())
        
#         # Global average pooling: 시간 축을 1로 축소
#         self.global_pool = layers.GlobalAveragePooling1D()
    
#     def call(self, inputs):
#         x = self.conv1(inputs)              # 출력: (batch, L, 8)
#         x = self.dwconv2(x)                 # 출력: (batch, L/4, 8)
#         x = self.pwconv2(x)                 # 출력: (batch, L/4, 16)
#         x = self.dwconv3(x)                 # 출력: (batch, L/8, 16)
#         x = self.pwconv3(x)                 # 출력: (batch, L/8, 32)
#         x = self.dwconv4(x)                 # 출력: (batch, L/32, 32)
#         x = self.pwconv4(x)                 # 출력: (batch, L/32, 64)
#         x = self.dwconv5(x)                 # 출력: (batch, L/64, 64)
#         x = self.pwconv5(x)                 # 출력: (batch, L/64, 128)
#         x = self.global_pool(x)             # 출력: (batch, 128)
#         return x

# 전체 EMCNN 모델: 3분기(Identity, Smoothing, Downsampling)를 각각 전체 5채널 입력에 대해 적용
# # 전체 EMCNN 모델: 5개의 채널을 각각 CNN 분기를 통과시켜 특징 추출 후 결합
# class EMCNN(tf.keras.Model):
#     def __init__(self, num_classes=4):
#         super(EMCNN, self).__init__()
#         self.branch1 = CNNBranch()  # 원본 신호
#         self.branch2 = CNNBranch()  # Smoothing s=2
#         self.branch3 = CNNBranch()  # Smoothing s=3
#         self.branch4 = CNNBranch()  # Downsampling d=2
#         self.branch5 = CNNBranch()  # Downsampling d=3
#         self.fc1 = Dense(128, activation=relu6())
#         self.fc2 = Dense(64, activation=relu6())
#         self.classifier = Dense(num_classes)  # logits

#     def call(self, inputs):
#         # 5개의 입력 채널을 각각 CNN 분기에 전달
#         x1 = self.branch1(inputs[:, :, 0:1])  # 원본 신호
#         x2 = self.branch2(inputs[:, :, 1:2])  # Smoothing s=2
#         x3 = self.branch3(inputs[:, :, 2:3])  # Smoothing s=3
#         x4 = self.branch4(inputs[:, :, 3:4])  # Downsampling d=2
#         x5 = self.branch5(inputs[:, :, 4:5])  # Downsampling d=3
        
#         # 특징 벡터를 연결
#         features = concatenate([x1, x2, x3, x4, x5])
#         x = self.fc1(features)
#         x = self.fc2(x)
#         logits = self.classifier(x)
#         return logits

# # def build_emcnn():
# #     inputs = Input(shape=input_shape)
# #     outputs = EMCNN(num_classes=num_classes, smoothing_kernel_size=2, downsample_factor=2)(inputs)
# #     return Model(inputs=inputs, outputs=outputs)

# # 모델 생성 함수
# def build_emcnn(input_shape=input_shape, num_classes=num_classes):
#     inputs = Input(shape=input_shape)
#     outputs = EMCNN(num_classes=num_classes)(inputs)
#     return Model(inputs=inputs, outputs=outputs)


###############################################
######!!!!위의 5채널 버전과 비교해보기!!!!!!!
################################################
# 3채널 버전의 EMCNN
# 하나의 CNN 분기를 구현한 레이어 (각 분기는 동일한 구조이나 가중치는 독립적)
class CNNBranch(layers.Layer):
    def __init__(self):
        super(CNNBranch, self).__init__()
        # 블록 1: 표준 1D 컨볼루션 (커널 7, stride 1, 출력채널 8)
        self.conv1 = layers.Conv1D(filters=8, kernel_size=7, strides=1, padding='same', activation=tf.nn.relu6)
        
        # 블록 2: depthwise conv (groups=8, 커널 7, stride 4) 후 pointwise conv (커널 1, 출력채널 16)
        self.dwconv2 = layers.Conv1D(filters=8, kernel_size=7, strides=4, padding='same', groups=8, activation=tf.nn.relu6)
        self.pwconv2 = layers.Conv1D(filters=16, kernel_size=1, strides=1, padding='same', activation=tf.nn.relu6)
        
        # 블록 3: depthwise conv (groups=16, 커널 7, stride 2) 후 pointwise conv (커널 1, 출력채널 32)
        self.dwconv3 = layers.Conv1D(filters=16, kernel_size=7, strides=2, padding='same', groups=16, activation=tf.nn.relu6)
        self.pwconv3 = layers.Conv1D(filters=32, kernel_size=1, strides=1, padding='same', activation=tf.nn.relu6)
        
        # 블록 4: depthwise conv (groups=32, 커널 7, stride 4) 후 pointwise conv (커널 1, 출력채널 64)
        self.dwconv4 = layers.Conv1D(filters=32, kernel_size=7, strides=4, padding='same', groups=32, activation=tf.nn.relu6)
        self.pwconv4 = layers.Conv1D(filters=64, kernel_size=1, strides=1, padding='same', activation=tf.nn.relu6)
        
        # 블록 5: depthwise conv (groups=64, 커널 7, stride 2) 후 pointwise conv (커널 1, 출력채널 128)
        self.dwconv5 = layers.Conv1D(filters=64, kernel_size=7, strides=2, padding='same', groups=64, activation=tf.nn.relu6)
        self.pwconv5 = layers.Conv1D(filters=128, kernel_size=1, strides=1, padding='same', activation=tf.nn.relu6)
        
        # Global average pooling: 시간 축을 평균하여 고정 길이 벡터로 변환
        self.global_pool = layers.GlobalAveragePooling1D()
    
    def call(self, inputs):
        x = self.conv1(inputs)         # (batch, L, 8)
        x = self.dwconv2(x)            # (batch, L/4, 8)
        x = self.pwconv2(x)            # (batch, L/4, 16)
        x = self.dwconv3(x)            # (batch, L/8, 16)
        x = self.pwconv3(x)            # (batch, L/8, 32)
        x = self.dwconv4(x)            # (batch, L/32, 32)
        x = self.pwconv4(x)            # (batch, L/32, 64)
        x = self.dwconv5(x)            # (batch, L/64, 64)
        x = self.pwconv5(x)            # (batch, L/64, 128)
        x = self.global_pool(x)        # (batch, 128)
        return x
    
# GroupedChannelSelection: 5채널 입력에서 각 그룹별로 우수 채널을 선택하고,
# smoothing 그룹에는 moving average (윈도우 크기 s), downsampling 그룹에는 시간 downsampling (간격 d) 적용
class GroupedChannelSelection(layers.Layer):
    def __init__(self, smoothing_scale=2, downsampling_factor=2):
        super(GroupedChannelSelection, self).__init__()
        self.smoothing_scale = smoothing_scale  # s, 2 또는 3 선택 가능
        self.downsampling_factor = downsampling_factor  # d, 2 또는 3 선택 가능

    def call(self, inputs):
        # inputs shape: (batch, length, 5)
        # 원본 채널 (index 0)
        x_identity = inputs[:, :, 0:1]
        
        # smoothing 그룹: 채널 1~2
        smoothing_group = inputs[:, :, 1:3]  # shape: (batch, length, 2)
        smoothing_var = tf.math.reduce_variance(smoothing_group, axis=1)  # (batch, 2)
        _, smoothing_top_idx = tf.nn.top_k(smoothing_var, k=1)  # (batch, 1)
        smoothing_top_idx = smoothing_top_idx + 1  # 실제 채널 인덱스 (1 또는 2)
        smoothing_selected = tf.gather(inputs, smoothing_top_idx, axis=2, batch_dims=1)  # (batch, length, 1)
        
        # moving average smoothing 적용: kernel of shape (smoothing_scale, 1, 1)
        kernel = tf.ones((self.smoothing_scale, 1, 1)) / self.smoothing_scale
        # conv1D: stride=1, 'SAME' padding
        smoothing_transformed = tf.nn.conv1d(smoothing_selected, filters=kernel, stride=1, padding='SAME')
        
        # downsampling 그룹: 채널 3~4
        downsampling_group = inputs[:, :, 3:5]  # (batch, length, 2)
        downsampling_var = tf.math.reduce_variance(downsampling_group, axis=1)  # (batch, 2)
        _, downsampling_top_idx = tf.nn.top_k(downsampling_var, k=1)  # (batch, 1)
        downsampling_top_idx = downsampling_top_idx + 3  # 실제 채널 인덱스 (3 또는 4)
        downsampling_selected = tf.gather(inputs, downsampling_top_idx, axis=2, batch_dims=1)  # (batch, length, 1)
        
        # downsampling 변환: 시간 축에서 downsampling_factor 간격으로 선택
        downsampling_transformed = downsampling_selected[:, ::self.downsampling_factor, :]
        
        # 각 branch는 이후 GlobalAveragePooling를 통해 고정 크기 벡터로 변환되므로,
        # 여기서는 세 branch(원본, smoothed, downsampled)를 개별적으로 반환합니다.
        return x_identity, smoothing_transformed, downsampling_transformed

# 3채널 버전의 EMCNN: 입력은 5채널이지만, Lambda를 이용해 우수한 채널만 선택
class EMCNN_3ch(tf.keras.Model):
    def __init__(self, num_classes=4, smoothing_scale=2, downsampling_factor=2):
        super(EMCNN_3ch, self).__init__()
        self.group_selection = GroupedChannelSelection(smoothing_scale, downsampling_factor)
        self.branch_identity = CNNBranch()
        self.branch_smoothing = CNNBranch()
        self.branch_downsampling = CNNBranch()
        
        self.fc1 = layers.Dense(128, activation=tf.nn.relu6)
        self.fc2 = layers.Dense(64, activation=tf.nn.relu6)
        self.classifier = layers.Dense(num_classes)
    
    def call(self, inputs):
        # inputs shape: (batch, 1280, 5)
        # GroupedChannelSelection로 각 그룹별 채널 선택 및 변환 진행
        x_identity, x_smoothing, x_downsampling = self.group_selection(inputs)
        
        # 각 분기에 대해 CNNBranch 적용
        feat_identity = self.branch_identity(x_identity)       # (batch, 128)
        feat_smoothing = self.branch_smoothing(x_smoothing)       # (batch, 128)
        feat_downsampling = self.branch_downsampling(x_downsampling)  # (batch, 128)
        
        # 3개 분기의 특징 벡터를 concat (축=1 → 결과: (batch, 384))
        features = tf.concat([feat_identity, feat_smoothing, feat_downsampling], axis=1)
        x = self.fc1(features)
        x = self.fc2(x)
        logits = self.classifier(x)
        return logits

# 모델 생성 함수 (Functional API로 래핑)
def build_emcnn_3ch(input_shape=input_shape, num_classes=num_classes):
    inputs = Input(shape=input_shape)
    outputs = EMCNN_3ch(num_classes=num_classes)(inputs)
    return Model(inputs=inputs, outputs=outputs)

# # === MobileNet 기반 Feature Extraction CNN ===
# def emcnn_branch(input_layer):
#     """ EMCNN의 각 branch에 해당하는 CNN 블록 (MobileNet 기반) """
#     x = Conv1D(8, kernel_size=7, strides=1, padding='same', activation='relu',
#                kernel_regularizer=l2(0.001))(input_layer)
    
#     # Depthwise Separable Convolution 적용
#     x = DepthwiseConv1D(kernel_size=7, strides=4, padding='same', depth_multiplier=1,
#                         depthwise_regularizer=l2(0.001))(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Conv1D(16, kernel_size=1, strides=1, padding='same', activation='relu',
#                kernel_regularizer=l2(0.001))(x)

#     x = DepthwiseConv1D(kernel_size=7, strides=2, padding='same', depth_multiplier=1,
#                         depthwise_regularizer=l2(0.001))(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Conv1D(32, kernel_size=1, strides=1, padding='same', activation='relu',
#                kernel_regularizer=l2(0.001))(x)

#     x = DepthwiseConv1D(kernel_size=7, strides=4, padding='same', depth_multiplier=1,
#                         depthwise_regularizer=l2(0.001))(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Conv1D(64, kernel_size=1, strides=1, padding='same', activation='relu',
#                kernel_regularizer=l2(0.001))(x)

#     x = DepthwiseConv1D(kernel_size=7, strides=2, padding='same', depth_multiplier=1,
#                         depthwise_regularizer=l2(0.001))(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Conv1D(128, kernel_size=1, strides=1, padding='same', activation='relu',
#                kernel_regularizer=l2(0.001))(x)

#     # Global Average Pooling 적용
#     x = GlobalAveragePooling1D()(x)
#     return x

# # =============================================================================
# # EMCNN 전체 모델 (입력 shape: (1280, 5))
# # =============================================================================
# def build_emcnn():
#     inputs = Input(shape=input_shape)  # 각 sample의 shape: (1280, 5)

#     # Lambda Layer를 사용하여 각 채널(branch)를 추출 (axis=2에서 채널 추출)
#     branch1 = emcnn_branch(Lambda(lambda x: tf.expand_dims(x[:, :, 0], axis=-1))(inputs))
#     branch2 = emcnn_branch(Lambda(lambda x: tf.expand_dims(x[:, :, 1], axis=-1))(inputs))
#     branch3 = emcnn_branch(Lambda(lambda x: tf.expand_dims(x[:, :, 2], axis=-1))(inputs))
#     branch4 = emcnn_branch(Lambda(lambda x: tf.expand_dims(x[:, :, 3], axis=-1))(inputs))
#     branch5 = emcnn_branch(Lambda(lambda x: tf.expand_dims(x[:, :, 4], axis=-1))(inputs))
    
#     merged = concatenate([branch1, branch2, branch3, branch4, branch5])
#     x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(merged)
#     x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
#     x = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
#     outputs = Dense(num_classes, activation='softmax')(x)
#     model = Model(inputs=inputs, outputs=outputs)
#     return model

# =============================================================================
# 슬라이딩 윈도우 함수 (메모리 최적화)
# =============================================================================
def sliding_window_ppg(continuous_data, window_size, step):
    win_length = window_size * 128
    step_length = step * 128
    return np.array([continuous_data[i:i + win_length] 
                     for i in range(0, len(continuous_data) - win_length + 1, step_length)],
                    dtype=np.float32)
    
# =============================================================================
# PPG 데이터 파일 경로와 라벨 정보를 subject 단위로 저장하는 함수
# =============================================================================
def load_ppg_file_paths(ppg_data_dir):
    subject_data = {}
    file_paths = glob.glob(os.path.join(ppg_data_dir, "*.npy"))
    print(f"총 {len(file_paths)}개의 PPG 파일을 찾았습니다.")
    for file_path in file_paths:
        base = os.path.basename(file_path)
        try:
            subject_id = base.split('_')[0]
            label = int(base.split('_')[-1].split('.')[0])
        except Exception as e:
            print("라벨/서브젝트 추출 오류:", file_path, e)
            continue
        if subject_id not in subject_data:
            subject_data[subject_id] = []
        subject_data[subject_id].append((file_path, label))
    return subject_data

# =============================================================================
# tf.data.Dataset을 활용하여 파일을 on-demand로 로드하는 함수
# =============================================================================
def create_dataset_from_files(file_tuples, window_size, step, batch_size=32):
    def generator():
        for file_path, label in file_tuples:
            data = np.load(file_path).astype(np.float32)
            data = np.transpose(data, (0, 2, 1))  # (60, 128, 5)
            continuous_data = np.concatenate(data, axis=0)  # (7680, 5)
            windows = sliding_window_ppg(continuous_data, window_size, step)
            for window in windows:
                yield window, label

    dataset = tf.data.Dataset.from_generator(generator,
        output_signature=(
            tf.TensorSpec(shape=(window_size * 128, 5), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    # 캐싱 및 병렬 전처리 적용
    dataset = dataset.cache()
    dataset = dataset.map(lambda x, y: (x, y), num_parallel_calls=tf.data.AUTOTUNE)
    # dataset = dataset.repeat()  # 데이터 반복
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# =============================================================================
# 파일 단위로 균형을 맞추기 위한 오버샘플링 함수 (파일 튜플 기반)
# =============================================================================
def balance_file_tuples(file_tuples):
    from collections import defaultdict
    class_dict = defaultdict(list)
    for tup in file_tuples:
        class_dict[tup[1]].append(tup)
    max_count = max(len(v) for v in class_dict.values())
    balanced = []
    for label, items in class_dict.items():
        count = len(items)
        if count < max_count:
            extra = [items[i] for i in np.random.choice(len(items), max_count - count, replace=True)]
            items = items + extra
        balanced.extend(items)
    np.random.shuffle(balanced)
    return balanced

# =============================================================================
# PPG 데이터 증강 함수들 (augment 및 balance_ppg_dataset)
# =============================================================================
def augment_ppg(ppg, noise_std=0.01, shift_max=5, scale_range=(0.9, 1.1), mask_freq=10, mask_factor=0.2, peak_shift=3):
    augmented_ppg = np.zeros_like(ppg)
    for ch in range(ppg.shape[1]):
        ppg_ch = ppg[:, ch]
        ppg_noisy = ppg_ch + np.random.normal(0, noise_std, ppg_ch.shape)
        shift = np.random.randint(-shift_max, shift_max)
        ppg_shifted = np.roll(ppg_noisy, shift, axis=0)
        scale_factor = np.random.uniform(*scale_range)
        ppg_scaled = ppg_shifted * scale_factor
        fft_signal = np.fft.fft(ppg_scaled)
        fft_signal[:mask_freq] *= (1 - mask_factor)
        ppg_freq = np.real(np.fft.ifft(fft_signal))
        peaks, _ = signal.find_peaks(ppg_freq, distance=30)
        if len(peaks) > 0:
            shift_values = np.random.randint(-peak_shift, peak_shift, size=len(peaks))
            for i, s in zip(peaks, shift_values):
                if 0 <= i + s < len(ppg_freq):
                    ppg_freq[i] = ppg_freq[i + s]
        augmented_ppg[:, ch] = ppg_freq
    return augmented_ppg

def balance_ppg_dataset(data_X, _, y, noise_std_ppg=0.01):
    y_labels = np.argmax(y, axis=1)
    classes = np.unique(y_labels)
    counts = {cls: np.sum(y_labels == cls) for cls in classes}
    max_count = max(counts.values())
    
    X_list = [data_X]
    y_list = [y]
    
    for cls in classes:
        indices = np.where(y_labels == cls)[0]
        num_needed = max_count - len(indices)
        for _ in range(num_needed):
            idx = np.random.choice(indices)
            sample = data_X[idx]
            aug_sample = augment_ppg(sample, noise_std=noise_std_ppg)
            if aug_sample.shape != sample.shape:
                print(f"증강된 데이터 차원 불일치! 원본: {sample.shape}, 증강: {aug_sample.shape}")
            X_list.append(aug_sample[np.newaxis, ...])
            y_list.append(y[idx][np.newaxis, ...])
    
    X_augmented = np.concatenate(X_list, axis=0)
    y_augmented = np.concatenate(y_list, axis=0)
    idxs = np.arange(len(y_augmented))
    np.random.shuffle(idxs)
    return X_augmented[idxs], X_augmented[idxs], y_augmented[idxs]

# =============================================================================
# 학습 곡선(accuracy, loss) 플롯 저장 함수
# =============================================================================
def plot_training_curves(history, save_path):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# =============================================================================
# 디버깅용: 한 파일에서 생성되는 슬라이딩 윈도우의 개수를 계산하는 함수
# =============================================================================
def count_windows_for_file(file_path, window_size, step):
    data = np.load(file_path).astype(np.float32)
    data = np.transpose(data, (0, 2, 1))
    continuous_data = np.concatenate(data, axis=0)
    total_samples = continuous_data.shape[0]
    win_length = window_size * 128
    step_length = step * 128
    num_windows = (total_samples - win_length) // step_length + 1
    return num_windows

# =============================================================================
# LOSO 평가: 각 테스트 subject에 대해, 각 trial(=sample)을 window 단위로 평가
# - 훈련 데이터: 테스트 subject가 아닌 모든 trial의 window를 합친 후,
#   train_test_split을 통해 80:20 비율(학습/검증)로 분할하고, 각 데이터셋에 증강 적용
# - 테스트: 테스트 subject의 모든 window에 대해 전체 평가 후, trial 단위 평가 진행
# =============================================================================
def train_subject_leave_one_out(subject_data, epochs=300, batch_size=128, window_size=10, step=1):
    parent_dir = "/home/bcml1/sigenv/_4주차_ppg/LOSO_result_samsplit_3layer"
    os.makedirs(parent_dir, exist_ok=True)
    
    subject_ids = sorted(subject_data.keys())
    for test_subj in subject_ids:
        print(f"\nLOSO - Test Subject: {test_subj}")
        
        # 테스트 대상이 아닌 모든 파일들 수집
        train_files = []
        for subj in subject_ids:
            if subj != test_subj:
                train_files.extend(subject_data[subj])
        
        # 파일 튜플 균형 맞추기 (오버샘플링)
        train_files = balance_file_tuples(train_files)
        
        # Train/Validation Split (8:2, stratify 적용)
        if len(set([t[1] for t in train_files])) > 1:
            train_files, val_files = train_test_split(train_files, test_size=0.2,
                                                       stratify=[t[1] for t in train_files],
                                                       random_state=42)
        else:
            train_files, val_files = train_test_split(train_files, test_size=0.2,
                                                       stratify=None,
                                                       random_state=42)
        
        # 디버깅: 파일 단위 및 총 윈도우 수 출력
        # 디버깅: 파일 단위 및 총 윈도우 수 출력
        total_train_windows = sum(count_windows_for_file(fp, window_size, step) for fp, _ in train_files)
        total_val_windows = sum(count_windows_for_file(fp, window_size, step) for fp, _ in val_files)
        total_test_windows = sum(count_windows_for_file(fp, window_size, step) for fp, _ in subject_data[test_subj])
        print(f"Train files: {len(train_files)}, Total Train Windows: {total_train_windows}")
        print(f"Validation files: {len(val_files)}, Total Val Windows: {total_val_windows}")
        print(f"Test files (subject {test_subj}): {len(subject_data[test_subj])}, Total Test Windows: {total_test_windows}")

        # steps_per_epoch를 전체 윈도우 수로 계산
        steps_per_epoch = total_train_windows // batch_size
        validation_steps = total_val_windows // batch_size

        train_dataset = create_dataset_from_files(train_files, window_size, step, batch_size)
        val_dataset = create_dataset_from_files(val_files, window_size, step, batch_size)
        test_dataset = create_dataset_from_files(subject_data[test_subj], window_size, step, batch_size)

        # 모델 생성 및 컴파일
        # model = build_emcnn()
        model = build_emcnn_3ch()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    metrics=['accuracy'])

        history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset,
                            steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, verbose=1)

        result_dir = os.path.join(parent_dir, f"test_{test_subj}")
        os.makedirs(result_dir, exist_ok=True)
        
        model.save(os.path.join(result_dir, f"{test_subj}_model.keras"))
        np.save(os.path.join(result_dir, f"{test_subj}_history.npy"), history.history)
        plot_training_curves(history, os.path.join(result_dir, f"{test_subj}_training_curves.png"))
        
        # 1) 전체 테스트 subject의 윈도우 평가
        y_true_all = []
        y_pred_all = []
        for windows, labels in test_dataset:
            preds = model.predict(windows)
            y_true_all.extend(labels.numpy())
            y_pred_all.extend(np.argmax(preds, axis=1))
        
        overall_report = classification_report(y_true_all, y_pred_all, digits=4)
        overall_cm = confusion_matrix(y_true_all, y_pred_all)
        with open(os.path.join(result_dir, "classification_report_overall.txt"), "w") as f:
            f.write(overall_report)
        with open(os.path.join(result_dir, "confusion_matrix_overall.txt"), "w") as f:
            f.write(np.array2string(overall_cm))
        
        print(f"Test subject {test_subj} 평가 완료.")
        
if __name__ == "__main__":
    # data_dir = '/home/bcml1/2025_EMOTION/DEAP_PPG_10s_1soverlap_4label/'
    data_dir = '/home/bcml1/2025_EMOTION/DEAP_PPG_1s'
    # window_size = int(input("Enter window size (sec): "))  # 사용자 입력
    # step = int(input("Enter step size (sec): "))             # 사용자 입력
    # epochs = 50
    # batch_size = 128
    # subject_data = load_ppg_file_paths(data_dir, window_size=10, step=1)
    subject_data = load_ppg_file_paths(data_dir)
    # train_subject_leave_one_out(subject_data, epochs=300, batch_size=32)
    # 파일 경로와 라벨 정보를 subject 단위로 불러오기
    # subject_data = load_ppg_file_paths(data_dir)
    train_subject_leave_one_out(subject_data, epochs=300, batch_size=128, window_size=10, step=1)
    