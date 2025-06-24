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
        # TensorFlow Keras에서는 1D 전용 depthwise convolution 레이어가 없기 때문에 Conv1D에 groups 인자를 사용해서 각 채널별로 독립적인 연산을 수행하는 방식으로 구현
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

# =============================================================================
# 슬라이딩 윈도우 함수 (메모리 최적화)
# =============================================================================
# def sliding_window_ppg(continuous_data, window_size, step):
#     win_length = window_size * 128
#     step_length = step * 128
#     return np.array([continuous_data[i:i + win_length] 
#                      for i in range(0, len(continuous_data) - win_length + 1, step_length)],
#                     dtype=np.float32)
    
def load_data(data_dir, window_sec=10, sampling_rate=128, random_crop=True):
    """
    지정된 폴더 내의 npy 파일들을 불러오고, 파일명에서 감정 라벨(0: excited, 1: relaxed, 2: stressed, 3: bored)을 추출합니다.
    각 npy 파일은 (60, 5, 128) shape로 저장되어 있으며, 60개의 1초 분할 데이터를 의미합니다.
    
    이 함수에서는 1초 분할 데이터를 먼저 (60, 128, 5)로 transpose한 후, 두 축을 flatten하여
    전체 데이터를 (7680, 5) (즉, 60*128=7680 타임스텝)로 만듭니다.
    
    이후 window_sec에 해당하는 길이 (window_length = window_sec * sampling_rate, 기본값 1280)를
    랜덤(또는 시작부터)하게 crop하여 최종 입력 데이터를 만듭니다.
    
    Parameters:
      data_dir (str): npy 파일들이 있는 폴더 경로
      window_sec (int): 사용할 구간의 초 단위 길이 (기본 10초 → 10*128=1280 타임스텝)
      sampling_rate (int): 샘플링 주파수 (기본 128)
      random_crop (bool): True면 랜덤하게 crop, False면 항상 처음부터 crop
      
    Returns:
      X (np.ndarray): shape (num_samples, window_length, 5)
      y (np.ndarray): 각 sample의 정수형 라벨 (0: excited, 1: relaxed, 2: stressed, 3: bored)
    """
    file_list = glob.glob(os.path.join(data_dir, '*.npy'))
    data_list = []
    label_list = []
    window_length = window_sec * sampling_rate  # 예: 10초면 10*128 = 1280 타임스텝

    for file_path in file_list:
        # 파일명 예: sXX_trial_XX_label_X.npy
        filename = os.path.basename(file_path)
        # filename을 '_' 기준으로 분리한 후, 마지막 부분에서 라벨 숫자 추출
        # 예: ['s01', 'trial', '03', 'label', '2.npy'] → 라벨은 2
        parts = filename.split('_')
        label_str = parts[-1].split('.')[0]
        label = int(label_str)

        # npy 파일 로드: 원래 shape (60, 5, 128)
        data = np.load(file_path)
        # 현재 데이터 shape: (num_segments, channels, segment_length) = (60, 5, 128)
        # 모델에서는 (time, channels) 형식을 기대하므로, 먼저 (60, 128, 5)로 transpose
        data = np.transpose(data, (0, 2, 1))  # shape: (60, 128, 5)
        # 이후, 60개의 1초 데이터를 시간축으로 이어붙임 → (60*128, 5) = (7680, 5)
        data = data.reshape(-1, data.shape[-1])
        
        # 원하는 window_length (예, 1280 타임스텝)만큼 crop
        if data.shape[0] >= window_length:
            if random_crop:
                start = np.random.randint(0, data.shape[0] - window_length + 1)
            else:
                start = 0
            data = data[start:start+window_length, :]
        else:
            # 데이터가 window_length보다 짧으면 뒤를 0으로 pad
            pad_length = window_length - data.shape[0]
            data = np.pad(data, ((0, pad_length), (0, 0)), mode='constant')
        
        data_list.append(data)
        label_list.append(label)
    
    X = np.array(data_list)
    y = np.array(label_list)
    return X, y

def segment_data_with_overlap(X, y, window_size=10, overlap=1, sampling_rate=128):
    """
    X: np.ndarray, shape = (num_samples, total_timesteps, channels)
       예: (num_samples, 7680, 5) → 한 샘플당 60초 분량 (60*128=7680 타임스텝)
    y: np.ndarray, shape = (num_samples,)
       각 샘플의 라벨 (0: excited, 1: relaxed, 2: stressed, 3: bored)
       
    window_size: 각 window의 길이(초 단위). window_size가 1이면 원래 1초 분할 데이터를 그대로 사용.
    overlap: window들 간의 겹침(초 단위). window_size가 1일 경우는 무시됩니다.
    sampling_rate: 초당 타임스텝 수 (예, 128)
    
    반환:
      X_segments: np.ndarray, shape = (total_windows, window_length, channels)
      y_segments: np.ndarray, shape = (total_windows,)
    """
    X_segments = []
    y_segments = []
    
    # window_size가 1이면 원래의 1초 단위 분할 데이터를 그대로 사용
    if window_size == 1:
        window_length = sampling_rate  # 1초 분량
        step = sampling_rate           # non-overlapping segmentation
        for sample, label in zip(X, y):
            total_length = sample.shape[0]
            # total_length가 sampling_rate의 배수가 아닐 경우, 남은 부분은 무시하거나 pad할 수 있음
            for start in range(0, total_length - window_length + 1, step):
                segment = sample[start:start+window_length, :]
                X_segments.append(segment)
                y_segments.append(label)
    else:
        # window_size가 1보다 클 경우, overlap 적용 (window_size는 overlap보다 커야 함)
        window_length = window_size * sampling_rate
        overlap_length = overlap * sampling_rate
        if window_length <= overlap_length:
            raise ValueError("window_size는 overlap보다 커야 합니다.")
        step = window_length - overlap_length
        for sample, label in zip(X, y):
            total_length = sample.shape[0]
            for start in range(0, total_length - window_length + 1, step):
                segment = sample[start:start+window_length, :]
                X_segments.append(segment)
                y_segments.append(label)
    
    return np.array(X_segments), np.array(y_segments)


def augment_sample(sample, noise_std=0.01):
    """
    하나의 sample에 대해 가우시안 노이즈를 추가하여 증강합니다.
    
    Parameters:
      sample: np.ndarray, 증강할 데이터 (예: (window_length, channels))
      noise_std: float, 노이즈의 표준편차 (기본값 0.01)
      
    Returns:
      np.ndarray, 증강된 sample
    """
    noise = np.random.normal(loc=0.0, scale=noise_std, size=sample.shape)
    return sample + noise

def balance_data(X, y, noise_std=0.01):
    """
    주어진 X, y 데이터를 받아 각 라벨의 샘플 수가 같아지도록 증강합니다.
    
    부족한 라벨의 경우, 해당 라벨의 기존 샘플에 작은 노이즈를 추가하여 증강한 후,
    원본 데이터와 합쳐서 반환합니다.
    
    Parameters:
      X: np.ndarray, shape = (num_samples, window_length, channels)
         슬라이딩 윈도우 처리된 데이터
      y: np.ndarray, shape = (num_samples,)
         각 sample의 라벨 (0: excited, 1: relaxed, 2: stressed, 3: bored)
      noise_std: float, 증강 시 사용할 가우시안 노이즈의 표준편차 (기본값 0.01)
      
    Returns:
      X_balanced: np.ndarray, 증강 후의 데이터
      y_balanced: np.ndarray, 증강 후의 라벨
    """
    unique_labels, counts = np.unique(y, return_counts=True)
    max_count = np.max(counts)
    
    X_augmented = []
    y_augmented = []
    
    # 각 라벨별로 부족한 샘플 수만큼 증강
    for label in unique_labels:
        indices = np.where(y == label)[0]
        current_count = len(indices)
        num_to_augment = max_count - current_count
        
        for _ in range(num_to_augment):
            # 해당 라벨의 기존 sample 중 랜덤하게 선택 후 증강
            idx = np.random.choice(indices)
            sample = X[idx]
            aug_sample = augment_sample(sample, noise_std)
            X_augmented.append(aug_sample)
            y_augmented.append(label)
    
    # 원본 데이터와 증강 데이터를 합칩니다.
    X_balanced = np.concatenate([X, np.array(X_augmented)], axis=0)
    y_balanced = np.concatenate([y, np.array(y_augmented)], axis=0)
    
    # 데이터를 섞어줍니다.
    permutation = np.random.permutation(len(y_balanced))
    X_balanced = X_balanced[permutation]
    y_balanced = y_balanced[permutation]
    
    return X_balanced, y_balanced

# # =============================================================================
# # PPG 데이터 증강 함수들 (augment 및 balance_ppg_dataset)
# # =============================================================================
# def augment_ppg(ppg, noise_std=0.01, shift_max=5, scale_range=(0.9, 1.1), mask_freq=10, mask_factor=0.2, peak_shift=3):
#     augmented_ppg = np.zeros_like(ppg)
#     for ch in range(ppg.shape[1]):
#         ppg_ch = ppg[:, ch]
#         ppg_noisy = ppg_ch + np.random.normal(0, noise_std, ppg_ch.shape)
#         shift = np.random.randint(-shift_max, shift_max)
#         ppg_shifted = np.roll(ppg_noisy, shift, axis=0)
#         scale_factor = np.random.uniform(*scale_range)
#         ppg_scaled = ppg_shifted * scale_factor
#         fft_signal = np.fft.fft(ppg_scaled)
#         fft_signal[:mask_freq] *= (1 - mask_factor)
#         ppg_freq = np.real(np.fft.ifft(fft_signal))
#         peaks, _ = signal.find_peaks(ppg_freq, distance=30)
#         if len(peaks) > 0:
#             shift_values = np.random.randint(-peak_shift, peak_shift, size=len(peaks))
#             for i, s in zip(peaks, shift_values):
#                 if 0 <= i + s < len(ppg_freq):
#                     ppg_freq[i] = ppg_freq[i + s]
#         augmented_ppg[:, ch] = ppg_freq
#     return augmented_ppg

# def balance_ppg_dataset(data_X, _, y, noise_std_ppg=0.01):
#     y_labels = np.argmax(y, axis=1)
#     classes = np.unique(y_labels)
#     counts = {cls: np.sum(y_labels == cls) for cls in classes}
#     max_count = max(counts.values())
    
#     X_list = [data_X]
#     y_list = [y]
    
#     for cls in classes:
#         indices = np.where(y_labels == cls)[0]
#         num_needed = max_count - len(indices)
#         for _ in range(num_needed):
#             idx = np.random.choice(indices)
#             sample = data_X[idx]
#             aug_sample = augment_ppg(sample, noise_std=noise_std_ppg)
#             if aug_sample.shape != sample.shape:
#                 print(f"증강된 데이터 차원 불일치! 원본: {sample.shape}, 증강: {aug_sample.shape}")
#             X_list.append(aug_sample[np.newaxis, ...])
#             y_list.append(y[idx][np.newaxis, ...])
    
#     X_augmented = np.concatenate(X_list, axis=0)
#     y_augmented = np.concatenate(y_list, axis=0)
#     idxs = np.arange(len(y_augmented))
#     np.random.shuffle(idxs)
#     return X_augmented[idxs], X_augmented[idxs], y_augmented[idxs]

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
# def count_windows_for_file(file_path, window_size, step):
#     data = np.load(file_path).astype(np.float32)
#     data = np.transpose(data, (0, 2, 1))
#     continuous_data = np.concatenate(data, axis=0)
#     total_samples = continuous_data.shape[0]
#     win_length = window_size * 128
#     step_length = step * 128
#     num_windows = (total_samples - win_length) // step_length + 1
#     return num_windows

# =============================================================================
# LOSO 평가 함수: 각 subject를 테스트로 남기고 나머지로 학습/검증
# =============================================================================
def loso_train_and_test(data_dir, window_sec=10, sampling_rate=128, random_crop=True,
                    epochs=50, batch_size=32):
    """
    Leave-One-Subject-Out (LOSO) 평가 함수.
    
    Parameters:
      data_dir (str): npy 파일들이 위치한 폴더 경로. 파일명 형식 예: sXX_trial_XX_label_X.npy
      window_sec (int): 사용할 구간의 길이(초). 기본값은 10 (10*128=1280 타임스텝)
      sampling_rate (int): 샘플링 주파수. 기본값 128.
      random_crop (bool): True이면 랜덤하게 crop, False이면 시작부터 crop.
      epochs (int): 각 subject에 대해 학습할 에폭 수.
      batch_size (int): 배치 사이즈.
      
    Returns:
      all_results (dict): subject별 평가 결과 (테스트 loss, accuracy, confusion matrix, classification report, history)
    """
    parent_dir = "./LOSO_results"
    os.makedirs(parent_dir, exist_ok=True)
    subject_ids = sorted(subject_data.keys())
    
    for test_subj in subject_ids:
        print(f"\nLOSO Evaluation - Test Subject: {test_subj}")
        # 테스트 subject의 파일들
        test_files = subject_data[test_subj]
        # 나머지 subject의 파일들을 모아서 training용으로 사용
        train_files = []
        for subj in subject_ids:
            if subj != test_subj:
                train_files.extend(subject_data[subj])
        # 오버샘플링으로 train 파일 균형 맞추기
        train_files = balance_file_tuples(train_files)
        # train/validation split (stratify 적용)
        train_files, val_files = train_test_split(train_files, test_size=0.2, 
                                                   stratify=[t[1] for t in train_files],
                                                   random_state=42)
        print(f"Train files: {len(train_files)} / Validation files: {len(val_files)} / Test files: {len(test_files)}")
        
        # tf.data.Dataset 생성
        train_ds = create_dataset(train_files, window_size, step, batch_size)
        val_ds   = create_dataset(val_files, window_size, step, batch_size)
        test_ds  = create_dataset(test_files, window_size, step, batch_size)
        
        # 모델 생성 및 컴파일
        model = build_emcnn_3ch()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
        
        # 모델 학습
        history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, verbose=1)
        
        # 결과 저장 폴더
        result_dir = os.path.join(parent_dir, f"test_{test_subj}")
        os.makedirs(result_dir, exist_ok=True)
        model.save(os.path.join(result_dir, f"{test_subj}_model.keras"))
        np.save(os.path.join(result_dir, f"{test_subj}_history.npy"), history.history)
        plot_training_curves(history, os.path.join(result_dir, f"{test_subj}_training_curves.png"))
        
        # 테스트 데이터 평가 (모든 sliding window 단위로 예측 후 trial 단위 평가)
        y_true_all = []
        y_pred_all = []
        for X_batch, y_batch in test_ds:
            preds = model.predict(X_batch)
            y_true_all.extend(y_batch.numpy())
            y_pred_all.extend(np.argmax(preds, axis=1))
        
        report = classification_report(y_true_all, y_pred_all, digits=4)
        cm = confusion_matrix(y_true_all, y_pred_all)
        with open(os.path.join(result_dir, "classification_report.txt"), "w") as f:
            f.write(report)
        with open(os.path.join(result_dir, "confusion_matrix.txt"), "w") as f:
            f.write(np.array2string(cm))
        
        print(f"Test subject {test_subj} evaluation complete.")
        
# =============================================================================
# LOSO 평가: 각 테스트 subject에 대해, 각 trial(=sample)을 window 단위로 평가
# - 훈련 데이터: 테스트 subject가 아닌 모든 trial의 window를 합친 후,
#   train_test_split을 통해 80:20 비율(학습/검증)로 분할하고, 각 데이터셋에 증강 적용
# - 테스트: 테스트 subject의 모든 window에 대해 전체 평가 후, trial 단위 평가 진행
# =============================================================================
# def train_subject_leave_one_out(subject_data, epochs=300, batch_size=128, window_size=10, step=1):
#     parent_dir = "/home/bcml1/sigenv/_4주차_ppg/LOSO_result_samsplit_3layer"
#     os.makedirs(parent_dir, exist_ok=True)
    
#     subject_ids = sorted(subject_data.keys())
#     for test_subj in subject_ids:
#         print(f"\nLOSO - Test Subject: {test_subj}")
        
#         # 테스트 대상이 아닌 모든 파일들 수집
#         train_files = []
#         for subj in subject_ids:
#             if subj != test_subj:
#                 train_files.extend(subject_data[subj])
        
#         # 파일 튜플 균형 맞추기 (오버샘플링)
#         train_files = balance_file_tuples(train_files)
        
#         # Train/Validation Split (8:2, stratify 적용)
#         if len(set([t[1] for t in train_files])) > 1:
#             train_files, val_files = train_test_split(train_files, test_size=0.2,
#                                                        stratify=[t[1] for t in train_files],
#                                                        random_state=42)
#         else:
#             train_files, val_files = train_test_split(train_files, test_size=0.2,
#                                                        stratify=None,
#                                                        random_state=42)
        
#         # 디버깅: 파일 단위 및 총 윈도우 수 출력
#         # 디버깅: 파일 단위 및 총 윈도우 수 출력
#         total_train_windows = sum(count_windows_for_file(fp, window_size, step) for fp, _ in train_files)
#         total_val_windows = sum(count_windows_for_file(fp, window_size, step) for fp, _ in val_files)
#         total_test_windows = sum(count_windows_for_file(fp, window_size, step) for fp, _ in subject_data[test_subj])
#         print(f"Train files: {len(train_files)}, Total Train Windows: {total_train_windows}")
#         print(f"Validation files: {len(val_files)}, Total Val Windows: {total_val_windows}")
#         print(f"Test files (subject {test_subj}): {len(subject_data[test_subj])}, Total Test Windows: {total_test_windows}")

#         # steps_per_epoch를 전체 윈도우 수로 계산
#         steps_per_epoch = total_train_windows // batch_size
#         validation_steps = total_val_windows // batch_size

#         train_dataset = create_dataset_from_files(train_files, window_size, step, batch_size)
#         val_dataset = create_dataset_from_files(val_files, window_size, step, batch_size)
#         test_dataset = create_dataset_from_files(subject_data[test_subj], window_size, step, batch_size)

#         # 모델 생성 및 컴파일
#         # model = build_emcnn()
#         model = build_emcnn_3ch()
#         model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#                     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#                     metrics=['accuracy'])

#         history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset,
#                             steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, verbose=1)

#         result_dir = os.path.join(parent_dir, f"test_{test_subj}")
#         os.makedirs(result_dir, exist_ok=True)
        
#         model.save(os.path.join(result_dir, f"{test_subj}_model.keras"))
#         np.save(os.path.join(result_dir, f"{test_subj}_history.npy"), history.history)
#         plot_training_curves(history, os.path.join(result_dir, f"{test_subj}_training_curves.png"))
        
#         # 1) 전체 테스트 subject의 윈도우 평가
#         y_true_all = []
#         y_pred_all = []
#         for windows, labels in test_dataset:
#             preds = model.predict(windows)
#             y_true_all.extend(labels.numpy())
#             y_pred_all.extend(np.argmax(preds, axis=1))
        
#         overall_report = classification_report(y_true_all, y_pred_all, digits=4)
#         overall_cm = confusion_matrix(y_true_all, y_pred_all)
#         with open(os.path.join(result_dir, "classification_report_overall.txt"), "w") as f:
#             f.write(overall_report)
#         with open(os.path.join(result_dir, "confusion_matrix_overall.txt"), "w") as f:
#             f.write(np.array2string(overall_cm))
        
#         print(f"Test subject {test_subj} 평가 완료.")
        
if __name__ == "__main__":
    # data_dir = '/home/bcml1/2025_EMOTION/DEAP_PPG_10s_1soverlap_4label/'
    data_dir = '/home/bcml1/2025_EMOTION/DEAP_PPG_1s'
    subject_data = load_ppg_file_paths(data_dir)
    train_subject_leave_one_out(subject_data, epochs=300, batch_size=128, window_size=10, step=1)
    