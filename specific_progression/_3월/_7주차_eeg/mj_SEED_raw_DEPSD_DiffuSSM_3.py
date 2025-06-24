# DDPM 기반 denoiser: 기존의 평균 필터 대신 간단한 UNet 구조를 이용한 DenoiseUNetLayer를 도입하여 노이즈 제거를 수행합니다.

# 채널 Self‑Attention (SEBlock): 2D 매핑된 이미지에서 채널 간 중요도를 학습할 수 있도록 SEBlock을 추가해, 중요한 채널의 정보를 더 강조합니다.

# Transformer Layer 수 감소: 최종 모델에서 Transformer layer의 개수를 n_transformer_layers=1로 기본값을 줄였습니다.

#이 코드는 이전 버전의 코드에다가 10초 데이터를 1초 overlap 해서 데이터 증강을 구현하는 것까지 작성했음!
# 최신 diffusion기반 denoising하는 것을 DDPM 방식으로 구현하도록 해주고, 평균 필터 대신, 파라미터를 학습하는 네트워크(예: 작은 UNet)를 도입해보면, 노이즈 제거 과정에서 더 중요한 특징을 보존하도록 코드를 수정
# 2D 매핑 및 채널 가중치 적용할 때, 채널별 attention 메커니즘에 self-attention을 추가해서 중요한 채널에 더 집중하도록 개선
# Transformer layer의 개수를 줄임

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pywt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, BatchNormalization, 
                                     GlobalAveragePooling2D, Dense, Dropout, concatenate, 
                                     Lambda, Flatten, TimeDistributed)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import LayerNormalization, Concatenate

# -------------------------------------------------------------------------
# GPU 메모리 제한
# -------------------------------------------------------------------------
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

limit_gpu_memory(10000)

# subject normalize
# 먼저 subject_normalize 함수를 주석 해제하여 정의합니다.
def subject_normalize(X, S):
    X_norm = np.copy(X)
    unique_subjects = np.unique(S)
    for subj in unique_subjects:
        subj_idx = np.where(S == subj)[0]
        de_data = X[subj_idx, :, 0, ...]
        mean_de = np.mean(de_data)
        X_norm[subj_idx, :, 0, ...] = X_norm[subj_idx, :, 0, ...] - mean_de
        psd_data = X[subj_idx, :, 1, ...]
        mean_psd = np.mean(psd_data)
        X_norm[subj_idx, :, 1, ...] = X_norm[subj_idx, :, 1, ...] - mean_psd
    return X_norm

# TensorFlow용 래퍼 함수 (subject_info는 numpy array로 전달)
def tf_subject_normalize(x, subject_info):
    # x는 tf.Tensor, subject_info는 numpy array
    x_norm = tf.py_function(func=lambda x: subject_normalize(x, subject_info),
                            inp=[x],
                            Tout=tf.float32)
    # shape 정보를 복원합니다.
    x_norm.set_shape(x.get_shape())
    return x_norm

# # 선택적 정규화를 위한 NormalizeLayer
# class NormalizeLayer(tf.keras.layers.Layer):
#     def __init__(self, method="minmax", **kwargs):
#         super(NormalizeLayer, self).__init__(**kwargs)
#         self.method = method.lower()

#     def call(self, inputs):
#         x = tf.where(tf.math.is_nan(inputs), tf.zeros_like(inputs), inputs)
#         x = tf.where(tf.math.is_inf(x), tf.zeros_like(x), x)
#         axes = list(range(1, inputs.shape.rank))
#         if self.method == "zscore":
#             mean = tf.reduce_mean(x, axis=axes, keepdims=True)
#             std = tf.math.reduce_std(x, axis=axes, keepdims=True) + 1e-8
#             return (x - mean) / std
#         elif self.method == "minmax":
#             min_val = tf.reduce_min(x, axis=axes, keepdims=True)
#             max_val = tf.reduce_max(x, axis=axes, keepdims=True)
#             return (x - min_val) / (max_val - min_val + 1e-8)
#         else:
#             raise ValueError("Unknown normalization method. Choose 'zscore' or 'minmax'.")

#     def compute_output_shape(self, input_shape):
#         return input_shape

# -------------------------------------------------------------------------
# DDPM 기반 Denoiser (UNet 형태의 간단한 denoising 네트워크)
# -------------------------------------------------------------------------
def DenoiseUNetLayer(x):
    # x: (batch, height, width, channels) → 여기서는 (None, 9, 9, 1)로 가정
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(x)       # (None, 9, 9, 16)
    pool1 = MaxPooling2D((2,2), padding='same')(conv1)                      # (None, 5, 5, 16) -> 9일 경우 ceil(9/2)=5
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)     # (None, 5, 5, 32)
    up1 = tf.keras.layers.UpSampling2D((2,2))(conv2)                        # (None, 10, 10, 32)
    # up1를 (9,9,32)로 crop (즉, 첫 9행과 9열 선택)
    cropped_up1 = Lambda(lambda z: z[:, :9, :9, :])(up1)                   # (None, 9, 9, 32)
    concat1 = concatenate([conv1, cropped_up1])                             # (None, 9, 9, 16+32)
    conv3 = Conv2D(16, (3, 3), activation='relu', padding='same')(concat1)   # (None, 9, 9, 16)
    out = Conv2D(x.shape[-1], (1, 1), activation='linear', padding='same')(conv3)  # (None, 9, 9, 1)
    return out

# 래핑하여 Layer로 구현 (TimeDistributed 적용 시 사용)
class DenoiseUNet(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DenoiseUNet, self).__init__(**kwargs)
        # 레이어를 __init__에서 한 번만 생성합니다.
        self.conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')
        self.pool1 = MaxPooling2D((2, 2), padding='same')
        self.conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')
        self.up1 = tf.keras.layers.UpSampling2D((2, 2))
        self.crop = Lambda(lambda z: z[:, :9, :9, :])  # crop layer
        self.concat = concatenate  # concatenate 함수 사용
        self.conv3 = Conv2D(16, (3, 3), activation='relu', padding='same')
        self.out_conv = Conv2D(1, (1, 1), activation='linear', padding='same')
    
    def call(self, inputs):
        # inputs: (batch, 9, 9, 1)
        x1 = self.conv1(inputs)          # (None, 9, 9, 16)
        x = self.pool1(x1)               # (None, 5, 5, 16) (padding='same' -> ceil(9/2)=5)
        x = self.conv2(x)                # (None, 5, 5, 32)
        x = self.up1(x)                  # (None, 10, 10, 32)
        x = self.crop(x)                 # (None, 9, 9, 32)
        x = concatenate([x1, x])         # (None, 9, 9, 48)
        x = self.conv3(x)                # (None, 9, 9, 16)
        out = self.out_conv(x)           # (None, 9, 9, 1)
        return out

# -------------------------------------------------------------------------
# SEBlock (채널 self-attention: Squeeze & Excitation 블록)
# -------------------------------------------------------------------------
class SEBlock(tf.keras.layers.Layer):
    def __init__(self, reduction=2, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.reduction = reduction
    def build(self, input_shape):
        channels = input_shape[1]  # assuming channels_first: (batch, channels, height, width, 1)
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(data_format="channels_first")
        self.fc1 = Dense(channels // self.reduction, activation='relu', kernel_regularizer=l2(1e-4))
        self.fc2 = Dense(channels, activation='sigmoid', kernel_regularizer=l2(1e-4))
    def call(self, inputs):
        x = tf.squeeze(inputs, axis=-1)
        s = self.global_avg_pool(x)
        s = self.fc1(s)
        s = self.fc2(s)
        s = tf.reshape(s, (-1, inputs.shape[1], 1, 1, 1))
        return inputs * s

# -------------------------------------------------------------------------
# DiffuSSMLayer (기존과 동일)
# -------------------------------------------------------------------------
class DiffuSSMLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, dropout_rate=0.5, bottleneck_dim=16, **kwargs):
        super(DiffuSSMLayer, self).__init__(**kwargs)
        self.hourglass_down = Dense(d_model // 2, activation=tf.nn.gelu, kernel_regularizer=l2(1e-5))
        self.hourglass_up = Dense(d_model, activation=None, kernel_regularizer=l2(1e-5))
        self.dropout = Dropout(dropout_rate)
        self.layernorm = LayerNormalization(epsilon=1e-6)
        self.bi_gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(d_model, return_sequences=True), merge_mode='sum'
        )
        self.dropout_gru = Dropout(dropout_rate)
        self.ffn = tf.keras.Sequential([
            Dense(d_ff, activation=tf.nn.gelu, kernel_regularizer=l2(1e-5)),
            Dense(d_model, kernel_regularizer=l2(1e-5))
        ])
        self.dropout_ffn = Dropout(dropout_rate)
        self.layernorm_ffn = LayerNormalization(epsilon=1e-6)
        self.adapter_down = Dense(bottleneck_dim, activation='elu', kernel_regularizer=l2(1e-5))
        self.adapter_up = Dense(d_model, activation=None, kernel_regularizer=l2(1e-5))
        self.alpha = self.add_weight(shape=(1,), initializer=tf.zeros_initializer(), trainable=True, name="ssm_alpha")
    
    def call(self, inputs, training=False):
        hour_out = self.hourglass_down(inputs)
        hour_out = self.hourglass_up(hour_out)
        hour_out = self.dropout(hour_out, training=training)
        ssm_out = self.bi_gru(inputs, training=training)
        ssm_out = self.dropout_gru(ssm_out, training=training)
        ssm_out = self.alpha * ssm_out
        x = inputs + hour_out + ssm_out
        x = self.layernorm(x)
        ffn_out = self.ffn(x)
        ffn_out = self.dropout_ffn(ffn_out, training=training)
        x = self.layernorm_ffn(x + ffn_out)
        shortcut = x
        x = self.adapter_down(x)
        x = self.adapter_up(x)
        return x + shortcut

# -------------------------------------------------------------------------
# Helper 함수들: 1초 단위 segment를 10초 그룹으로 묶기 위한 함수들
# -------------------------------------------------------------------------
def split_trial_to_1sec_slices(trial):
    slices = [trial[:, i:i+1, :] for i in range(trial.shape[1])]
    return slices

# 1. process_slice_9x9 함수 수정 (원래의 (4,9,9,1) 형태로 복원)
def process_slice_9x9(slice_, channel_positions):
    slice_2d = np.squeeze(slice_, axis=1)  # (62, 4)
    mapped = np.zeros((9, 9, slice_2d.shape[-1]), dtype=np.float32)
    for ch in range(slice_2d.shape[0]):
        if ch not in channel_positions:
            continue
        row, col = channel_positions[ch]
        mapped[row, col, :] = slice_2d[ch, :]
    # 원래대로 (channels, height, width) -> (4,9,9)로 transpose 후, expand_dims하여 (4,9,9,1)
    img = np.transpose(mapped, (2, 0, 1))  # (4,9,9)
    img = np.expand_dims(img, axis=-1)     # (4,9,9,1)
    return img

# ===== 데이터 증강: sliding window (stride 1)으로 10초 그룹 생성 =====
def process_trial_9x9_new(de_trial, psd_trial, channel_positions, group_size=10):
    slices_de = split_trial_to_1sec_slices(de_trial)
    slices_psd = split_trial_to_1sec_slices(psd_trial)
    num_groups = len(slices_de) - group_size + 1
    sample_list = []
    for g in range(num_groups):
        group_de = slices_de[g:g+group_size]
        group_psd = slices_psd[g:g+group_size]
        time_steps = []
        for t in range(group_size):
            de_img = process_slice_9x9(group_de[t], channel_positions)    # (9,9,4)
            psd_img = process_slice_9x9(group_psd[t], channel_positions)      # (9,9,4)
            # 스택 시: (2, 9, 9, 4)
            modality_stack = np.stack([de_img, psd_img], axis=0)
            # 마지막에 채널 차원 추가하지 않아도 됩니다.
            time_steps.append(modality_stack)
        # (group_size, 2, 9, 9, 4)
        sample = np.stack(time_steps, axis=0)
        sample_list.append(sample)
    return np.array(sample_list)

# -------------------------------------------------------------------------
# replicate_labels, load_seediv_data, channel_positions 등 (이전과 동일)
# -------------------------------------------------------------------------
def replicate_labels_for_trials(trials_list, labels):
    replicated = []
    total_segments = 0
    for trial, label in zip(trials_list, labels):
        num_segments = trial.shape[0]
        total_segments += num_segments
        rep = np.repeat(label[np.newaxis, :], num_segments, axis=0)
        replicated.append(rep)
    all_labels = np.concatenate(replicated, axis=0) if replicated else None
    assert all_labels.shape[0] == total_segments, (
        f"라벨 복제 오류: 총 세그먼트 수는 {total_segments}개인데, 복제된 라벨 수는 {all_labels.shape[0]}개입니다."
    )
    print("replicate_labels_for_trials: 총 세그먼트 수 =", total_segments, "복제된 라벨 shape =", all_labels.shape)
    return all_labels

def load_seediv_data(base_dirs):
    data_de = {}
    data_psd = {}
    for base_dir in base_dirs:
        file_list = glob.glob(os.path.join(base_dir, "*.npy"))
        modality = None
        if "eeg_DE" in base_dir:
            modality = "de"
        elif "eeg_PSD" in base_dir:
            modality = "psd"
        else:
            continue
        for file in file_list:
            filename = os.path.basename(file)
            parts = filename.replace('.npy', '').split('_')
            if len(parts) < 6:
                continue
            subject = parts[0]
            trial = parts[3]
            try:
                label_val = int(parts[5])
            except:
                continue
            arr = np.load(file)
            if arr.shape[-1] == 5:
                arr = arr[..., 1:]
            if modality == "de":
                data_de[(subject, trial)] = (arr, label_val)
            elif modality == "psd":
                data_psd[(subject, trial)] = (arr, label_val)
    common_ids = set(data_de.keys()).intersection(set(data_psd.keys()))
    de_list, psd_list, label_list, subject_list = [], [], [], []
    for sid in sorted(common_ids):
        subj, trial = sid
        arr_de, label_de = data_de[sid]
        arr_psd, label_psd = data_psd[sid]
        if label_de != label_psd:
            continue
        de_list.append(arr_de)
        psd_list.append(arr_psd)
        label_list.append(label_de)
        subject_list.append(subj)
    return de_list, psd_list, label_list, subject_list

channel_positions_9x9 = {
    0: (0,3),   1: (0,4),   2: (0,5),
    3: (1,3),   4: (1,5),
    5: (2,0),   6: (2,1),   7: (2,2),   8: (2,3),   9: (2,4),
    10: (2,5),  11: (2,6),  12: (2,7),  13: (2,8),
    14: (3,0),  15: (3,1),  16: (3,2),  17: (3,3),  18: (3,4),
    19: (3,5),  20: (3,6),  21: (3,7),  22: (3,8),
    23: (4,0),  24: (4,1),  25: (4,2),  26: (4,3),  27: (4,4),
    28: (4,5),  29: (4,6),  30: (4,7),  31: (4,8),
    32: (5,0),  33: (5,1),  34: (5,2),  35: (5,3),  36: (5,4),
    37: (5,5),  38: (5,6),  39: (5,7),  40: (5,8),
    41: (6,0),  42: (6,1),  43: (6,2),  44: (6,3),  45: (6,4),
    46: (6,5),  47: (6,6),  48: (6,7),  49: (6,8),
    50: (7,1),  51: (7,2),  52: (7,3),  53: (7,4),  54: (7,5),
    55: (7,6),  56: (7,7),
    57: (8,2),  58: (8,3),  59: (8,4),  60: (8,5),  61: (8,6)
}

# -------------------------------------------------------------------------
# DiffusionDenoiseLayer -> DenoiseUNetLayer로 대체 (학습 가능한 denoiser)
# -------------------------------------------------------------------------
class DiffusionDenoiseLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DiffusionDenoiseLayer, self).__init__(**kwargs)
        # 기존 평균 필터 대신 학습 가능한 UNet denoiser 사용
        self.denoiser = DenoiseUNet()

    def call(self, inputs):
        # inputs: (batch, H, W, C)
        return self.denoiser(inputs)
    
    def compute_output_shape(self, input_shape):
        return input_shape

# -------------------------------------------------------------------------
# SpatialSpectralConvModule, SpatialSpectralAttention (변경 없음)
# -------------------------------------------------------------------------
class SpatialSpectralConvModule(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides):
        super(SpatialSpectralConvModule, self).__init__()
        self.spatial_conv = tf.keras.layers.Conv3D(
            filters, kernel_size=(1, *kernel_size), strides=(1, *strides),
            padding="same", activation="relu", kernel_regularizer=l2(1e-4)
        )
        self.spectral_conv = tf.keras.layers.Conv3D(
            filters, kernel_size=(kernel_size[0], 1, 1), strides=(1, *strides),
            padding="same", activation="relu", kernel_regularizer=l2(1e-4)
        )
    def call(self, inputs):
        spatial_features = self.spatial_conv(inputs)
        spectral_features = self.spectral_conv(inputs)
        return spatial_features + spectral_features

class SpatialSpectralAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(SpatialSpectralAttention, self).__init__()
        self.spatial_squeeze = tf.keras.layers.Conv3D(1, kernel_size=(1,1,1), activation="sigmoid")
        self.spectral_squeeze = tf.keras.layers.Conv3D(1, kernel_size=(1,1,1), activation="sigmoid")
    def call(self, inputs):
        spatial_mask = self.spatial_squeeze(inputs)
        spectral_mask = self.spectral_squeeze(inputs)
        return inputs * spatial_mask + inputs * spectral_mask

# -------------------------------------------------------------------------
# TransformerEncoderLayer (여러 레이어 적용 가능하도록 수정됨)
# -------------------------------------------------------------------------
class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(d_ff, activation=tf.nn.gelu, kernel_regularizer=l2(1e-4)),
            Dense(d_model, kernel_regularizer=l2(1e-4))
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
    
    def call(self, inputs, training=False):
        attn_output = self.mha(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# -------------------------------------------------------------------------
# SEBlock (채널 self-attention 추가)
# -------------------------------------------------------------------------
class SEBlock(tf.keras.layers.Layer):
    def __init__(self, reduction=2, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.reduction = reduction
    def build(self, input_shape):
        # input_shape: (batch, channels, height, width, 1)
        channels = input_shape[1]
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(data_format="channels_first")
        self.fc1 = Dense(channels // self.reduction, activation='relu', kernel_regularizer=l2(1e-4))
        self.fc2 = Dense(channels, activation='sigmoid', kernel_regularizer=l2(1e-4))
    def call(self, inputs):
        x = tf.squeeze(inputs, axis=-1)  # (batch, channels, height, width)
        s = self.global_avg_pool(x)      # (batch, channels)
        s = self.fc1(s)
        s = self.fc2(s)
        s = tf.reshape(s, (-1, inputs.shape[1], 1, 1, 1))
        return inputs * s

# -------------------------------------------------------------------------
# CNNBranchLayer 수정: SEBlock (채널 self-attention) 추가
# -------------------------------------------------------------------------
class CNNBranchLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_bands=4, **kwargs):
        super(CNNBranchLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_bands = num_bands
        self.attention = SpatialSpectralAttention()
        self.band_weights = self.add_weight(
            shape=(num_bands, 1, 1, 1),
            initializer='ones',
            trainable=True,
            name='band_weights'
        )
        # spatial_weights의 shape를 (1,9,9,1)로 수정합니다.
        self.spatial_weights = self.add_weight(
            shape=(1, 9, 9, 1),
            initializer='ones',
            trainable=True,
            name='spatial_weights'
        )
        self.se_block = SEBlock(reduction=2)
        self.conv1 = SpatialSpectralConvModule(8, kernel_size=(3,3), strides=(3,3))
        self.conv2 = SpatialSpectralConvModule(16, kernel_size=(1,1), strides=(1,1))
        self.conv3 = SpatialSpectralConvModule(32, kernel_size=(1,1), strides=(1,1))
        self.flatten = Flatten()
        self.dense = Dense(d_model, activation="relu", kernel_regularizer=l2(1e-4))
    
    def call(self, x):
        # x: (batch, num_bands, height, width, 1) → (batch, 4, 9, 9, 1) 예상
        x = x * self.band_weights
        x = x * self.spatial_weights  # 이제 (9,9)와 맞음
        x = self.attention(x)
        x = self.se_block(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

    def compute_output_shape(self, input_shape):
        # input_shape: (batch, num_bands, height, width, 1)
        return (input_shape[0], self.d_model)


# -------------------------------------------------------------------------
# 단일 세그먼트 모델: 입력 shape: (2, 4, 9, 9, 1)
# Gaussian 노이즈 추가 후, DenoiseUNet (학습 가능한 denoiser) 및 CNN branch 특징 추출
# -------------------------------------------------------------------------
def build_single_segment_model(d_model=64):
    input_seg = Input(shape=(2, 4, 9, 9, 1))
    # 각각의 모달리티 분리 (여기서 axis 1는 모달리티 인덱스: 0: DE, 1: PSD)
    de_image = Lambda(lambda x: x[:,0,...])(input_seg)  # shape: (batch, 4, 9, 9, 1)
    psd_image = Lambda(lambda x: x[:,1,...])(input_seg)
    # 노이즈 추가
    de_noisy = Lambda(lambda x: x + tf.random.normal(tf.shape(x), mean=0.0, stddev=0.1))(de_image)
    psd_noisy = Lambda(lambda x: x + tf.random.normal(tf.shape(x), mean=0.0, stddev=0.1))(psd_image)
    # DDPM 기반 denoising: 학습 가능한 UNet denoiser (TimeDistributed 적용)
    de_denoised = TimeDistributed(DiffusionDenoiseLayer())(de_noisy)
    psd_denoised = TimeDistributed(DiffusionDenoiseLayer())(psd_noisy)
    # CNN branch 처리
    cnn_de = CNNBranchLayer(d_model, num_bands=4)
    cnn_psd = CNNBranchLayer(d_model, num_bands=4)
    feature_de = cnn_de(de_denoised)
    feature_psd = cnn_psd(psd_denoised)
    combined_features = Concatenate()([feature_de, feature_psd])
    combined_features = Dense(d_model, activation="relu", kernel_regularizer=l2(1e-4))(combined_features)
    model = Model(inputs=input_seg, outputs=combined_features)
    return model

# -------------------------------------------------------------------------
# 최종 모델: 입력 shape: (T, 2, 4, 9, 9, 1) with T=10 (10초 그룹)
# → 데이터 증강(슬라이딩 윈도우) 및 Transformer layer는 n_transformer_layers 개 반복 적용 (여기서는 n_transformer_layers=1로 줄임)
# -------------------------------------------------------------------------
def build_model(input_shape=(10, 2, 4, 9, 9, 1),
                subject_info=None,  # subject_info는 각 샘플의 주체 정보를 담은 numpy array
                n_diff_layers=2,
                n_transformer_layers=1,
                d_ff=512,
                p_drop=0.5,
                d_model=64,
                noise_std=0.1):
    """
    최종 모델 구성:
      1. 입력: 10초 그룹 데이터 (10, 2, 4, 9, 9, 1)
      2. 만약 subject_info가 제공되면, subject_normalize 함수를 이용해 정규화 수행
      3. 단일 세그먼트 모델(TimeDistributed)로 각 세그먼트별 특징 추출
      4. Gaussian 노이즈 추가 후 DiffuSSMLayer 반복 적용 (노이즈 제거 및 특징 보강)
      5. TransformerEncoderLayer를 통해 전역 시간 정보 학습 (기본 1개)
      6. Flatten 후 Dense layer로 4개 감정 클래스 분류
    """
    inputs = Input(shape=input_shape)
    # subject_info가 제공되면, tf_subject_normalize를 이용해 정규화
    if subject_info is not None:
        x = Lambda(lambda x: tf_subject_normalize(x, subject_info))(inputs)
    else:
        x = inputs
    
    # 단일 세그먼트 모델로 각 세그먼트별 특징 추출
    single_seg_model = build_single_segment_model(d_model)
    features_seq = TimeDistributed(single_seg_model)(x)
    
    # 추가 노이즈 (diffusion 기반 노이즈 제거 준비)
    noisy_features_seq = Lambda(
        lambda x: x + tf.random.normal(tf.shape(x), mean=0.0, stddev=noise_std),
        name="add_diffusion_noise"
    )(features_seq)
    x = noisy_features_seq
    
    # DiffuSSMLayer 반복 적용
    for _ in range(n_diff_layers):
        x = DiffuSSMLayer(d_model, d_ff, dropout_rate=p_drop)(x)
    
    # Transformer Layer 적용 (기본 1개)
    for _ in range(n_transformer_layers):
        x = TransformerEncoderLayer(d_model, n_heads=4, d_ff=d_ff, dropout_rate=p_drop)(x)
    
    x = Flatten()(x)
    outputs = Dense(4, activation="softmax", kernel_regularizer=l2(1e-4))(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# -------------------------------------------------------------------------
# GradualUnfreeze, compute_channel_importance, plot_channel_importance, inter_subject_cv_training_9x9
# (LOSO 학습/평가 파이프라인은 이전과 동일)
# -------------------------------------------------------------------------
class GradualUnfreeze(Callback):
    def __init__(self, unfreeze_schedule, unfreeze_lr=3e-4):
        super().__init__()
        self.unfreeze_schedule = unfreeze_schedule
        self.unfreeze_lr = unfreeze_lr
    def on_epoch_begin(self, epoch, logs=None):
        if epoch in self.unfreeze_schedule:
            print(f"\nUnfreezing layers at epoch {epoch}...")
            for layer_name in self.unfreeze_schedule[epoch]:
                for layer in self.model.layers:
                    if layer.name == layer_name:
                        layer.trainable = True
                        print(f"Layer {layer.name} unfreezed.")
            self.model.optimizer.learning_rate.assign(self.unfreeze_lr)
            print(f"Learning rate set to {self.unfreeze_lr} after unfreezing.")

def compute_channel_importance(model, X_test_final, y_test, baseline_acc, channel_positions_9x9, batch_size=16):
    num_channels = 62
    channel_importance = np.zeros(num_channels, dtype=np.float32)
    for ch in range(num_channels):
        X_test_mod = np.copy(X_test_final)
        if ch in channel_positions_9x9:
            row, col = channel_positions_9x9[ch]
            X_test_mod[:, :, :, :, row, col, :] = 0.0
        test_dataset_mod = tf.data.Dataset.from_tensor_slices((X_test_mod, to_categorical(y_test, num_classes=4))).batch(batch_size)
        _, masked_acc = model.evaluate(test_dataset_mod, verbose=0)
        channel_importance[ch] = baseline_acc - masked_acc
    importance_map = np.zeros((9, 9), dtype=np.float32)
    for ch in range(num_channels):
        if ch in channel_positions_9x9:
            r, c = channel_positions_9x9[ch]
            importance_map[r, c] = channel_importance[ch]
    return importance_map, channel_importance

def plot_channel_importance(importance_map, save_path="channel_importance.png"):
    plt.figure(figsize=(6,5))
    sns.heatmap(importance_map, annot=True, fmt=".3f", cmap="Reds")
    plt.title("Channel Importance (Permutation-based)")
    plt.xlabel("X-axis (columns)")
    plt.ylabel("Y-axis (rows)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Channel importance heatmap saved to {save_path}")

# LOSO 학습/평가 파이프라인 (기존과 동일)
def inter_subject_cv_training_9x9(base_dirs, result_dir, epochs=150, batch_size=16, target_subjects=None, pretrain_lr=1e-4, fine_tune_lr=1e-4, unfreeze_lr=1e-5):
    os.makedirs(result_dir, exist_ok=True)
    overall_folder = os.path.join(result_dir, "overall")
    os.makedirs(overall_folder, exist_ok=True)
    de_trials, psd_trials, label_list, subject_list = load_seediv_data(base_dirs)
    subject_data = {}
    for de, psd, label, subj in zip(de_trials, psd_trials, label_list, subject_list):
        if subj not in subject_data:
            subject_data[subj] = {"de": [], "psd": [], "labels": []}
        subject_data[subj]["de"].append(de)
        subject_data[subj]["psd"].append(psd)
        subject_data[subj]["labels"].append(label)
    overall_acc = {}
    overall_reports = {}
    subjects = sorted(subject_data.keys(), key=lambda x: int(x) if x.isdigit() else x)
    if target_subjects is not None:
        subjects = [subj for subj in subjects if subj in target_subjects]
    for test_subj in subjects:
        print(f"\n========== LOSO: Test subject: {test_subj} ==========")
        X_de_test_trials = subject_data[test_subj]["de"]
        X_psd_test_trials = subject_data[test_subj]["psd"]
        y_test_trials = np.array(subject_data[test_subj]["labels"])
        test_indices = np.arange(len(X_de_test_trials))
        ft_idx, final_idx = train_test_split(test_indices, train_size=0.2, random_state=42, stratify=y_test_trials)
        X_de_test_ft = [X_de_test_trials[i] for i in ft_idx]
        X_psd_test_ft = [X_psd_test_trials[i] for i in ft_idx]
        y_test_ft = to_categorical([y_test_trials[i] for i in ft_idx], num_classes=4)
        X_de_test_final = [X_de_test_trials[i] for i in final_idx]
        X_psd_test_final = [X_psd_test_trials[i] for i in final_idx]
        y_test_final = to_categorical([y_test_trials[i] for i in final_idx], num_classes=4)
        X_de_train_trials = []
        X_psd_train_trials = []
        y_train_list = []
        for subj in subjects:
            if subj == test_subj:
                continue
            X_de_train_trials.extend(subject_data[subj]["de"])
            X_psd_train_trials.extend(subject_data[subj]["psd"])
            y_train_list.extend(subject_data[subj]["labels"])
        y_train_list = np.array(y_train_list)
        y_cat_train = to_categorical(y_train_list, num_classes=4)
        num_train_trials = len(X_de_train_trials)
        indices = np.arange(num_train_trials)
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y_train_list)
        X_de_train_split = [X_de_train_trials[i] for i in train_idx]
        X_psd_train_split = [X_psd_train_trials[i] for i in train_idx]
        X_de_val_split = [X_de_train_trials[i] for i in val_idx]
        X_psd_val_split = [X_psd_train_trials[i] for i in val_idx]
        y_train_split = y_cat_train[train_idx]
        y_val_split = y_cat_train[val_idx]
        X_de_train_slices = [process_trial_9x9_new(de, psd, channel_positions_9x9, group_size=10) 
                              for de, psd in zip(X_de_train_split, X_psd_train_split)]
        y_train = replicate_labels_for_trials(X_de_train_slices, y_train_split)
        X_de_train = np.concatenate(X_de_train_slices, axis=0)
        X_de_val_slices = [process_trial_9x9_new(de, psd, channel_positions_9x9, group_size=10) 
                            for de, psd in zip(X_de_val_split, X_psd_val_split)]
        y_val = replicate_labels_for_trials(X_de_val_slices, y_val_split)
        X_de_val = np.concatenate(X_de_val_slices, axis=0)
        X_de_test_ft_slices = [process_trial_9x9_new(de, psd, channel_positions_9x9, group_size=10) 
                                for de, psd in zip(X_de_test_ft, X_psd_test_ft)]
        y_test_ft = replicate_labels_for_trials(X_de_test_ft_slices, y_test_ft)
        X_de_test_ft = np.concatenate(X_de_test_ft_slices, axis=0)
        X_de_test_final_slices = [process_trial_9x9_new(de, psd, channel_positions_9x9, group_size=10) 
                                   for de, psd in zip(X_de_test_final, X_psd_test_final)]
        y_test_final = replicate_labels_for_trials(X_de_test_final_slices, y_test_final)
        X_de_test_final = np.concatenate(X_de_test_final_slices, axis=0)
        print(f"X_de_test_ft shape: {X_de_test_ft.shape}, y_test_ft shape: {y_test_ft.shape}")
        print(f"X_de_test_final shape: {X_de_test_final.shape}, y_test_final shape: {y_test_final.shape}")
        print(f"X_de_train.shape: {X_de_train.shape}, X_de_val.shape: {X_de_val.shape}")
        train_dataset = tf.data.Dataset.from_tensor_slices((X_de_train, y_train)).shuffle(1000).batch(batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_de_val, y_val)).batch(batch_size)
        ft_dataset = tf.data.Dataset.from_tensor_slices((X_de_test_ft, y_test_ft)).shuffle(1000).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((X_de_test_final, y_test_final)).batch(batch_size)
        print(f"Test subject {test_subj}: Starting pre-training phase for {epochs} epochs...")
        model = build_model(input_shape=(10, 2, 4, 9, 9, 1), n_diff_layers=2, n_transformer_layers=1, d_ff=512, p_drop=0.3, d_model=64, noise_std=0.1)
        model.compile(optimizer=Adam(learning_rate=pretrain_lr), loss="categorical_crossentropy", metrics=["accuracy"])
        model.summary()
        pretrain_callbacks = [EarlyStopping(monitor='val_accuracy', patience=50, min_delta=0.001, restore_best_weights=True)]
        history_pretrain = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=pretrain_callbacks, verbose=1)
        subj_folder = os.path.join(result_dir, f"s{test_subj.zfill(2)}")
        os.makedirs(subj_folder, exist_ok=True)
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(history_pretrain.history['loss'], label='Pre-train Train Loss')
        plt.plot(history_pretrain.history['val_loss'], label='Pre-train Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Pre-training Loss Curve')
        plt.subplot(1,2,2)
        plt.plot(history_pretrain.history['accuracy'], label='Pre-train Train Acc')
        plt.plot(history_pretrain.history['val_accuracy'], label='Pre-train Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Pre-training Accuracy Curve')
        pretrain_curve_path = os.path.join(subj_folder, "pretrain_training_curves.png")
        plt.savefig(pretrain_curve_path)
        plt.close()
        print(f"Test subject {test_subj}: Pre-training curves saved to {pretrain_curve_path}")
        for layer in model.layers:
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.BatchNormalization, tf.keras.layers.MaxPooling2D, tf.keras.layers.GlobalAveragePooling2D)):
                layer.trainable = False
            else:
                layer.trainable = True
        model.compile(optimizer=Adam(learning_rate=fine_tune_lr), loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"Test subject {test_subj}: Starting fine-tuning phase...")
        unfreeze_schedule = {
            3: ['cnn_branch_layer/conv3'],
            6: ['cnn_branch_layer/conv2'],
            9: ['cnn_branch_layer/conv1', 'transformer_encoder_layer'],
        }
        gradual_unfreeze_cb = GradualUnfreeze(unfreeze_schedule, unfreeze_lr=unfreeze_lr)
        scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: fine_tune_lr * np.cos(epoch / epochs * np.pi / 2))
        history_finetune = model.fit(ft_dataset, epochs=20, validation_data=val_dataset, callbacks=[gradual_unfreeze_cb, scheduler], verbose=1)
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(history_finetune.history['loss'], label='Fine-tune Train Loss')
        plt.plot(history_finetune.history['val_loss'], label='Fine-tune Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Fine-tuning Loss Curve')
        plt.subplot(1,2,2)
        plt.plot(history_finetune.history['accuracy'], label='Fine-tune Train Acc')
        plt.plot(history_finetune.history['val_accuracy'], label='Fine-tune Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Fine-tuning Accuracy Curve')
        finetune_curve_path = os.path.join(subj_folder, "finetune_training_curves.png")
        plt.savefig(finetune_curve_path)
        plt.close()
        print(f"Test subject {test_subj}: Fine-tuning curves saved to {finetune_curve_path}")
        test_loss, test_acc = model.evaluate(test_dataset, verbose=0)
        print(f"Test subject {test_subj}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        overall_acc[test_subj] = test_acc
        y_test_int = np.argmax(y_test_final, axis=-1)
        importance_map, channel_imp = compute_channel_importance(model=model, X_test_final=X_de_test_final, y_test=y_test_int, baseline_acc=test_acc, channel_positions_9x9=channel_positions_9x9, batch_size=batch_size)
        imp_fig_path = os.path.join(subj_folder, "channel_importance.png")
        plot_channel_importance(importance_map, save_path=imp_fig_path)
        np.save(os.path.join(subj_folder, "channel_importance.npy"), channel_imp)
        y_pred_prob = model.predict(test_dataset)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(np.concatenate([y for _, y in test_dataset], axis=0), axis=1)
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, digits=2)
        overall_reports[test_subj] = report
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        cm_path = os.path.join(subj_folder, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        print(f"Test subject {test_subj}: Confusion matrix saved to {cm_path}")
        report_path = os.path.join(subj_folder, "classification.txt")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Test subject {test_subj}: Classification report saved to {report_path}")
        model_save_path = os.path.join(subj_folder, "model_eeg_9x9.keras")
        model.save(model_save_path)
        print(f"Test subject {test_subj}: Model saved to {model_save_path}")
    overall_avg_acc = np.mean(list(overall_acc.values()))
    overall_report_path = os.path.join(overall_folder, "overall_classification.txt")
    with open(overall_report_path, "w") as f:
        f.write("Overall LOSO Test Accuracy: {:.4f}\n\n".format(overall_avg_acc))
        for subj in sorted(overall_reports.keys(), key=lambda x: int(x) if x.isdigit() else x):
            f.write(f"Test Subject {subj}:\n")
            f.write(overall_reports[subj])
            f.write("\n\n")
    print(f"Overall results saved to {overall_report_path}")

# -------------------------------------------------------------------------
# 메인 실행 부분: base_dirs 설정
# -------------------------------------------------------------------------
base_dirs = [
    "/home/bcml1/2025_EMOTION/SEED_IV/eeg_DE/1",
    "/home/bcml1/2025_EMOTION/SEED_IV/eeg_DE/2",
    "/home/bcml1/2025_EMOTION/SEED_IV/eeg_DE/3",
    "/home/bcml1/2025_EMOTION/SEED_IV/eeg_PSD/1",
    "/home/bcml1/2025_EMOTION/SEED_IV/eeg_PSD/2",
    "/home/bcml1/2025_EMOTION/SEED_IV/eeg_PSD/3"
]
RESULT_DIR = "/home/bcml1/sigenv/_7주차_eeg/mj_rawDEPSD_DiffuSSM_3"

inter_subject_cv_training_9x9(
    base_dirs=base_dirs,
    result_dir=RESULT_DIR,
    epochs=150,
    batch_size=16,
    target_subjects=[str(i) for i in range(1, 16)],
    pretrain_lr=1e-4,
    fine_tune_lr=1e-4,
    unfreeze_lr=1e-5
)
