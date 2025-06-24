#transformer 레이어를 제외한 코드. subject 별 정규화를 적용한 코드임

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

limit_gpu_memory(5000)

# -------------------------------------------------------------------------
# subject normalize 함수 및 TF 래퍼
# -------------------------------------------------------------------------
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

def tf_subject_normalize(x, subject_info):
    # x: (batch, T, 2, 4, 9, 9, 1)
    x_norm = tf.py_function(func=lambda x: subject_normalize(x, subject_info),
                            inp=[x],
                            Tout=tf.float32)
    x_norm.set_shape(x.get_shape())
    return x_norm

# -------------------------------------------------------------------------
# DDPM 기반 Denoiser (학습 가능한 UNet denoiser)
# -------------------------------------------------------------------------
class DenoiseUNet(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DenoiseUNet, self).__init__(**kwargs)
        self.conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')
        self.pool1 = MaxPooling2D((2, 2), padding='same')
        self.conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')
        self.up1 = tf.keras.layers.UpSampling2D((2, 2))
        self.crop = Lambda(lambda z: z[:, :9, :9, :])
        self.conv3 = Conv2D(16, (3, 3), activation='relu', padding='same')
        self.out_conv = Conv2D(1, (1, 1), activation='linear', padding='same')
    
    def call(self, inputs):
        # inputs: (batch, 9, 9, 1)
        x1 = self.conv1(inputs)          # (None, 9, 9, 16)
        x = self.pool1(x1)               # (None, 5, 5, 16)
        x = self.conv2(x)                # (None, 5, 5, 32)
        x = self.up1(x)                  # (None, 10, 10, 32)
        x = self.crop(x)                 # (None, 9, 9, 32)
        x = concatenate([x1, x])         # (None, 9, 9, 48)
        x = self.conv3(x)                # (None, 9, 9, 16)
        out = self.out_conv(x)           # (None, 9, 9, 1)
        return out

class DiffusionDenoiseLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DiffusionDenoiseLayer, self).__init__(**kwargs)
        self.denoiser = DenoiseUNet()
    def call(self, inputs):
        return self.denoiser(inputs)
    def compute_output_shape(self, input_shape):
        return input_shape

# -------------------------------------------------------------------------
# SEBlock (채널 self-attention)
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
# SpatialSpectralConvModule & SpatialSpectralAttention
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
# TransformerEncoderLayer (Transformer 레이어는 기본 1개로 축소)
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
# DiffuSSMLayer (Diffusion 기반 SSM Layer)
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
# CNNBranchLayer (SEBlock와 공간/채널 가중치 적용)
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
        x = x * self.band_weights
        x = x * self.spatial_weights
        x = self.attention(x)
        x = self.se_block(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.d_model)

# -------------------------------------------------------------------------
# 단일 세그먼트 모델: (2, 4, 9, 9, 1) 입력, 모달리티별 노이즈 추가, DDPM 기반 denoising, CNNBranch 적용
# -------------------------------------------------------------------------
def build_single_segment_model(d_model=64):
    input_seg = Input(shape=(2, 4, 9, 9, 1))
    de_image = Lambda(lambda x: x[:,0,...])(input_seg)
    psd_image = Lambda(lambda x: x[:,1,...])(input_seg)
    de_noisy = Lambda(lambda x: x + tf.random.normal(tf.shape(x), mean=0.0, stddev=0.1))(de_image)
    psd_noisy = Lambda(lambda x: x + tf.random.normal(tf.shape(x), mean=0.0, stddev=0.1))(psd_image)
    de_denoised = TimeDistributed(DiffusionDenoiseLayer())(de_noisy)
    psd_denoised = TimeDistributed(DiffusionDenoiseLayer())(psd_noisy)
    cnn_de = CNNBranchLayer(d_model, num_bands=4)
    cnn_psd = CNNBranchLayer(d_model, num_bands=4)
    feature_de = cnn_de(de_denoised)
    feature_psd = cnn_psd(psd_denoised)
    combined_features = Concatenate()([feature_de, feature_psd])
    combined_features = Dense(d_model, activation="relu", kernel_regularizer=l2(1e-4))(combined_features)
    model = Model(inputs=input_seg, outputs=combined_features)
    return model

# -------------------------------------------------------------------------
# 최종 모델: 두 입력 ([data, subject_info])을 받아서 처리
# -------------------------------------------------------------------------
def build_model(data_input_shape=(10, 2, 4, 9, 9, 1),
                subject_info_shape=(1,),
                n_diff_layers=2,
                n_transformer_layers=1,
                d_ff=512,
                p_drop=0.5,
                d_model=64,
                noise_std=0.1):
    data_input = Input(shape=data_input_shape, name="data_input")
    subject_input = Input(shape=subject_info_shape, name="subject_input")
    
    # subject 정규화 적용
    x = Lambda(lambda inputs: tf_subject_normalize(inputs[0], inputs[1]),
               name="subject_normalize")([data_input, subject_input])
    
    single_seg_model = build_single_segment_model(d_model)
    features_seq = TimeDistributed(single_seg_model, name="TimeDistributed_CNN")(x)
    
    noisy_features_seq = Lambda(lambda x: x + tf.random.normal(tf.shape(x), mean=0.0, stddev=noise_std),
                                name="add_diffusion_noise")(features_seq)
    x = noisy_features_seq
    for _ in range(n_diff_layers):
        x = DiffuSSMLayer(d_model, d_ff, dropout_rate=p_drop)(x)
    # Transformer layer 생략 또는 주석 처리 (필요 시 추가 가능)
    x = Flatten()(x)
    outputs = Dense(4, activation="softmax", kernel_regularizer=l2(1e-4))(x)
    
    model = Model(inputs=[data_input, subject_input], outputs=outputs)
    return model

# -------------------------------------------------------------------------
# Helper 함수들: 슬라이딩 윈도우 증강 및 기타
# -------------------------------------------------------------------------
def split_trial_to_1sec_slices(trial):
    return [trial[:, i:i+1, :] for i in range(trial.shape[1])]

def process_slice_9x9(slice_, channel_positions):
    slice_2d = np.squeeze(slice_, axis=1)  # (62, 4)
    mapped = np.zeros((9, 9, slice_2d.shape[-1]), dtype=np.float32)
    for ch in range(slice_2d.shape[0]):
        if ch not in channel_positions:
            continue
        row, col = channel_positions[ch]
        mapped[row, col, :] = slice_2d[ch, :]
    img = np.transpose(mapped, (2, 0, 1))  # (4,9,9)
    img = np.expand_dims(img, axis=-1)     # (4,9,9,1)
    return img

# def process_trial_9x9_new(de_trial, psd_trial, channel_positions, group_size=10):
#     slices_de = split_trial_to_1sec_slices(de_trial)
#     slices_psd = split_trial_to_1sec_slices(psd_trial)
#     num_groups = len(slices_de) - group_size + 1
#     sample_list = []
#     for g in range(num_groups):
#         group_de = slices_de[g:g+group_size]
#         group_psd = slices_psd[g:g+group_size]
#         time_steps = []
#         for t in range(group_size):
#             de_img = process_slice_9x9(group_de[t], channel_positions)
#             psd_img = process_slice_9x9(group_psd[t], channel_positions)
#             modality_stack = np.stack([de_img, psd_img], axis=0)
#             time_steps.append(modality_stack)
#         sample = np.stack(time_steps, axis=0)
#         sample_list.append(sample)
#     return np.array(sample_list)


# def replicate_subjects_for_trials(trials_list, subject_labels):
#     """
#     각 trial의 슬라이스 수에 맞춰 subject 정보를 복제하여, 
#     최종적으로 (total_segments, 1) 형태의 numpy array를 반환합니다.
#     """
#     replicated = []
#     total_segments = 0
#     for trial, subj in zip(trials_list, subject_labels):
#         # subj가 리스트가 아닌 단일 값(예: "2")가 되도록 강제 변환
#         subj = str(subj)  
#         num_segments = trial.shape[0]
#         total_segments += num_segments
#         rep = np.repeat(subj, num_segments)  # rep는 (num_segments,) shape의 1차원 배열
#         replicated.append(rep)
#     all_subjects = np.concatenate(replicated, axis=0) if replicated else None
#     all_subjects = all_subjects.reshape(-1, 1)
#     print("replicate_subjects_for_trials: 총 세그먼트 수 =", total_segments, "복제된 subject 정보 shape =", all_subjects.shape)
#     return all_subjects

# def replicate_labels_for_trials(trials_list, labels):
#     replicated = []
#     total_segments = 0
#     for trial, label in zip(trials_list, labels):
#         num_segments = trial.shape[0]
#         total_segments += num_segments
#         rep = np.repeat(label[np.newaxis, :], num_segments, axis=0)
#         replicated.append(rep)
#     all_labels = np.concatenate(replicated, axis=0) if replicated else None
#     assert all_labels.shape[0] == total_segments, (
#         f"라벨 복제 오류: 총 세그먼트 수는 {total_segments}개인데, 복제된 라벨 수는 {all_labels.shape[0]}개입니다."
#     )
#     print("replicate_labels_for_trials: 총 세그먼트 수 =", total_segments, "복제된 라벨 shape =", all_labels.shape)
#     return all_labels
def process_trial_9x9_new(de_trial, psd_trial, channel_positions, group_size=10):
    slices_de = split_trial_to_1sec_slices(de_trial)
    slices_psd = split_trial_to_1sec_slices(psd_trial)
    # 슬라이스 수가 group_size보다 작으면 빈 배열 반환
    if len(slices_de) < group_size:
        return np.empty((0, group_size, 2, 4, 9, 9, 1), dtype=np.float32)
    num_groups = len(slices_de) - group_size + 1
    sample_list = []
    for g in range(num_groups):
        group_de = slices_de[g:g+group_size]
        group_psd = slices_psd[g:g+group_size]
        time_steps = []
        for t in range(group_size):
            de_img = process_slice_9x9(group_de[t], channel_positions)
            psd_img = process_slice_9x9(group_psd[t], channel_positions)
            modality_stack = np.stack([de_img, psd_img], axis=0)
            time_steps.append(modality_stack)
        sample = np.stack(time_steps, axis=0)
        sample_list.append(sample)
    return np.array(sample_list)


def replicate_subjects_for_trials(trials_list, subject_labels):
    """
    각 trial의 group 수에 맞춰 subject 정보를 복제하여,
    최종적으로 (total_groups, 1) 형태의 numpy array를 반환합니다.
    빈 trial은 건너뜁니다.
    """
    replicated = []
    total_groups = 0
    for trial, subj in zip(trials_list, subject_labels):
        if trial.shape[0] == 0:
            continue  # 빈 trial은 건너뛰기
        subj = str(subj)  # 단일 값으로 변환
        num_groups = trial.shape[0]
        total_groups += num_groups
        rep = np.repeat(subj, num_groups)  # 1차원 배열, 각 그룹마다 subject 정보 복제
        replicated.append(rep)
    if replicated:
        all_subjects = np.concatenate(replicated, axis=0)
        all_subjects = all_subjects.reshape(-1, 1)
    else:
        all_subjects = np.empty((0, 1))
    print("replicate_subjects_for_trials: 총 그룹 수 =", total_groups, "복제된 subject 정보 shape =", all_subjects.shape)
    return all_subjects


def replicate_labels_for_trials(trials_list, labels):
    """
    각 trial의 group 수에 맞춰 라벨 정보를 복제하여,
    최종적으로 (total_groups, label_dim) 형태의 numpy array를 반환합니다.
    빈 trial은 건너뜁니다.
    """
    replicated = []
    total_groups = 0
    for trial, label in zip(trials_list, labels):
        if trial.shape[0] == 0:
            continue
        num_groups = trial.shape[0]
        total_groups += num_groups
        rep = np.repeat(label[np.newaxis, :], num_groups, axis=0)
        replicated.append(rep)
    if replicated:
        all_labels = np.concatenate(replicated, axis=0)
    else:
        all_labels = np.empty((0,))
    assert all_labels.shape[0] == total_groups, (
        f"라벨 복제 오류: 총 그룹 수는 {total_groups}개인데, 복제된 라벨 수는 {all_labels.shape[0]}개입니다."
    )
    print("replicate_labels_for_trials: 총 그룹 수 =", total_groups, "복제된 라벨 shape =", all_labels.shape)
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
# LOSO 학습/평가 파이프라인
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

def inter_subject_cv_training_9x9(base_dirs, result_dir, epochs=150, batch_size=16,
                                  target_subjects=None, pretrain_lr=1e-4,
                                  fine_tune_lr=1e-4, unfreeze_lr=1e-5):
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
        subject_train_list = []
        for subj in subjects:
            if subj == test_subj:
                continue
            X_de_train_trials.extend(subject_data[subj]["de"])
            X_psd_train_trials.extend(subject_data[subj]["psd"])
            y_train_list.extend(subject_data[subj]["labels"])
            subject_train_list.extend([subj] * len(subject_data[subj]["de"]))
        
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
        subject_train_split = np.array(subject_train_list)[train_idx]
        subject_val_split = np.array(subject_train_list)[val_idx]
        
        X_de_train_slices = [process_trial_9x9_new(de, psd, channel_positions_9x9, group_size=10)
                              for de, psd in zip(X_de_train_split, X_psd_train_split)]
        y_train = replicate_labels_for_trials(X_de_train_slices, y_train_split)
        # subject_info_train = replicate_subjects_for_trials(X_de_train_slices, subject_train_split)
        subject_info_train = replicate_subjects_for_trials(X_de_train_slices, subject_train_split)
        X_de_train = np.concatenate(X_de_train_slices, axis=0)
        
        X_de_val_slices = [process_trial_9x9_new(de, psd, channel_positions_9x9, group_size=10)
                            for de, psd in zip(X_de_val_split, X_psd_val_split)]
        y_val = replicate_labels_for_trials(X_de_val_slices, y_val_split)
        # subject_info_val = replicate_subjects_for_trials(X_de_val_slices, subject_val_split)
        subject_info_val = replicate_subjects_for_trials(X_de_val_slices, subject_val_split)
        X_de_val = np.concatenate(X_de_val_slices, axis=0)
        
        X_de_test_ft_slices = [process_trial_9x9_new(de, psd, channel_positions_9x9, group_size=10)
                                for de, psd in zip(X_de_test_ft, X_psd_test_ft)]
        y_test_ft = replicate_labels_for_trials(X_de_test_ft_slices, y_test_ft)
        # subject_info_ft = replicate_subjects_for_trials(X_de_test_ft_slices, np.array([test_subj]*len(X_de_test_ft_slices)))
        subject_info_ft = replicate_subjects_for_trials(X_de_test_ft_slices, np.array([test_subj]*len(X_de_test_ft_slices)))
        X_de_test_ft = np.concatenate(X_de_test_ft_slices, axis=0)
        
        X_de_test_final_slices = [process_trial_9x9_new(de, psd, channel_positions_9x9, group_size=10)
                                   for de, psd in zip(X_de_test_final, X_psd_test_final)]
        y_test_final = replicate_labels_for_trials(X_de_test_final_slices, y_test_final)
        # subject_info_test = replicate_subjects_for_trials(X_de_test_final_slices, np.array([test_subj]*len(X_de_test_final_slices)))
        subject_info_test = replicate_subjects_for_trials(X_de_test_final_slices, np.array([test_subj]*len(X_de_test_final_slices)))
        X_de_test_final = np.concatenate(X_de_test_final_slices, axis=0)
        
        print(f"X_de_test_ft shape: {X_de_test_ft.shape}, y_test_ft shape: {y_test_ft.shape}")
        print(f"X_de_test_final shape: {X_de_test_final.shape}, y_test_final shape: {y_test_final.shape}")
        print(f"X_de_train.shape: {X_de_train.shape}, X_de_val.shape: {X_de_val.shape}")
        
        train_dataset = tf.data.Dataset.from_tensor_slices(([X_de_train, subject_info_train], y_train)).shuffle(1000).batch(batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices(([X_de_val, subject_info_val], y_val)).batch(batch_size)
        ft_dataset = tf.data.Dataset.from_tensor_slices(([X_de_test_ft, subject_info_ft], y_test_ft)).shuffle(1000).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices(([X_de_test_final, subject_info_test], y_test_final)).batch(batch_size)
        
        print(f"Test subject {test_subj}: Starting pre-training phase for {epochs} epochs...")
        model = build_model(data_input_shape=(10, 2, 4, 9, 9, 1), subject_info_shape=(1,),
                            n_diff_layers=2, n_transformer_layers=1, d_ff=512, p_drop=0.3,
                            d_model=64, noise_std=0.1)
        model.compile(optimizer=Adam(learning_rate=pretrain_lr), loss="categorical_crossentropy", metrics=["accuracy"])
        model.summary()
        pretrain_callbacks = [EarlyStopping(monitor='val_accuracy', patience=50, min_delta=0.001, restore_best_weights=True)]
        
        history_pretrain = model.fit(
            [X_de_train, subject_info_train],
            y_train,
            validation_data=([X_de_val, subject_info_val], y_val),
            epochs=epochs,
            callbacks=pretrain_callbacks,
            verbose=1
        )
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
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.BatchNormalization,
                                  tf.keras.layers.MaxPooling2D, tf.keras.layers.GlobalAveragePooling2D)):
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
RESULT_DIR = "/home/bcml1/sigenv/_7주차_eeg/mj_rawDEPSD_DiffuSSM_5"

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
