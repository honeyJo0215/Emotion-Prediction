# 기존 DiffusionDenoiseLayer 대신, PyWavelets를 이용한 wavelet 기반 denoising을 수행하는 새로운 WaveletDenoiseLayer를 정의

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pywt  # PyWavelets 라이브러리 추가
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
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import Concatenate

# -------------------------------------------------------------------------
# GPU 메모리 제한 (필요 시)
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

limit_gpu_memory(1000)

# -------------------------------------------------------------------------
# Wavelet 기반 Denoising Layer (PyWavelets 이용)
# -------------------------------------------------------------------------
class WaveletDenoiseLayer(tf.keras.layers.Layer):
    def __init__(self, wavelet='db8', level=1, threshold_factor=1.0, **kwargs):
        super(WaveletDenoiseLayer, self).__init__(**kwargs)
        self.wavelet = wavelet
        self.level = level
        self.threshold_factor = threshold_factor

    def call(self, inputs):
        # inputs shape: (batch, height, width, channels)
        def denoise_image(img):
            # img: numpy array of shape (height, width, channels)
            denoised_channels = []
            for c in range(img.shape[-1]):
                channel_data = img[..., c]
                # wavelet decomposition
                coeffs = pywt.wavedec2(channel_data, self.wavelet, level=self.level)
                # noise estimation using first level detail coefficients
                # coeffs[1] is a tuple: (cH, cV, cD)
                detail_coeffs = coeffs[1]
                detail_arr = np.concatenate([np.ravel(subband) for subband in detail_coeffs])
                sigma = np.median(np.abs(detail_arr)) / 0.6745
                # universal threshold
                uthresh = sigma * np.sqrt(2 * np.log(channel_data.size)) * self.threshold_factor
                # thresholding detail coefficients
                new_coeffs = [coeffs[0]]  # approximation remains unchanged
                for detail in coeffs[1:]:
                    new_detail = tuple(pywt.threshold(subband, value=uthresh, mode='soft') for subband in detail)
                    new_coeffs.append(new_detail)
                # reconstruct image
                denoised_channel = pywt.waverec2(new_coeffs, self.wavelet)
                # crop to original shape if necessary
                denoised_channel = denoised_channel[:channel_data.shape[0], :channel_data.shape[1]]
                denoised_channels.append(denoised_channel[..., np.newaxis])
            denoised_img = np.concatenate(denoised_channels, axis=-1)
            return denoised_img.astype(np.float32)
        
        # tf.map_fn로 배치 내 각 샘플에 대해 적용
        def process_sample(x):
            denoised = denoise_image(x)
            return denoised
        
        output = tf.map_fn(lambda x: tf.py_function(process_sample, [x], Tout=tf.float32), inputs)
        output.set_shape(inputs.shape)
        return output

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
# 기존 전처리 및 2D 매핑 helper 함수들 (변경 없음)
# -------------------------------------------------------------------------
def split_trial_to_time_slices(trial):
    channels, T, bands = trial.shape
    slices = [trial[:, t:t+1, :] for t in range(T)]
    return np.array(slices)

def replicate_labels_for_trials(trials_list, labels):
    replicated = [np.repeat(label[np.newaxis, :], trial.shape[1] // 2, axis=0) 
                  for trial, label in zip(trials_list, labels)]
    return np.concatenate(replicated, axis=0) if replicated else None

def load_seediv_data(base_dirs, de_keys=["de_movingAve"], psd_keys=["psd_movingAve"]):
    data_de = {}
    data_psd = {}
    for base_dir in base_dirs:
        file_list = glob.glob(os.path.join(base_dir, "*.npy"))
        for file in file_list:
            filename = os.path.basename(file)
            parts = filename.replace('.npy','').split('_')
            if len(parts) < 8:
                continue
            subject, trial = parts[0], parts[3]
            key_name = parts[4] + "_" + parts[5]
            try:
                label_val = int(parts[7])
            except:
                continue
            arr = np.load(file)[..., 1:]
            if key_name in de_keys:
                data_de[(subject, trial)] = (arr, label_val)
            elif key_name in psd_keys:
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

def map_channels_to_2d_9x9(segment, channel_positions):
    num_channels, num_bands = segment.shape
    mapped = np.zeros((9, 9, num_bands), dtype=np.float32)
    for ch_idx in range(num_channels):
        if ch_idx not in channel_positions:
            continue
        row, col = channel_positions[ch_idx]
        for band_idx in range(num_bands):
            mapped[row, col, band_idx] = segment[ch_idx, band_idx]
    return mapped

def process_slice_9x9(slice_, channel_positions):
    slice_2d = np.squeeze(slice_, axis=1)  # (62,4)
    mapped_9x9 = map_channels_to_2d_9x9(slice_2d, channel_positions)  # (9,9,4)
    img = np.transpose(mapped_9x9, (2,0,1))  # (4,9,9)
    img = np.expand_dims(img, axis=-1)        # (4,9,9,1)
    return img

def process_trial_9x9(de_trial, psd_trial, channel_positions):
    slices_de = split_trial_to_time_slices(de_trial)    # (T, 62,1,bands)
    slices_psd = split_trial_to_time_slices(psd_trial)    # (T, 62,1,bands)
    T = slices_de.shape[0]
    if T % 2 != 0:
        T = T - 1
    sample_list = []
    for t in range(0, T, 2):
        combined_slice1 = np.concatenate([slices_de[t], slices_psd[t]], axis=1)
        combined_slice2 = np.concatenate([slices_de[t+1], slices_psd[t+1]], axis=1)
        imgs1 = []
        imgs2 = []
        for feature_idx in range(2):  # 0: DE, 1: PSD
            img1 = process_slice_9x9(combined_slice1[:, feature_idx:feature_idx+1, :], channel_positions)
            img2 = process_slice_9x9(combined_slice2[:, feature_idx:feature_idx+1, :], channel_positions)
            imgs1.append(img1)
            imgs2.append(img2)
        sample1 = np.stack(imgs1, axis=0)  # (2,4,9,9,1)
        sample2 = np.stack(imgs2, axis=0)  # (2,4,9,9,1)
        sample = np.expand_dims(np.stack([sample1, sample2], axis=0), axis=0)  # (1,2,2,4,9,9,1)
        sample_list.append(sample)
    return np.concatenate(sample_list, axis=0)

# =============================================================================
# CNN+Transformer 관련 Layer들 (기존과 동일)
# =============================================================================
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

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation=tf.nn.gelu, kernel_regularizer=l2(1e-4)),
            tf.keras.layers.Dense(d_model, kernel_regularizer=l2(1e-4))
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
    def call(self, inputs, training=False):
        attn_output = self.mha(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# =============================================================================
# CNN Branch: 수정 – 입력 2D 매핑 이미지에 대해 채널(주파수밴드)별로 학습 가능한 가중치(band_weights) 적용
# =============================================================================
class CNNBranchLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_bands=4, **kwargs):
        super(CNNBranchLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_bands = num_bands
        self.preprocess = PreprocessLayer()
        self.attention = SpatialSpectralAttention()
        self.band_weights = self.add_weight(shape=(num_bands, 1, 1, 1),
                                            initializer='ones',
                                            trainable=True,
                                            name='band_weights')
        self.conv1 = SpatialSpectralConvModule(8, kernel_size=(3,3), strides=(3,3))
        self.conv2 = SpatialSpectralConvModule(16, kernel_size=(1,1), strides=(1,1))
        self.conv3 = SpatialSpectralConvModule(32, kernel_size=(1,1), strides=(1,1))
        self.flatten = Flatten()
        self.dense = Dense(d_model, activation="relu", kernel_regularizer=l2(1e-4))
    def call(self, x):
        x = self.preprocess(x)
        x = x * self.band_weights
        x = self.attention(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

# =============================================================================
# 단일 세그먼트 모델: 입력 shape: (2, 4, 9, 9, 1) (2: [DE, PSD])
# Gaussian 노이즈 추가 후 WaveletDenoiseLayer를 적용하고, 이후 CNN branch로 특징 추출
# =============================================================================
def build_single_segment_model(d_model=64):
    input_seg = Input(shape=(2, 4, 9, 9, 1))
    # 모달리티 분리
    de_image = Lambda(lambda x: x[:,0,...])(input_seg)  # (4,9,9,1)
    psd_image = Lambda(lambda x: x[:,1,...])(input_seg) # (4,9,9,1)
    # Gaussian 노이즈 추가 (각 프레임에 대해)
    de_noisy = Lambda(lambda x: x + tf.random.normal(tf.shape(x), mean=0.0, stddev=0.1))(de_image)
    psd_noisy = Lambda(lambda x: x + tf.random.normal(tf.shape(x), mean=0.0, stddev=0.1))(psd_image)
    # Wavelet Denoise: DiffusionDenoiseLayer 대신 WaveletDenoiseLayer 적용
    de_denoised = TimeDistributed(WaveletDenoiseLayer(wavelet='db8', level=1, threshold_factor=1.0))(de_noisy)  # (4,9,9,1)
    psd_denoised = TimeDistributed(WaveletDenoiseLayer(wavelet='db8', level=1, threshold_factor=1.0))(psd_noisy)  # (4,9,9,1)
    # CNN Branch를 통한 특징 추출 (채널 중요도 반영 포함)
    cnn_de = CNNBranchLayer(d_model, num_bands=4)
    cnn_psd = CNNBranchLayer(d_model, num_bands=4)
    feature_de = cnn_de(de_denoised)    # (d_model,)
    feature_psd = cnn_psd(psd_denoised)   # (d_model,)
    combined_features = Concatenate()([feature_de, feature_psd])  # (2*d_model,)
    combined_features = Dense(d_model, activation="relu", kernel_regularizer=l2(1e-4))(combined_features)
    model = Model(inputs=input_seg, outputs=combined_features)
    return model

# =============================================================================
# 최종 모델: TimeDistributed로 각 세그먼트를 처리한 후,
# diffusion forward 과정을 모사하기 위해 노이즈를 추가하고,
# 여러 DiffuSSMLayer 블록을 통해 역확산(denoising) 및 특징 재구성을 수행한 후 감정 분류
# =============================================================================
def build_model(input_shape=(2, 2, 4, 9, 9, 1), n_diff_layers=2, d_ff=512, p_drop=0.5, d_model=64, noise_std=0.1):
    """
    모델 입력: (T, 2, 4, 9, 9, 1)
      - T = 2: 2초 동안의 연속된 세그먼트 (예: 2초)
      - 2: 2개의 모달리티 (DE, PSD)
      - 4: 각 세그먼트당 4 프레임 (4밴드)
      - 9x9: 2D 매핑 이미지 크기, 마지막 1: 채널
    """
    inputs = Input(shape=input_shape)
    # 1초 세그먼트마다 단일 세그먼트 모델 적용 (TimeDistributed)
    single_seg_model = build_single_segment_model(d_model)
    features_seq = TimeDistributed(single_seg_model, name="TimeDistributed_CNN")(inputs)
    # diffusion forward: 노이즈 추가하여 forward diffusion 과정 모사
    noisy_features_seq = Lambda(lambda x: x + tf.random.normal(tf.shape(x), mean=0.0, stddev=noise_std), name="add_diffusion_noise")(features_seq)
    # 역확산(denoising) 및 특징 재구성: DiffuSSMLayer n_diff_layers 번 적용
    x = noisy_features_seq  # shape: (batch, T, d_model)
    for _ in range(n_diff_layers):
        x = DiffuSSMLayer(d_model, d_ff, dropout_rate=p_drop)(x)
    # 최종적으로 시퀀스 정보를 flatten 후 감정 분류 (4 클래스)
    x = Flatten()(x)
    outputs = Dense(4, activation="softmax", kernel_regularizer=l2(1e-4))(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# =============================================================================
# (이하 GradualUnfreeze, compute_channel_importance, plot_channel_importance, inter_subject_cv_training_9x9 등은 기존과 동일)
# =============================================================================
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

def compute_channel_importance(model, X_test_final, y_test, baseline_acc, 
                               channel_positions_9x9, batch_size=16):
    num_channels = 62
    channel_importance = np.zeros(num_channels, dtype=np.float32)
    for ch in range(num_channels):
        X_test_mod = np.copy(X_test_final)
        if ch in channel_positions_9x9:
            row, col = channel_positions_9x9[ch]
            X_test_mod[:, :, :, :, row, col, :] = 0.0
        test_dataset_mod = tf.data.Dataset.from_tensor_slices(
            (X_test_mod, to_categorical(y_test, num_classes=4))
        ).batch(batch_size)
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

def inter_subject_cv_training_9x9(base_dirs, result_dir,
                                  epochs=150, batch_size=16,
                                  target_subjects=None,
                                  pretrain_lr=3e-4, fine_tune_lr=3e-4, unfreeze_lr=3e-5):
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
        X_de_train_slices = [process_trial_9x9(de, psd, channel_positions_9x9) 
                            for de, psd in zip(X_de_train_split, X_psd_train_split)]
        y_train = replicate_labels_for_trials(X_de_train_split, y_train_split)
        X_de_train = np.concatenate(X_de_train_slices, axis=0)
        X_de_val_slices = [process_trial_9x9(de, psd, channel_positions_9x9) 
                        for de, psd in zip(X_de_val_split, X_psd_val_split)]
        y_val = replicate_labels_for_trials(X_de_val_split, y_val_split)
        X_de_val = np.concatenate(X_de_val_slices, axis=0)
        X_de_test_ft_slices = [process_trial_9x9(de, psd, channel_positions_9x9) 
                            for de, psd in zip(X_de_test_ft, X_psd_test_ft)]
        y_test_ft = replicate_labels_for_trials(X_de_test_ft, y_test_ft)
        X_de_test_ft = np.concatenate(X_de_test_ft_slices, axis=0)
        X_de_test_final_slices = [process_trial_9x9(de, psd, channel_positions_9x9) 
                                for de, psd in zip(X_de_test_final, X_psd_test_final)]
        y_test_final = replicate_labels_for_trials(X_de_test_final, y_test_final)
        X_de_test_final = np.concatenate(X_de_test_final_slices, axis=0)
        print(f"X_de_test_ft shape: {X_de_test_ft.shape}, y_test_ft shape: {y_test_ft.shape}")
        print(f"X_de_test_final shape: {X_de_test_final.shape}, y_test_final shape: {y_test_final.shape}")
        print(f"X_de_train.shape: {X_de_train.shape}, X_de_val.shape: {X_de_val.shape}")
        print(f"X_de_test_ft.shape: {X_de_test_ft.shape}, X_de_test_final.shape: {X_de_test_final.shape}")
        train_dataset = tf.data.Dataset.from_tensor_slices((X_de_train, y_train)).shuffle(1000).batch(batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_de_val, y_val)).batch(batch_size)
        ft_dataset = tf.data.Dataset.from_tensor_slices((X_de_test_ft, y_test_ft)).shuffle(1000).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((X_de_test_final, y_test_final)).batch(batch_size)        
        print(f"Test subject {test_subj}: Starting pre-training phase for {epochs} epochs...")
        model = build_model(input_shape=(2, 2, 4, 9, 9, 1), n_diff_layers=2, d_ff=512, p_drop=0.3, d_model=64, noise_std=0.1)
        
        model.compile(optimizer=Adam(learning_rate=pretrain_lr),
                      loss="categorical_crossentropy", metrics=["accuracy"])
        model.summary()
        pretrain_callbacks = [EarlyStopping(monitor='val_accuracy', patience=50, min_delta=0.001, restore_best_weights=True)]
        history_pretrain = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset,
                                     callbacks=pretrain_callbacks, verbose=1)
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
            if isinstance(layer, (tf.keras.layers.Conv2D,
                                  tf.keras.layers.BatchNormalization,
                                  tf.keras.layers.MaxPooling2D,
                                  tf.keras.layers.GlobalAveragePooling2D)):
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
        scheduler = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: fine_tune_lr * np.cos(epoch / epochs * np.pi / 2)
        )
        history_finetune = model.fit(
            ft_dataset, epochs=20, 
            validation_data=val_dataset,
            callbacks=[gradual_unfreeze_cb, scheduler],
            verbose=1
        )
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
        importance_map, channel_imp = compute_channel_importance(
            model=model,
            X_test_final=X_de_test_final,
            y_test=y_test_int,
            baseline_acc=test_acc,
            channel_positions_9x9=channel_positions_9x9,
            batch_size=batch_size
        )
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
    
# (아래는 메인 실행 부분 예시)
base_dirs = [
    "/home/bcml1/2025_EMOTION/SEED_IV/eeg_feature_smooth/1_npy_sample",
    "/home/bcml1/2025_EMOTION/SEED_IV/eeg_feature_smooth/2_npy_sample",
    "/home/bcml1/2025_EMOTION/SEED_IV/eeg_feature_smooth/3_npy_sample"
]
RESULT_DIR = "/home/bcml1/sigenv/_7주차_eeg/mj_Diffu_ele"

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
