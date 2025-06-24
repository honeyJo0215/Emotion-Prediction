# 모델을 구현할 때, EEG 신호를 2D 매핑을 할 때, 그거를 노이즈가 들어간 이미지로 만들고, 
# 이 이미지를 diffusion 모델로 확산시켜서 denoise를 진행해서, 
# 원래의 2D mapping의 신호를 확산시켜서 DiffuSSM을 거쳐서 4가지로 감정을 분류하는 모델

# 9x9 2D mapping일 때의 것을 이용해
# DiffuSSM 모델
# Diffusion Model 이용해서 2D mapping할 때의 것을 해보기....!!!!!!!!
# 노이즈를 낀 데이터를 원본으로 돌려서 합치는 것.


# +2D mapping에서 각 채널 들 중에서 감정인식을 할 때 큰 영향을 주는 부분의 
# weight를 더 크게 설정하는 것도 추가해보면 어떨까? 잘 나오지 않을까?

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
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

limit_gpu_memory(10000)

# -------------------------------------------------------------------------
# DiffusionDenoiseLayer: 2D 입력 (예: (4,9,9,1))에 대해 노이즈 제거를 모사하는 diffusion block
# -------------------------------------------------------------------------
class DiffusionDenoiseLayer(tf.keras.layers.Layer):
    def __init__(self, num_steps=3, filters=16, **kwargs):
        super(DiffusionDenoiseLayer, self).__init__(**kwargs)
        self.num_steps = num_steps
        self.conv_layers = []
        self.bn_layers = []
        for _ in range(num_steps):
            self.conv_layers.append(
                tf.keras.layers.Conv2D(filters, kernel_size=(3,3), padding='same', activation='relu')
            )
            self.bn_layers.append(tf.keras.layers.BatchNormalization())
        self.final_conv = tf.keras.layers.Conv2D(1, kernel_size=(1,1), padding='same')

    def call(self, x):
        out = x
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            out = conv(out)
            out = bn(out)
        out = self.final_conv(out)
        return out

    def compute_output_shape(self, input_shape):
        # TimeDistributed에서 이 정보가 필요함
        return input_shape[:-1] + (1,)


#감정인식에서 중요한 채널 weight를 더 크게 하는 방식-> 아직 사용하지는 않을거.
class ChannelWeightedAttention(tf.keras.layers.Layer):
    def __init__(self, num_channels=4):
        super(ChannelWeightedAttention, self).__init__()
        self.dense = tf.keras.layers.Dense(num_channels, activation='softmax')

    def call(self, x):
        # x shape: (batch, 4, 9, 9, 1)
        weights = self.dense(tf.reduce_mean(x, axis=[2,3,4]))  # (batch, 4)
        weights = tf.reshape(weights, (-1, 4, 1, 1, 1))
        return x * weights
# -------------------------------------------------------------------------
# 기존 helper 함수들 (EEG trial 슬라이스 분리, 채널 2D 매핑 등)
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

# === CNN+Transformer 관련 Layer들 (변경 없음) ===
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

# === Diffusion를 적용한 Single segment model ===
def build_single_segment_model(d_model=64):
    # 입력 shape: (2, 4, 9, 9, 1) → 2: [DE, PSD]
    input_seg = tf.keras.layers.Input(shape=(2, 4, 9, 9, 1))
    # 각 모달리티 분리
    de_image = tf.keras.layers.Lambda(lambda x: x[:,0,...])(input_seg)    # (4,9,9,1)
    psd_image = tf.keras.layers.Lambda(lambda x: x[:,1,...])(input_seg)   # (4,9,9,1)
    
    # Gaussian 노이즈 추가 (각 프레임에 대해)
    de_noisy = tf.keras.layers.Lambda(lambda x: x + tf.random.normal(tf.shape(x), mean=0.0, stddev=0.1))(de_image)
    psd_noisy = tf.keras.layers.Lambda(lambda x: x + tf.random.normal(tf.shape(x), mean=0.0, stddev=0.1))(psd_image)
    
    # TimeDistributed 래퍼를 사용하여 각 프레임별로 DiffusionDenoiseLayer 적용
    de_denoised = TimeDistributed(DiffusionDenoiseLayer())(de_noisy)   # 각 프레임에 대해 denoising, 출력: (4,9,9,1)
    psd_denoised = TimeDistributed(DiffusionDenoiseLayer())(psd_noisy)   # 출력: (4,9,9,1)
    
    # 이후 CNN branch를 통해 특징 추출
    cnn_de = CNNBranchLayer(d_model)
    cnn_psd = CNNBranchLayer(d_model)
    feature_de = cnn_de(de_denoised)      # (d_model,)
    feature_psd = cnn_psd(psd_denoised)     # (d_model,)
    
    combined_features = tf.keras.layers.Concatenate()([feature_de, feature_psd])  # (2*d_model,)
    combined_features = tf.keras.layers.Dense(d_model, activation="relu", kernel_regularizer=l2(1e-4))(combined_features)
    model = tf.keras.models.Model(inputs=input_seg, outputs=combined_features)
    return model

# === CNN Branch ===
class CNNBranchLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, **kwargs):
        super(CNNBranchLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.preprocess = PreprocessLayer()
        self.attention = SpatialSpectralAttention()
        self.conv1 = SpatialSpectralConvModule(8, kernel_size=(3,3), strides=(3,3))
        self.conv2 = SpatialSpectralConvModule(16, kernel_size=(1,1), strides=(1,1))
        self.conv3 = SpatialSpectralConvModule(32, kernel_size=(1,1), strides=(1,1))
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(d_model, activation="relu", kernel_regularizer=l2(1e-4))
    def call(self, x):
        x = self.preprocess(x)
        x = self.attention(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

# === Preprocessing Layer (변경 없음) ===
class PreprocessLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        x = tf.where(tf.math.is_nan(inputs), tf.zeros_like(inputs), inputs)
        x = tf.where(tf.math.is_inf(x), tf.zeros_like(x), x)
        axis = list(range(1, inputs.shape.rank))
        min_val = tf.reduce_min(x, axis=axis, keepdims=True)
        max_val = tf.reduce_max(x, axis=axis, keepdims=True)
        range_val = max_val - min_val + 1e-8
        return (x - min_val) / range_val

# === 최종 모델: TimeDistributed + Transformer ===
def build_model(input_shape=(2, 2, 4, 9, 9, 1), n_layers=2, n_heads=4, d_ff=512, p_drop=0.3, d_model=64):
    """
    모델 입력: (T, 2, 4, 9, 9, 1)
     - T = 2: 8초 샘플을 2 타임스텝으로 나눔 (각 타임스텝은 4초)
     - 두 번째 차원: 2개의 모달리티 (DE, PSD)
     - 그 다음 차원: 4 (각 타임슬라이스의 4프레임)
     - 그 후 9x9의 2D 매핑 이미지와 마지막 채널 1
    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    # TimeDistributed를 통해 각 8초 샘플 내의 각 타임스텝마다 single segment 모델 적용
    single_seg_model = build_single_segment_model(d_model)
    features_seq = tf.keras.layers.TimeDistributed(single_seg_model)(inputs)
    
    x = features_seq
    for _ in range(n_layers):
        x = TransformerEncoderLayer(d_model, n_heads, d_ff, dropout_rate=p_drop)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(4, activation="softmax", kernel_regularizer=l2(1e-4))(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model

# === GradualUnfreeze, compute_channel_importance, plot_channel_importance, inter_subject_cv_training_9x9 ===
# (이하 코드는 기존과 동일)

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
                                  pretrain_lr=3e-4, fine_tune_lr=3e-4, unfreeze_lr=1e-5):
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
        model = build_model(input_shape=(2, 2, 4, 9, 9, 1), n_layers=2, n_heads=4, d_ff=512, p_drop=0.3, d_model=64)
        model.compile(optimizer=Adam(learning_rate=pretrain_lr),
                      loss="categorical_crossentropy", metrics=["accuracy"])
        model.summary()
        pretrain_callbacks = [EarlyStopping(monitor='val_accuracy', patience=20, min_delta=0.001, restore_best_weights=True)]
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

# -------------------------------------------------------------------------
# 메인 실행 부분
# -------------------------------------------------------------------------
base_dirs = [
    "/home/bcml1/2025_EMOTION/SEED_IV/eeg_feature_smooth/1_npy_sample",
    "/home/bcml1/2025_EMOTION/SEED_IV/eeg_feature_smooth/2_npy_sample",
    "/home/bcml1/2025_EMOTION/SEED_IV/eeg_feature_smooth/3_npy_sample"
]
RESULT_DIR = "/home/bcml1/sigenv/_7주차_eeg/mj_Diffu_1"

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
