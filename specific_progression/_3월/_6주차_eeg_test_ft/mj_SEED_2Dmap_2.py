#4초의 DE feature + 4초의 PSD => 8초의 DE + PSD 가 입력으로 들어감.
# 이 구조는 4초 DE와 4초 PSD를 별도의 채널로 다루므로, 시간 차원에서는 여전히 4초 분량의 데이터를 입력으로 사용하지만, 두 종류의 특성(모달리티)을 결합하여 하나의 샘플로 만듭니다.
# 따라서 “8초의 DE + PSD”라는 표현은 두 모달리티가 결합된 것을 의미하지만, 실제 시간 길이는 4초입니다.
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

limit_gpu_memory(5000)

# -------------------------------------------------------------------------
# Helper 함수: trial 데이터를 시간 슬라이스로 분리 및 라벨 복제
# -------------------------------------------------------------------------
def split_trial_to_time_slices(trial):
    """
    trial.shape: (62, T, bands)
      - 62: 채널 수
      - T: time length (초 단위 or 샘플 단위)
      - bands: 주파수 밴드 수 (예: 4 or 5)
    """
    channels, T, bands = trial.shape
    slices = [trial[:, t:t+1, :] for t in range(T)]  # 각 time step별로 잘라냄
    return np.array(slices)  # shape: (T, 62, 1, bands)

def split_trials_list_to_time_slices(trials_list):
    slices_list = [split_trial_to_time_slices(trial) for trial in trials_list]
    return np.concatenate(slices_list, axis=0) if slices_list else None

def replicate_labels_for_trials(trials_list, labels):
    """
    trial.shape[1] (T)에 맞춰 label을 반복
    예: trial이 길이 T=4라면, 동일 라벨을 4번 복제
    """
    replicated = [np.repeat(label[np.newaxis, :], trial.shape[1], axis=0) 
                  for trial, label in zip(trials_list, labels)]
    return np.concatenate(replicated, axis=0) if replicated else None

# -------------------------------------------------------------------------
# SEED-IV 데이터 로드 함수 (DE/PSD)
# -------------------------------------------------------------------------
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
            arr = np.load(file)[..., 1:]  # delta(0번째 밴드) 제거
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

# -------------------------------------------------------------------------
# (9×9) 채널 위치 매핑 딕셔너리
# 0-based 인덱스(0~61) → (row,col)
# -------------------------------------------------------------------------
channel_positions_9x9 = {
    0: (0,3),   # FP1
    1: (0,4),   # FPZ
    2: (0,5),   # FP2
    3: (1,3),   # AF3
    4: (1,5),   # AF4
    5: (2,0),   # F7
    6: (2,1),   # F5
    7: (2,2),   # F3
    8: (2,3),   # F1
    9: (2,4),   # FZ
    10: (2,5),  # F2
    11: (2,6),  # F4
    12: (2,7),  # F6
    13: (2,8),  # F8
    14: (3,0),  # FT7
    15: (3,1),  # FC5
    16: (3,2),  # FC3
    17: (3,3),  # FC1
    18: (3,4),  # FCZ
    19: (3,5),  # FC2
    20: (3,6),  # FC4
    21: (3,7),  # FC6
    22: (3,8),  # FT8
    23: (4,0),  # T7
    24: (4,1),  # C5
    25: (4,2),  # C3
    26: (4,3),  # C1
    27: (4,4),  # CZ
    28: (4,5),  # C2
    29: (4,6),  # C4
    30: (4,7),  # C6
    31: (4,8),  # T8
    32: (5,0),  # TP7
    33: (5,1),  # CP5
    34: (5,2),  # CP3
    35: (5,3),  # CP1
    36: (5,4),  # CPZ
    37: (5,5),  # CP2
    38: (5,6),  # CP4
    39: (5,7),  # CP6
    40: (5,8),  # TP8
    41: (6,0),  # P7
    42: (6,1),  # P5
    43: (6,2),  # P3
    44: (6,3),  # P1
    45: (6,4),  # PZ
    46: (6,5),  # P2
    47: (6,6),  # P4
    48: (6,7),  # P6
    49: (6,8),  # P8
    50: (7,1),  # PO7
    51: (7,2),  # PO5
    52: (7,3),  # PO3
    53: (7,4),  # POZ
    54: (7,5),  # PO4
    55: (7,6),  # PO6
    56: (7,7),  # PO8
    57: (8,2),  # CB1
    58: (8,3),  # O1
    59: (8,4),  # OZ
    60: (8,5),  # O2
    61: (8,6),  # CB2
}

# -------------------------------------------------------------------------
# (62,4) → (9,9,4) 매핑 함수
# -------------------------------------------------------------------------
def map_channels_to_2d_9x9(segment, channel_positions):
    """
    segment: shape (62, 4)
    반환: (9,9,4)
    """
    num_channels, num_bands = segment.shape
    mapped = np.zeros((9, 9, num_bands), dtype=np.float32)
    for ch_idx in range(num_channels):
        if ch_idx not in channel_positions:
            continue
        row, col = channel_positions[ch_idx]
        for band_idx in range(num_bands):
            mapped[row, col, band_idx] = segment[ch_idx, band_idx]
    return mapped

# -------------------------------------------------------------------------
# (62,1,4) → (62,4) → (9,9,4) → (4,9,9,1)
# -------------------------------------------------------------------------
def process_slice_9x9(slice_, channel_positions):
    # slice_: (62,1,4)
    slice_2d = np.squeeze(slice_, axis=1)  # (62,4)
    mapped_9x9 = map_channels_to_2d_9x9(slice_2d, channel_positions)  # (9,9,4)
    # (4,9,9)로 transpose 후 채널 추가
    img = np.transpose(mapped_9x9, (2,0,1))  # (4,9,9)
    img = np.expand_dims(img, axis=-1)      # (4,9,9,1)
    return img

# -------------------------------------------------------------------------
# 각 trial DE/PSD 데이터를 슬라이스별로 처리하여, 
# 각 슬라이스를 (1, 2, 4, 9, 9, 1) 모양의 개별 샘플로 반환 (T=1, 2, 4,9,9,1)
# -------------------------------------------------------------------------
def process_trial_9x9(de_trial, psd_trial, channel_positions):
    # 슬라이스별 분리: (T, 62, 1, 4)
    slices_de = split_trial_to_time_slices(de_trial)    # (T, 62, 1, bands)
    slices_psd = split_trial_to_time_slices(psd_trial)  # (T, 62, 1, bands)
    sample_list = []
    T = slices_de.shape[0]

    # DE와 PSD 데이터를 한쌍씩 묶어서 처리
    for t in range(T):
        combined_slice = np.concatenate([slices_de[t], slices_psd[t]], axis=1)  # (62, 2, 4)
        imgs = []
        for feature_idx in range(2):  # DE(0), PSD(1)
            img = process_slice_9x9(combined_slice[:, feature_idx:feature_idx+1, :], channel_positions)  # (4,9,9,1)
            imgs.append(img)
        sample = np.stack(imgs, axis=0)  # (2, 4, 9, 9, 1)
        sample = np.expand_dims(sample, axis=0)  # (1, 2, 4, 9, 9, 1)
        sample_list.append(sample)

    return np.concatenate(sample_list, axis=0)  # (T, 2, 4, 9, 9, 1)

# def process_trial_9x9(de_trial, psd_trial, channel_positions):
#     slices_de = split_trial_to_time_slices(de_trial)
#     slices_psd = split_trial_to_time_slices(psd_trial)
#     sample_list = []
#     T = slices_de.shape[0]

#     for t in range(T):
#         d_img = process_slice_9x9(slices_de[t], channel_positions)
#         p_img = process_slice_9x9(slices_psd[t], channel_positions)
#         sample = np.stack([d_img, p_img], axis=0)  # (2,4,9,9,1)
#         sample = np.expand_dims(sample, axis=0)    # (1,2,4,9,9,1)
#         sample_list.append(sample)

#     return np.concatenate(sample_list, axis=0)  # (T,2,4,9,9,1)

# # -------------------------------------------------------------------------
# # 데이터 로드
# # -------------------------------------------------------------------------
# base_dirs = [
#     "/home/bcml1/2025_EMOTION/SEED_IV/eeg_feature_smooth/1_npy_sample",
#     "/home/bcml1/2025_EMOTION/SEED_IV/eeg_feature_smooth/2_npy_sample",
#     "/home/bcml1/2025_EMOTION/SEED_IV/eeg_feature_smooth/3_npy_sample"
# ]
# de_list, psd_list, label_list, subject_list = load_seediv_data(base_dirs)

# X_list, Y_list, S_list = [], [], []
# for de_trial, psd_trial, label, subj in zip(de_list, psd_list, label_list, subject_list):
#     # de_trial, psd_trial shape: (62, T, 4)  (T는 time length)
#     sample_9x9 = process_trial_9x9(de_trial, psd_trial, channel_positions_9x9)
#     X_list.append(sample_9x9)  # (T,2,4,9,9,1)
#     Y_list.append(label)
#     S_list.append(subj)

# X = np.array(X_list)  # (num_trials, T, 2,4,9,9,1)
# Y = np.array(Y_list)
# S = np.array(S_list)
# print(f"Loaded {X.shape[0]} trials. Each trial sample shape: {X.shape[1:]}")

# === CNN+Transformer 관련 Layer들 ===

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

# === Preprocessing Layer ===
class PreprocessLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        x = tf.where(tf.math.is_nan(inputs), tf.zeros_like(inputs), inputs)
        x = tf.where(tf.math.is_inf(x), tf.zeros_like(x), x)
        axis = list(range(1, inputs.shape.rank))
        min_val = tf.reduce_min(x, axis=axis, keepdims=True)
        max_val = tf.reduce_max(x, axis=axis, keepdims=True)
        range_val = max_val - min_val + 1e-8
        return (x - min_val) / range_val

# === Single segment model ===
def build_single_segment_model(d_model=64):
    input_seg = tf.keras.layers.Input(shape=(2, 4, 9, 9, 1))  # (2,4,9,9,1)
    de_input = tf.keras.layers.Lambda(lambda x: x[:,0,...])(input_seg)  # (4,9,9,1)
    psd_input = tf.keras.layers.Lambda(lambda x: x[:,1,...])(input_seg) # (4,9,9,1)
    cnn_de = CNNBranchLayer(d_model)
    cnn_psd = CNNBranchLayer(d_model)
    feature_de = cnn_de(de_input)      # (d_model,)
    feature_psd = cnn_psd(psd_input)   # (d_model,)
    combined_features = tf.keras.layers.Concatenate()([feature_de, feature_psd])  # (2*d_model,)
    combined_features = tf.keras.layers.Dense(d_model, activation="relu", kernel_regularizer=l2(1e-4))(combined_features)
    model = tf.keras.models.Model(inputs=input_seg, outputs=combined_features)
    return model

# === 최종 모델: TimeDistributed + Transformer ===
def build_model(input_shape=(1, 2, 4, 9, 9, 1), n_layers=2, n_heads=4, d_ff=512, p_drop=0.3, d_model=64):
    """
    input_shape: (T=1, 2,4,9,9,1)
    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    single_seg_model = build_single_segment_model(d_model)
    # (batch, T, d_model)
    features_seq = tf.keras.layers.TimeDistributed(single_seg_model)(inputs)
    x = features_seq
    for _ in range(n_layers):
        x = TransformerEncoderLayer(d_model, n_heads, d_ff, dropout_rate=p_drop)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(4, activation="softmax", kernel_regularizer=l2(1e-4))(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model

# -----------------------------------------------------------------------------
# Custom Callback for Gradual Unfreezing (unfreeze_lr를 인자로 받음)
# -----------------------------------------------------------------------------
# class GradualUnfreeze(Callback):
#     def __init__(self, unfreeze_epoch, layers_to_unfreeze, unfreeze_lr=3e-4):
#         """
#         unfreeze_epoch: fine-tuning 단계에서 unfreeze를 시작할 epoch 번호 (예: 5)
#         layers_to_unfreeze: unfreeze할 layer 클래스 목록 (예: [tf.keras.layers.Conv2D])
#         unfreeze_lr: unfreeze 후 적용할 새로운 학습률
#         """
#         super().__init__()
#         self.unfreeze_epoch = unfreeze_epoch
#         self.layers_to_unfreeze = layers_to_unfreeze
#         self.unfreeze_lr = unfreeze_lr

#     def on_epoch_begin(self, epoch, logs=None):
#         if epoch + 1 == self.unfreeze_epoch:
#             print(f"\nUnfreezing layers at epoch {epoch+1}...")
#             for layer in self.model.layers:
#                 if any(isinstance(layer, lt) for lt in self.layers_to_unfreeze):
#                     layer.trainable = True
#                     print(f"Layer {layer.name} unfreezed.")
#             self.model.optimizer.learning_rate.assign(self.unfreeze_lr)
#             print(f"Learning rate set to {self.unfreeze_lr} after unfreezing.")
            
class GradualUnfreeze(Callback):
    def __init__(self, unfreeze_schedule, unfreeze_lr=3e-4):
        """
        unfreeze_schedule: {epoch: [layer_name1, layer_name2, ...]} 형태의 dict
        """
        super().__init__()
        self.unfreeze_schedule = unfreeze_schedule
        self.unfreeze_lr = unfreeze_lr

    def on_epoch_begin(self, epoch, logs=None):
        """ 특정 epoch가 되면, unfreeze_schedule에 정의된 레이어들을 unfreeze """
        if epoch in self.unfreeze_schedule:
            print(f"\nUnfreezing layers at epoch {epoch}...")
            for layer_name in self.unfreeze_schedule[epoch]:
                for layer in self.model.layers:
                    if layer.name == layer_name:
                        layer.trainable = True
                        print(f"Layer {layer.name} unfreezed.")

            # 학습률 업데이트
            self.model.optimizer.learning_rate.assign(self.unfreeze_lr)
            print(f"Learning rate set to {self.unfreeze_lr} after unfreezing.")


def compute_channel_importance(model, X_test_final, y_test, baseline_acc, 
                               channel_positions_9x9, batch_size=16):
    """
    - model: 이미 fine-tuning까지 끝난 학습된 모델
    - X_test_final: shape (N, 1, 2, 4, 9, 9, 1) 형태 (LOSO 최종 테스트 세트)
    - y_test: shape (N,) (one-hot 이전의 라벨)
    - baseline_acc: 마스킹 없이 얻은 기본 Test Accuracy
    - channel_positions_9x9: {채널인덱스: (row, col)} 형태의 딕셔너리
    """
    num_channels = 62
    channel_importance = np.zeros(num_channels, dtype=np.float32)
    
    # 채널별로 순회하며 해당 채널을 0으로 마스킹한 뒤 모델 성능 측정
    for ch in range(num_channels):
        X_test_mod = np.copy(X_test_final)

        # 9x9 매핑 후 (4,9,9,1)의 특정 채널(row,col)을 0으로 설정
        if ch in channel_positions_9x9:
            row, col = channel_positions_9x9[ch]
            X_test_mod[:, :, :, :, row, col, :] = 0.0
        
        # TensorFlow Dataset 구성
        test_dataset_mod = tf.data.Dataset.from_tensor_slices(
            (X_test_mod, to_categorical(y_test, num_classes=4))
        ).batch(batch_size)
        
        # 모델 평가
        _, masked_acc = model.evaluate(test_dataset_mod, verbose=0)
        
        # 중요도 계산
        channel_importance[ch] = baseline_acc - masked_acc
    
    # 9×9 매트릭스로 매핑
    importance_map = np.zeros((9, 9), dtype=np.float32)
    for ch in range(num_channels):
        if ch in channel_positions_9x9:
            r, c = channel_positions_9x9[ch]
            importance_map[r, c] = channel_importance[ch]
    
    return importance_map, channel_importance

def plot_channel_importance(importance_map, save_path="channel_importance.png"):
    """
    importance_map: shape (9,9)
    """
    plt.figure(figsize=(6,5))
    sns.heatmap(importance_map, annot=True, fmt=".3f", cmap="Reds")
    plt.title("Channel Importance (Permutation-based)")
    plt.xlabel("X-axis (columns)")
    plt.ylabel("Y-axis (rows)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Channel importance heatmap saved to {save_path}")
    
# -------------------------------------------------------------------------
# LOSO 학습 함수 (pre-training, fine-tuning, unfreeze 적용 + test subject 데이터 split)
# -------------------------------------------------------------------------
def inter_subject_cv_training_9x9(base_dirs, result_dir,
                                  epochs=150, batch_size=16,
                                  target_subjects=None,
                                  pretrain_lr=3e-4, fine_tune_lr=3e-4, unfreeze_lr=3e-4):
    os.makedirs(result_dir, exist_ok=True)
    overall_folder = os.path.join(result_dir, "overall")
    os.makedirs(overall_folder, exist_ok=True)

    # 1) 데이터 로드
    de_trials, psd_trials, label_list, subject_list = load_seediv_data(base_dirs)

    # 2) subject별 그룹화
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
        # 1) Test subject 데이터 분리
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

        # 2) 나머지 subject 데이터 (train)
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

        # 3) Trial 단위 Train/Validation split
        num_train_trials = len(X_de_train_trials)
        indices = np.arange(num_train_trials)
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y_train_list)
        X_de_train_split = [X_de_train_trials[i] for i in train_idx]
        X_psd_train_split = [X_psd_train_trials[i] for i in train_idx]
        X_de_val_split = [X_de_train_trials[i] for i in val_idx]
        X_psd_val_split = [X_psd_train_trials[i] for i in val_idx]
        y_train_split = y_cat_train[train_idx]
        y_val_split = y_cat_train[val_idx]

        # 4) 각 trial별로 process_trial_9x9 적용 → 최종 입력 shape: (N, 2, 4, 9, 9, 1)
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

        # 5) Dataset 구성
        # Dataset 구성 시 시간 차원 추가
        X_de_train = np.expand_dims(X_de_train, axis=1)  # (N, 1, 2, 4, 9, 9, 1)
        X_de_val = np.expand_dims(X_de_val, axis=1)      # (N, 1, 2, 4, 9, 9, 1)
        X_de_test_ft = np.expand_dims(X_de_test_ft, axis=1)  # (N, 1, 2, 4, 9, 9, 1)
        X_de_test_final = np.expand_dims(X_de_test_final, axis=1)  # (N, 1, 2, 4, 9, 9, 1)

        # X_de_train.shape       # (N, 1, 2, 4, 9, 9, 1)
        # X_de_val.shape         # (M, 1, 2, 4, 9, 9, 1)
        # X_de_test_ft.shape     # (K, 1, 2, 4, 9, 9, 1)
        # X_de_test_final.shape  # (L, 1, 2, 4, 9, 9, 1)

        print(f"X_de_train.shape: {X_de_train.shape}, X_de_val.shape: {X_de_val.shape}")
        print(f"X_de_test_ft.shape: {X_de_test_ft.shape}, X_de_test_final.shape: {X_de_test_final.shape}")
        
        train_dataset = tf.data.Dataset.from_tensor_slices((X_de_train, y_train)).shuffle(1000).batch(batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_de_val, y_val)).batch(batch_size)
        ft_dataset = tf.data.Dataset.from_tensor_slices((X_de_test_ft, y_test_ft)).shuffle(1000).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((X_de_test_final, y_test_final)).batch(batch_size)        
        
        # ---------------------------
        # 6) 모델 생성 및 Pre-training (pretrain_lr 적용) - 입력 shape: (62,1,4)
        # ---------------------------
        model = build_model(input_shape=(1, 2, 4, 9, 9, 1), n_layers=2, n_heads=4, d_ff=512, p_drop=0.3, d_model=64)
        # model.compile(optimizer=Adam(learning_rate=pretrain_lr),
        #               loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.compile(optimizer=Adam(learning_rate=pretrain_lr),
                      loss="categorical_crossentropy", metrics=["accuracy"])
        model.summary()

        # Pre-training Phase: 사용 데이터는 training subjects의 train_dataset 및 val_dataset
        # pretrain_epochs = int(epochs * 0.6)
        pretrain_epochs = 150
        # finetune_epochs = epochs - pretrain_epochs
        finetune_epochs = 20
        
        pretrain_callbacks = [EarlyStopping(monitor='val_accuracy', patience=20, min_delta=0.001, restore_best_weights=True)]
        
        print(f"Test subject {test_subj}: Starting pre-training phase for {pretrain_epochs} epochs...")
        history_pretrain = model.fit(train_dataset, epochs=pretrain_epochs, validation_data=val_dataset,
                                     callbacks=pretrain_callbacks, verbose=1)

        # 저장: Pre-training 학습 곡선
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

        # ---------------------------
        # 7) Fine-tuning Phase: 테스트 subject의 fine-tuning 데이터(ft_dataset)로 미세 조정
        # ---------------------------
        # 처음에는 분류기(Dense)만 학습하도록 feature extractor 동결
        for layer in model.layers:
            if isinstance(layer, (tf.keras.layers.Conv2D,
                                  tf.keras.layers.BatchNormalization,
                                  tf.keras.layers.MaxPooling2D,
                                  tf.keras.layers.GlobalAveragePooling2D)):
                layer.trainable = False
            else:
                layer.trainable = True

        # Fine-tuning 단계의 학습률 적용
        model.compile(optimizer=Adam(learning_rate=fine_tune_lr), loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"Test subject {test_subj}: Starting fine-tuning phase for {finetune_epochs} epochs...")
        # Gradual Unfreeze Callback: fine-tuning 5번째 epoch에 Conv2D 레이어 일부 동결 해제
        #개선된 unfreeze schedule
        unfreeze_schedule = {
            3: ['cnn_branch_layer/conv3'],
            6: ['cnn_branch_layer/conv2'],
            9: ['cnn_branch_layer/conv1', 'transformer_encoder_layer'],
        }
        gradual_unfreeze_cb = GradualUnfreeze(unfreeze_schedule, unfreeze_lr=1e-5)
        scheduler = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: 1e-5 * np.cos(epoch / finetune_epochs * np.pi / 2)
        )
        history_finetune = model.fit(
            ft_dataset, epochs=finetune_epochs, 
            validation_data=val_dataset,
            callbacks=[gradual_unfreeze_cb, scheduler],
            verbose=1
        )
        # gradual_unfreeze_cb = GradualUnfreeze(unfreeze_epoch=5, layers_to_unfreeze=[tf.keras.layers.Conv2D], unfreeze_lr=unfreeze_lr)
        # history_finetune = model.fit(ft_dataset, epochs=finetune_epochs, validation_data=val_dataset,
        #                              callbacks=[gradual_unfreeze_cb], verbose=1)
        
        # history_finetune = model.fit(ft_dataset, epochs=finetune_epochs, validation_data=val_dataset,
        #                      callbacks=[],  # 여기서 gradual_unfreeze_cb를 제거하고 빈 리스트 또는 다른 callback을 사용하면 됩니다.
        #                      verbose=1)

        # 저장: Fine-tuning 학습 곡선
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

        # ---------------------------
        # 6) 최종 테스트 평가 (테스트 subject의 80% 데이터)
        # ---------------------------
        test_loss, test_acc = model.evaluate(test_dataset, verbose=0)
        print(f"Test subject {test_subj}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        overall_acc[test_subj] = test_acc

        # 채널 중요도 계산 (여기서 X_de_test, X_psd_test, y_test는 최종 테스트 데이터)
        # one-hot 인코딩된 y_test_final을 정수 레이블로 변환
        y_test_int = np.argmax(y_test_final, axis=-1)
        importance_map, channel_imp = compute_channel_importance(
            model=model,
            X_test_final=X_de_test_final,   # 이제 올바른 형태로 전달됨
            y_test=y_test_int,
            baseline_acc=test_acc,
            channel_positions_9x9=channel_positions_9x9,
            batch_size=batch_size
        )
        # Heatmap 저장
        imp_fig_path = os.path.join(subj_folder, "channel_importance.png")
        plot_channel_importance(importance_map, save_path=imp_fig_path)
        np.save(os.path.join(subj_folder, "channel_importance.npy"), channel_imp)
        
        # ---------------------------
        # 7) 혼동행렬/리포트
        # ---------------------------
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
RESULT_DIR = "/home/bcml1/sigenv/_6주차_eeg_test_ft/mj_SEED_2Dmap_2"

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