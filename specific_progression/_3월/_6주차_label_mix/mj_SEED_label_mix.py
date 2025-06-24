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
# 라벨 무작위화 버전의 데이터 로드 함수
# -------------------------------------------------------------------------
def load_seediv_data_random_label(base_dirs, de_keys=["de_movingAve"], psd_keys=["psd_movingAve"]):
    data_de = {}
    data_psd = {}
    for base_dir in base_dirs:
        file_list = glob.glob(os.path.join(base_dir, "*.npy"))
        for file in file_list:
            filename = os.path.basename(file)
            parts = filename.replace('.npy','').split('_')
            if len(parts) < 6:
                continue
            subject, trial = parts[0], parts[3]
            key_name = parts[4] + "_" + parts[5]
            arr = np.load(file)[..., 1:]  # delta(0번째 밴드) 제거
            if key_name in de_keys:
                data_de[(subject, trial)] = arr
            elif key_name in psd_keys:
                data_psd[(subject, trial)] = arr

    common_ids = set(data_de.keys()).intersection(set(data_psd.keys()))
    de_list, psd_list, label_list, subject_list = [], [], [], []
    for sid in sorted(common_ids):
        subj, trial = sid
        arr_de = data_de[sid]
        arr_psd = data_psd[sid]
        # --- 라벨을 0~3 범위 무작위로 생성 ---
        label_val = np.random.randint(0, 4)
        de_list.append(arr_de)
        psd_list.append(arr_psd)
        label_list.append(label_val)
        subject_list.append(subj)
    return de_list, psd_list, label_list, subject_list

# -------------------------------------------------------------------------
# (9×9) 채널 위치 매핑 딕셔너리
# -------------------------------------------------------------------------
channel_positions_9x9 = {
    0: (0,4),   1: (0,5),   2: (0,6),
    3: (1,3),   4: (1,4),
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
    57: (8,0),  58: (8,1),  59: (8,2),  60: (8,3),  61: (8,4),
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

def process_slice_9x9(slice_, channel_positions):
    # slice_: (62,1,4)
    slice_2d = np.squeeze(slice_, axis=1)  # (62,4)
    mapped_9x9 = map_channels_to_2d_9x9(slice_2d, channel_positions)  # (9,9,4)
    # (4,9,9)로 transpose 후 채널 추가
    img = np.transpose(mapped_9x9, (2,0,1))  # (4,9,9)
    img = np.expand_dims(img, axis=-1)      # (4,9,9,1)
    return img

def process_trial_9x9(de_trial, psd_trial, channel_positions):
    slices_de = split_trial_to_time_slices(de_trial)   # (T,62,1,4)
    slices_psd = split_trial_to_time_slices(psd_trial) # (T,62,1,4)
    processed_slices = []
    T = slices_de.shape[0]
    for t in range(T):
        d_img = process_slice_9x9(slices_de[t], channel_positions)  # (4,9,9,1)
        p_img = process_slice_9x9(slices_psd[t], channel_positions) # (4,9,9,1)
        # DE/PSD 두 모달리티 합치기: (2,4,9,9,1)
        sample = np.stack([d_img, p_img], axis=0)
        processed_slices.append(sample)
    return np.stack(processed_slices, axis=0)  # (T,2,4,9,9,1)

# === 아래는 CNN+Transformer 구현(예시) 부분 ===
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

class PreprocessLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        x = tf.where(tf.math.is_nan(inputs), tf.zeros_like(inputs), inputs)
        x = tf.where(tf.math.is_inf(x), tf.zeros_like(x), x)
        axis = list(range(1, inputs.shape.rank))
        min_val = tf.reduce_min(x, axis=axis, keepdims=True)
        max_val = tf.reduce_max(x, axis=axis, keepdims=True)
        range_val = max_val - min_val + 1e-8
        return (x - min_val) / range_val

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

def build_model(input_shape, n_layers=2, n_heads=4, d_ff=512, p_drop=0.3, d_model=64):
    """
    input_shape: (T, 2,4,9,9,1)
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

# -------------------- 예시용 Callback (Gradual Unfreeze 등) --------------------
class GradualUnfreeze(Callback):
    def __init__(self, unfreeze_epoch, layers_to_unfreeze, unfreeze_lr=3e-4):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch
        self.layers_to_unfreeze = layers_to_unfreeze
        self.unfreeze_lr = unfreeze_lr

    def on_epoch_begin(self, epoch, logs=None):
        if epoch + 1 == self.unfreeze_epoch:
            print(f"\nUnfreezing layers at epoch {epoch+1}...")
            for layer in self.model.layers:
                if any(isinstance(layer, lt) for lt in self.layers_to_unfreeze):
                    layer.trainable = True
                    print(f"Layer {layer.name} unfreezed.")
            self.model.optimizer.learning_rate.assign(self.unfreeze_lr)
            print(f"Learning rate set to {self.unfreeze_lr} after unfreezing.")
            
# -------------------- 예시용 create_seediv_cnn_model (간단버전) --------------------
def create_seediv_cnn_model(input_shape=(62,1,4), num_classes=4, pretrain_lr=3e-4):
    # 아주 간단한 모델 예시
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=Adam(pretrain_lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -------------------------------------------------------------------------
# inter_subject_cv_training_9x9 함수 (라벨 무작위 매핑 부분 추가)
# -------------------------------------------------------------------------
def inter_subject_cv_training_9x9(base_dirs, result_dir,
                                  epochs=150, batch_size=16,
                                  target_subjects=None,
                                  pretrain_lr=3e-4, fine_tune_lr=3e-4, unfreeze_lr=3e-4):
    os.makedirs(result_dir, exist_ok=True)
    overall_folder = os.path.join(result_dir, "overall")
    os.makedirs(overall_folder, exist_ok=True)

    # ---------------------------------------------------------------------
    # (1) 라벨 무작위화 버전 데이터 로드
    # ---------------------------------------------------------------------
    de_trials, psd_trials, label_list, subject_list = load_seediv_data_random_label(base_dirs)

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
        # 1) Test set 분리
        X_de_test_trials = subject_data[test_subj]["de"]
        X_psd_test_trials = subject_data[test_subj]["psd"]
        y_test_trials = np.array(subject_data[test_subj]["labels"])
        y_cat_test = to_categorical(y_test_trials, num_classes=4)

        # 2) Training set (다른 subject)
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

        # 3) Train/Validation split
        num_train_trials = len(X_de_train_trials)
        indices = np.arange(num_train_trials)
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y_train_list)
        X_de_train_split = [X_de_train_trials[i] for i in train_idx]
        X_de_val_split = [X_de_train_trials[i] for i in val_idx]
        X_psd_train_split = [X_psd_train_trials[i] for i in train_idx]
        X_psd_val_split = [X_psd_train_trials[i] for i in val_idx]
        y_train_split = y_cat_train[train_idx]
        y_val_split = y_cat_train[val_idx]

        # 4) 시간 슬라이스 + 라벨 복제
        X_de_train = split_trials_list_to_time_slices(X_de_train_split)
        X_psd_train = split_trials_list_to_time_slices(X_psd_train_split)
        y_train = replicate_labels_for_trials(X_de_train_split, y_train_split)

        X_de_val = split_trials_list_to_time_slices(X_de_val_split)
        X_psd_val = split_trials_list_to_time_slices(X_psd_val_split)
        y_val = replicate_labels_for_trials(X_de_val_split, y_val_split)

        X_de_test = split_trials_list_to_time_slices(X_de_test_trials)
        X_psd_test = split_trials_list_to_time_slices(X_psd_test_trials)
        y_test = replicate_labels_for_trials(X_de_test_trials, y_cat_test)

        # 5) Dataset
        train_dataset = tf.data.Dataset.from_tensor_slices(((X_de_train, X_psd_train), y_train)).shuffle(1000).batch(batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices(((X_de_val, X_psd_val), y_val)).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices(((X_de_test, X_psd_test), y_test)).batch(batch_size)

        # 6) 모델 생성 + Pre-training
        model = build_model(input_shape=(62, 1, 4), n_layers=2, n_heads=4, d_ff=512, p_drop=0.3, d_model=64)
        # 라벨이 one-hot이 아니라면 sparse_categorical_crossentropy, one-hot이면 categorical_crossentropy
        # 여기서는 y_train, y_val, y_test가 one-hot 형태이므로 categorical_crossentropy
        model.compile(optimizer=Adam(learning_rate=pretrain_lr), loss="categorical_crossentropy", metrics=["accuracy"])
        model.summary()

        pretrain_epochs = int(epochs * 0.6)
        finetune_epochs = epochs - pretrain_epochs
        pretrain_callbacks = [EarlyStopping(monitor='val_accuracy', patience=20, min_delta=0.001, restore_best_weights=True)]
        
        print(f"Test subject {test_subj}: Starting pre-training phase for {pretrain_epochs} epochs...")
        history_pretrain = model.fit(train_dataset, epochs=pretrain_epochs, validation_data=val_dataset,
                                     callbacks=pretrain_callbacks, verbose=1)

        # 결과 저장 폴더
        subj_folder = os.path.join(result_dir, f"s{test_subj.zfill(2)}")
        os.makedirs(subj_folder, exist_ok=True)

        # Pre-training 곡선
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

        # 7) Fine-tuning (feature extractor 동결 후 점진적 해제)
        for layer in model.layers:
            if isinstance(layer, (tf.keras.layers.Conv2D,
                                  tf.keras.layers.BatchNormalization,
                                  tf.keras.layers.MaxPooling2D,
                                  tf.keras.layers.GlobalAveragePooling2D)):
                layer.trainable = False
            else:
                layer.trainable = True

        model.compile(optimizer=Adam(learning_rate=fine_tune_lr), loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"Test subject {test_subj}: Starting fine-tuning phase for {finetune_epochs} epochs...")
        gradual_unfreeze_cb = GradualUnfreeze(unfreeze_epoch=5, layers_to_unfreeze=[tf.keras.layers.Conv2D], unfreeze_lr=unfreeze_lr)
        history_finetune = model.fit(train_dataset, epochs=finetune_epochs, validation_data=val_dataset,
                                     callbacks=[gradual_unfreeze_cb], verbose=1)

        # Fine-tuning 곡선
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

        # 8) 테스트
        test_loss, test_acc = model.evaluate(test_dataset, verbose=0)
        print(f"Test subject {test_subj}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        overall_acc[test_subj] = test_acc

        # 혼동행렬
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

        report_path = os.path.join(subj_folder, "classification.txt")
        with open(report_path, "w") as f:
            f.write(report)

        model_save_path = os.path.join(subj_folder, "model_eeg_9x9.keras")
        model.save(model_save_path)

    overall_avg_acc = np.mean(list(overall_acc.values()))
    overall_report_path = os.path.join(overall_folder, "overall_classification.txt")
    with open(overall_report_path, "w") as f:
        f.write("Overall LOSO Test Accuracy: {:.4f}\n\n".format(overall_avg_acc))
        for subj in sorted(overall_reports.keys(), key=lambda x: int(x) if x.isdigit() else x):
            f.write(f"Test Subject {subj}:\n")
            f.write(overall_reports[subj])
            f.write("\n\n")

    print(f"Overall results saved to {overall_report_path}")

# -------------------------------
# 실제 실행 예시
# -------------------------------
if __name__ == "__main__":
    base_dirs = [
        "/home/bcml1/2025_EMOTION/SEED_IV/eeg_feature_smooth/1_npy_sample",
        "/home/bcml1/2025_EMOTION/SEED_IV/eeg_feature_smooth/2_npy_sample",
        "/home/bcml1/2025_EMOTION/SEED_IV/eeg_feature_smooth/3_npy_sample"
    ]
    RESULT_DIR = "/home/bcml1/sigenv/_6주차_eeg_test_ft/mj_SEED_2Dmap_1"

    inter_subject_cv_training_9x9(
        base_dirs=base_dirs,
        result_dir=RESULT_DIR,
        epochs=100,
        batch_size=16,
        target_subjects=[str(i) for i in range(1, 16)],
        pretrain_lr=3e-4,
        fine_tune_lr=1e-5,
        unfreeze_lr=1e-5
    )
