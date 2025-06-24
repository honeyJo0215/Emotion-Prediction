import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.layers import (Input, Dense, GlobalAveragePooling1D, Concatenate, Dropout, 
                                     BatchNormalization, Reshape)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

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

limit_gpu_memory(7000)

# =============================================================================
# 전역 하이퍼파라미터 설정
# =============================================================================
WINDOW_SIZE = 10       # 10초 sample
SLIDING_STEP = 1       # 1초 overlap
PPG_SEGMENT_LENGTH = 128  # 원래 파일이 (60,5,128)로 되어있음
NUM_CLASSES = 4
EPOCHS = 300
BATCH_SIZE = 128
DROPOUT_RATE = 0.5

input_shape = (1280, 5)
num_classes = 4  # 예: HVHA, HVLA, LVLA, LVHA

# =============================================================================
# placeholder: 전처리 함수 (사용자에 맞게 구현 필요)
# =============================================================================
def preprocess_ppg_trial(data, fs=128):
    # 예: 간단히 정규화만 수행 (실제 구현에 맞게 수정)
    return data / np.max(np.abs(data), axis=(1,2), keepdims=True)

# =============================================================================
# placeholder: 피험자별 정규화 함수 (사용자에 맞게 구현 필요)
# =============================================================================
def subject_wise_normalize(X):
    # 예: 각 subject 데이터의 평균과 표준편차로 정규화
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True)
    return (X - mean) / (std + 1e-6)

# =============================================================================
# SE Block: 채널 중요도 재조정 (Hu et al., 2018)
# =============================================================================
class SEBlock(layers.Layer):
    def __init__(self, reduction=4):
        super(SEBlock, self).__init__()
        self.reduction = reduction

    def build(self, input_shape):
        filters = int(input_shape[-1])
        self.global_avg_pool = GlobalAveragePooling1D()
        self.reshape = Reshape((1, filters))
        self.dense1 = Dense(filters // self.reduction, activation='relu',
                            kernel_initializer='he_normal', use_bias=False)
        self.dense2 = Dense(filters, activation='sigmoid',
                            kernel_initializer='he_normal', use_bias=False)
        super(SEBlock, self).build(input_shape)

    def call(self, inputs):
        se = self.global_avg_pool(inputs)
        se = self.reshape(se)
        se = self.dense1(se)
        se = self.dense2(se)
        return inputs * se

# =============================================================================
# Improved CNN Branch: SeparableConv1D, BatchNormalization, SE block 적용
# =============================================================================
class ImprovedCNNBranch(layers.Layer):
    def __init__(self):
        super(ImprovedCNNBranch, self).__init__()
        self.conv1 = layers.Conv1D(8, 7, strides=1, padding='same',
                                   activation=tf.nn.relu6,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=regularizers.l2(1e-6))
        self.bn1 = BatchNormalization()
        self.sepconv2 = layers.SeparableConv1D(16, 7, strides=4, padding='same',
                                               activation=tf.nn.relu6,
                                               depthwise_initializer='he_normal',
                                               pointwise_initializer='he_normal',
                                               depthwise_regularizer=regularizers.l2(1e-6),
                                               pointwise_regularizer=regularizers.l2(1e-6))
        self.bn2 = BatchNormalization()
        self.sepconv3 = layers.SeparableConv1D(32, 7, strides=2, padding='same',
                                               activation=tf.nn.relu6,
                                               depthwise_initializer='he_normal',
                                               pointwise_initializer='he_normal',
                                               depthwise_regularizer=regularizers.l2(1e-6),
                                               pointwise_regularizer=regularizers.l2(1e-6))
        self.bn3 = BatchNormalization()
        self.sepconv4 = layers.SeparableConv1D(64, 7, strides=4, padding='same',
                                               activation=tf.nn.relu6,
                                               depthwise_initializer='he_normal',
                                               pointwise_initializer='he_normal',
                                               depthwise_regularizer=regularizers.l2(1e-6),
                                               pointwise_regularizer=regularizers.l2(1e-6))
        self.bn4 = BatchNormalization()
        self.sepconv5 = layers.SeparableConv1D(128, 7, strides=2, padding='same',
                                               activation=tf.nn.relu6,
                                               depthwise_initializer='he_normal',
                                               pointwise_initializer='he_normal',
                                               depthwise_regularizer=regularizers.l2(1e-6),
                                               pointwise_regularizer=regularizers.l2(1e-6))
        self.bn5 = BatchNormalization()
        self.se = SEBlock(reduction=4)
        self.global_pool = GlobalAveragePooling1D()
        
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.sepconv2(x)
        x = self.bn2(x, training=training)
        x = self.sepconv3(x)
        x = self.bn3(x, training=training)
        x = self.sepconv4(x)
        x = self.bn4(x, training=training)
        x = self.sepconv5(x)
        x = self.bn5(x, training=training)
        x = self.se(x)
        x = self.global_pool(x)
        return x

# =============================================================================
# Grouped Channel Selection: 원본, smoothing, downsampling 분기
# =============================================================================
class GroupedChannelSelection(layers.Layer):
    def __init__(self, smoothing_scale=2, downsampling_factor=2):
        super(GroupedChannelSelection, self).__init__()
        self.smoothing_scale = smoothing_scale
        self.downsampling_factor = downsampling_factor

    def call(self, inputs):
        x_identity = inputs[:, :, 0:1]
        smoothing_group = inputs[:, :, 1:3]
        smoothing_var = tf.math.reduce_variance(smoothing_group, axis=1)
        _, smoothing_top_idx = tf.nn.top_k(smoothing_var, k=1)
        smoothing_top_idx = smoothing_top_idx + 1
        smoothing_selected = tf.gather(inputs, smoothing_top_idx, axis=2, batch_dims=1)
        kernel = tf.ones((self.smoothing_scale, 1, 1)) / self.smoothing_scale
        smoothing_transformed = tf.nn.conv1d(smoothing_selected, filters=kernel, stride=1, padding='SAME')
        downsampling_group = inputs[:, :, 3:5]
        downsampling_var = tf.math.reduce_variance(downsampling_group, axis=1)
        _, downsampling_top_idx = tf.nn.top_k(downsampling_var, k=1)
        downsampling_top_idx = downsampling_top_idx + 3
        downsampling_selected = tf.gather(inputs, downsampling_top_idx, axis=2, batch_dims=1)
        downsampling_transformed = downsampling_selected[:, ::self.downsampling_factor, :]
        return x_identity, smoothing_transformed, downsampling_transformed

# =============================================================================
# 개선된 EMCNN_3ch: 각 분기에 ImprovedCNNBranch 적용, FC층에 Dropout 추가
# =============================================================================
class ImprovedEMCNN_3ch(tf.keras.Model):
    def __init__(self, num_classes=4, smoothing_scale=2, downsampling_factor=2, dropout_rate=0.5):
        super(ImprovedEMCNN_3ch, self).__init__()
        self.group_selection = GroupedChannelSelection(smoothing_scale, downsampling_factor)
        self.branch_identity = ImprovedCNNBranch()
        self.branch_smoothing = ImprovedCNNBranch()
        self.branch_downsampling = ImprovedCNNBranch()
        self.fc1 = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-6))
        self.dropout1 = Dropout(dropout_rate)
        self.fc2 = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-6))
        self.dropout2 = Dropout(dropout_rate)
        self.classifier = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=False):
        x_identity, x_smoothing, x_downsampling = self.group_selection(inputs)
        feat_identity = self.branch_identity(x_identity, training=training)
        feat_smoothing = self.branch_smoothing(x_smoothing, training=training)
        feat_downsampling = self.branch_downsampling(x_downsampling, training=training)
        features = Concatenate()([feat_identity, feat_smoothing, feat_downsampling])
        x = self.fc1(features)
        x = self.dropout1(x, training=training)
        x = self.fc2(x)
        x = self.dropout2(x, training=training)
        logits = self.classifier(x)
        return logits

def build_improved_emcnn_3ch(input_shape=input_shape, num_classes=num_classes):
    inputs = Input(shape=input_shape)
    outputs = ImprovedEMCNN_3ch(num_classes=num_classes)(inputs)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

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
    
    # 증강 데이터가 있을 경우 np.stack()으로 올바른 차원으로 만듭니다.
    if len(X_augmented) > 0:
        X_augmented_arr = np.stack(X_augmented, axis=0)
        y_augmented_arr = np.array(y_augmented)
        X_balanced = np.concatenate([X, X_augmented_arr], axis=0)
        y_balanced = np.concatenate([y, y_augmented_arr], axis=0)
    else:
        X_balanced, y_balanced = X, y
    
    # 데이터를 섞어줍니다.
    permutation = np.random.permutation(len(y_balanced))
    X_balanced = X_balanced[permutation]
    y_balanced = y_balanced[permutation]
    
    return X_balanced, y_balanced


# =============================================================================
# 데이터 로드 및 전처리: load_ppg_data 함수 (메모리 효율적이며, 전처리/증강 포함)
# =============================================================================
def load_ppg_data(data_dir='/home/bcml1/2025_EMOTION/DEAP_PPG_1s',
                  ppg_segment_length=PPG_SEGMENT_LENGTH, 
                  window_size=WINDOW_SIZE, 
                  step=SLIDING_STEP):
    subject_data = {}
    file_paths = glob.glob(os.path.join(data_dir, "*.npy"))
    print(f"총 {len(file_paths)}개의 PPG 파일을 찾았습니다.")
    pattern = r"(s\d+)_trial_(\d+)_label_(\d+)\.npy"
    
    for file_path in file_paths:
        base = os.path.basename(file_path)
        match = re.match(pattern, base)
        if not match:
            print(f"파일명 패턴 불일치: {base}")
            continue
        subject_id = match.group(1)       # 예: s01
        trial_str = match.group(2)          # 예: 00
        trial_id = int(trial_str)           # 정수형 trial id
        label = int(match.group(3))
        
        data = np.load(file_path).astype(np.float32)  # 원래 shape: (60, 5, 128)
        # (60, 5, 128) -> (60, 128, 5)
        data = np.transpose(data, (0, 2, 1))
        data = data[:, :ppg_segment_length, :]  # (60, 128, 5)
        
        # trial 단위 전처리 (예: 필터링, 노이즈 제거 등)
        data = preprocess_ppg_trial(data, fs=128)
        
        num_segments = data.shape[0]  # 60초
        windowed_samples = []
        trial_ids_list = []
        # sliding window: 0~(60-10)초, 1초 간격 → 총 51개 segment
        for start in range(0, num_segments - window_size + 1, step):
            window = data[start:start+window_size]  # (10, 128, 5)
            combined = np.concatenate(window, axis=0)  # (10*128, 5) = (1280, 5)
            windowed_samples.append(combined)
            trial_ids_list.append(trial_id)
        
        if not windowed_samples:
            print(f"유효한 sliding window가 없습니다: {file_path}")
            continue
        
        windowed_samples = np.array(windowed_samples)  # (51, 1280, 5)
        labels_array = np.full((windowed_samples.shape[0],), label, dtype=np.int32)
        
        if subject_id not in subject_data:
            subject_data[subject_id] = {'X': [], 'y': [], 'trial_ids': []}
        subject_data[subject_id]['X'].append(windowed_samples)
        subject_data[subject_id]['y'].append(labels_array)
        subject_data[subject_id]['trial_ids'].append(np.array(trial_ids_list))
    
    for subject in subject_data:
        subject_data[subject]['X'] = np.concatenate(subject_data[subject]['X'], axis=0)
        subject_data[subject]['y'] = np.concatenate(subject_data[subject]['y'], axis=0)
        subject_data[subject]['trial_ids'] = np.concatenate(subject_data[subject]['trial_ids'], axis=0)
        print(f"{subject} - PPG X shape: {subject_data[subject]['X'].shape}, "
              f"y shape: {subject_data[subject]['y'].shape}, "
              f"trial_ids shape: {subject_data[subject]['trial_ids'].shape}")
        
        # 오버샘플링을 통한 클래스 균형 맞추기
        X_data = subject_data[subject]['X']
        y_data = subject_data[subject]['y']
        unique_classes, counts = np.unique(y_data, return_counts=True)
        max_count = counts.max()
        noise_std = 1e-6
        X_aug_list = [X_data]
        y_aug_list = [y_data]
        trial_ids_list = [subject_data[subject]['trial_ids']]
        for cls in unique_classes:
            cls_indices = np.where(y_data == cls)[0]
            if len(cls_indices) < max_count:
                diff = max_count - len(cls_indices)
                sampled_indices = np.random.choice(cls_indices, diff, replace=True)
                X_samples = X_data[sampled_indices]
                noise = np.random.normal(0.0, noise_std, X_samples.shape)
                X_aug = X_samples + noise
                X_aug_list.append(X_aug)
                y_aug_list.append(np.full(diff, cls, dtype=y_data.dtype))
                trial_ids_list.append(subject_data[subject]['trial_ids'][sampled_indices])
        subject_data[subject]['X'] = np.concatenate(X_aug_list, axis=0)
        subject_data[subject]['y'] = np.concatenate(y_aug_list, axis=0)
        subject_data[subject]['trial_ids'] = np.concatenate(trial_ids_list, axis=0)
        
        # 피험자별 정규화 적용
        subject_data[subject]['X'] = subject_wise_normalize(subject_data[subject]['X'])
        
        print(f"{subject} after augmentation & normalization: PPG X shape: {subject_data[subject]['X'].shape}, "
              f"y shape: {subject_data[subject]['y'].shape}, "
              f"trial_ids shape: {subject_data[subject]['trial_ids'].shape}")
    return subject_data

# =============================================================================
# LOSO Train/Test 함수 (Improved EMCNN_3ch 모델 사용, trial별 평가 및 라벨 균형 적용)
# =============================================================================
def loso_train_test(data_dir, epochs=300, batch_size=128, window_size=10, step=1, sampling_rate=128, noise_std=0.01):
    parent_dir = "/home/bcml1/sigenv/_4주차_ppg/LOSO_result_samsplit3_3layer_raw"
    os.makedirs(parent_dir, exist_ok=True)
    
    # load_ppg_data를 사용하여 subject 단위로 데이터를 로드 (메모리 효율적 처리)
    subject_data = load_ppg_data(data_dir, ppg_segment_length=PPG_SEGMENT_LENGTH, 
                                 window_size=window_size, step=step)
    subject_ids = sorted(subject_data.keys())
    
    for test_subj in subject_ids:
        print(f"\nLOSO Evaluation - Test Subject: {test_subj}")
        test_X = subject_data[test_subj]['X']
        test_y = subject_data[test_subj]['y']
        test_trial_ids = subject_data[test_subj]['trial_ids']
        
        train_X_list, train_y_list = [], []
        for subj in subject_ids:
            if subj == test_subj:
                continue
            train_X_list.append(subject_data[subj]['X'])
            train_y_list.append(subject_data[subj]['y'])
        train_X = np.concatenate(train_X_list, axis=0)
        train_y = np.concatenate(train_y_list, axis=0)
        
        print(f"Subject {test_subj}: Train X shape: {train_X.shape}, y shape: {train_y.shape}, Test X shape: {test_X.shape}")
        
        # train/val split
        train_X, val_X, train_y, val_y = train_test_split(
            train_X, train_y, test_size=0.2, random_state=42, stratify=train_y)
        
        # 훈련 데이터 라벨 균형 맞추기
        train_X, train_y = balance_data(train_X, train_y, noise_std)
        print(f"After balancing: Train X shape: {train_X.shape}, y shape: {train_y.shape}")
        
        # tf.data.Dataset 생성 (test 데이터는 순서 유지)
        def create_ds(X, y, shuffle=True):
            ds = tf.data.Dataset.from_tensor_slices((X, y))
            if shuffle:
                ds = ds.shuffle(buffer_size=1000)
            ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            return ds
        
        train_ds = create_ds(train_X, train_y, shuffle=True)
        val_ds = create_ds(val_X, val_y, shuffle=True)
        test_ds = create_ds(test_X, test_y, shuffle=False)
        
        model = build_improved_emcnn_3ch(input_shape=input_shape, num_classes=NUM_CLASSES)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks, verbose=1)
        
        result_dir = os.path.join(parent_dir, f"subject_{test_subj}")
        os.makedirs(result_dir, exist_ok=True)
        model.save(os.path.join(result_dir, f"{test_subj}_model.keras"))
        np.save(os.path.join(result_dir, f"{test_subj}_history.npy"), history.history)
        
        # 학습 곡선 저장
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.title(f"{test_subj} Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title(f"{test_subj} Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, "training_curves.png"))
        plt.close()
        
        y_true_all = []
        y_pred_all = []
        for X_batch, y_batch in test_ds:
            preds = model.predict(X_batch)
            y_true_all.extend(y_batch.numpy())
            y_pred_all.extend(np.argmax(preds, axis=1))
        
        overall_report = classification_report(y_true_all, y_pred_all, digits=4)
        overall_cm = confusion_matrix(y_true_all, y_pred_all)
        with open(os.path.join(result_dir, "classification_report.txt"), "w") as f:
            f.write(overall_report)
        with open(os.path.join(result_dir, "confusion_matrix.txt"), "w") as f:
            f.write(np.array2string(overall_cm))
        print(f"Test subject {test_subj} evaluation complete.")
        
        # Trial별 평가: test_trial_ids의 순서를 그대로 사용
        unique_trials = np.unique(test_trial_ids)
        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)
        for trial in unique_trials:
            trial_indices = np.where(test_trial_ids == trial)[0]
            if len(trial_indices) == 0:
                continue
            trial_y_true = y_true_all[trial_indices]
            trial_y_pred = y_pred_all[trial_indices]
            trial_report = classification_report(trial_y_true, trial_y_pred, digits=4)
            trial_cm = confusion_matrix(trial_y_true, trial_y_pred)
            with open(os.path.join(result_dir, f"classification_report_trial_{trial:02d}.txt"), "w") as f:
                f.write(trial_report)
            with open(os.path.join(result_dir, f"confusion_matrix_trial_{trial:02d}.txt"), "w") as f:
                f.write(np.array2string(trial_cm))
            print(f"Trial {trial:02d} evaluation:")
            print(trial_report)
        
    return

# =============================================================================
# Main
# =============================================================================
def main():
    data_dir = "/home/bcml1/2025_EMOTION/DEAP_PPG_1s_nonorm"  # 데이터 경로
    loso_train_test(data_dir, epochs=EPOCHS, batch_size=BATCH_SIZE, window_size=WINDOW_SIZE, 
                    step=SLIDING_STEP, sampling_rate=128, noise_std=0.01)

if __name__ == "__main__":
    main()
