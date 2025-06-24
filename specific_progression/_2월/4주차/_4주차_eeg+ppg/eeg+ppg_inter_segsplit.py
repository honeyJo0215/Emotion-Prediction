import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import mode  # trial별 다수결을 위한 모듈

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, 
                                     Flatten, Dense, Reshape, Concatenate, Lambda)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

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
# 전역 하이퍼파라미터 설정 (초 단위)
# =============================================================================
# "window_size"는 sample을 구성할 초 수 (즉, 몇 개의 1초 segment를 이어붙일지)
WINDOW_SIZE = 10       # 예: 10초짜리 sample
SLIDING_STEP = 1       # 1초 간격 (overlap)

# PPG의 1초 segment 길이 (128 Hz → 128 샘플, 5채널)
PPG_SEGMENT_LENGTH = 128

# 학습 관련 하이퍼파라미터
EPOCHS = 150
BATCH_SIZE = 32
DROPOUT_RATE = 0.2
NUM_CLASSES = 4

# =============================================================================
# EEG 데이터 로딩
# 각 파일은 1초 분량의 segment(예: shape (4,6,6))라고 가정.
# window_size (초)와 SLIDING_STEP (초)를 적용하여 sample 생성:
# sample shape = (H, W, C * window_size). 예: (4,6,6*10) → (4,6,60) if window_size=10.
# 라벨은 각 trial의 index (1~40)에 맞추어 label_data[trial_idx-1]에서 가져옴.
# =============================================================================
def load_eeg_data_sliding_window(subject_id, data_dir, label_dir, window_size=WINDOW_SIZE, step=SLIDING_STEP):
    pattern = os.path.join(data_dir, f"{subject_id}_sample_*_segment_*_2D_DE.npy")
    file_list = glob.glob(pattern)
    if len(file_list) == 0:
        raise ValueError(f"{subject_id}에 해당하는 EEG 데이터 파일이 없습니다.")
    
    # trial_dict: key = trial index (sample_YY), value = list of (segment_index, data)
    trial_dict = {}
    regex = re.compile(r"(s\d{2})_sample_(\d{2})_segment_(\d{3})_label_.*_2D_DE\.npy")
    
    for file in file_list:
        basename = os.path.basename(file)
        m = regex.match(basename)
        if not m:
            print(f"패턴에 맞지 않는 파일명: {basename}")
            continue
        subj, sample_str, segment_str = m.group(1), m.group(2), m.group(3)
        trial_idx = int(sample_str)    # trial index (1~40)
        segment_idx = int(segment_str)   # 각 trial 내 segment 번호 (총 60)
        try:
            data = np.load(file)
        except Exception as e:
            print(f"파일 로드 에러 {file}: {e}")
            continue
        trial_dict.setdefault(trial_idx, []).append((segment_idx, data))
    
    X_list = []
    y_list = []
    trial_ids = []  # 각 sample의 trial id 기록
    
    # EEG 라벨 파일 로드 (40 trial의 라벨)
    label_file = os.path.join(label_dir, f"{subject_id}_emotion_labels.npy")
    try:
        label_data = np.load(label_file, allow_pickle=True)
    except Exception as e:
        raise ValueError(f"{subject_id} 라벨 파일 로드 실패: {e}")
    
    # 각 trial별로 정렬 후, sliding window 적용
    for trial_idx, seg_list in trial_dict.items():
        seg_list_sorted = sorted(seg_list, key=lambda x: x[0])
        segments = np.array([item[1] for item in seg_list_sorted])  # shape: (num_segments, H, W, C)
        num_segments = segments.shape[0]
        if num_segments < window_size:
            continue
        # sliding window 적용 (1초 단위 overlap)
        for start in range(0, num_segments - window_size + 1, step):
            window_segments = segments[start:start+window_size]
            # 각 segment의 마지막 차원을 이어붙임 → shape: (H, W, C*window_size)
            combined = np.concatenate(window_segments, axis=-1)
            X_list.append(combined)
            trial_ids.append(trial_idx)
            # 라벨 매칭: trial index는 1~40, label_data는 0-indexed이므로 trial_idx-1 사용
            label_value = label_data[trial_idx - 1]
            if hasattr(label_value, 'shape') and label_value.shape != ():
                label_value = int(label_value[0])
            else:
                label_value = int(label_value)
            y_list.append(label_value)
    
    if not X_list:
        raise ValueError(f"{subject_id}에 유효한 EEG 데이터가 없습니다.")
    
    X = np.array(X_list)
    y = to_categorical(np.array(y_list), num_classes=NUM_CLASSES)
    trial_ids = np.array(trial_ids)
    print(f"{subject_id}: EEG 데이터 {X.shape} 생성됨 (총 {X.shape[0]} 샘플)")
    return X, y, trial_ids

# =============================================================================
# PPG 데이터 로딩 및 증강
# 각 파일은 60개의 1초 segment (각 segment의 shape = (PPG_SEGMENT_LENGTH, 5))라고 가정.
# window_size (초)와 SLIDING_STEP (초)를 적용하여 sample 생성:
# sample shape = (PPG_SEGMENT_LENGTH * window_size, 5). 예: (128*10,5) → (1280,5) if window_size=10.
# =============================================================================
def load_ppg_data(data_dir='/home/bcml1/2025_EMOTION/DEAP_PPG_1s_nonorm', 
                  ppg_segment_length=PPG_SEGMENT_LENGTH, 
                  window_size=WINDOW_SIZE, 
                  step=SLIDING_STEP):
    subject_data = {}
    file_paths = glob.glob(os.path.join(data_dir, "*.npy"))
    print(f"총 {len(file_paths)}개의 PPG 파일을 찾았습니다.")
    
    for file_path in file_paths:
        base = os.path.basename(file_path)
        try:
            subject_id = base.split('_')[0]
            label_str = base.split('_')[-1].split('.')[0]
            label = int(label_str)
        except Exception as e:
            print("라벨/서브젝트 추출 오류:", file_path, e)
            continue
        data = np.load(file_path)  # 원래 shape: (60, 5, 128) → 60 segment, 5채널, 128 샘플(1초)
        # 만약 (60, 5, 128)라면 transpose하여 (60, 128, 5)
        data = np.transpose(data, (0, 2, 1))
        # 필요한 segment 길이 만큼 슬라이싱 (기본 128)
        data = data[:, :ppg_segment_length, :]
        
        num_segments = data.shape[0]  # 보통 60
        windowed_samples = []
        # sliding window 적용 (1초 단위 overlap)
        for start in range(0, num_segments - window_size + 1, step):
            window = data[start:start+window_size]  # shape: (window_size, ppg_segment_length, 5)
            # time axis로 concatenate → shape: (ppg_segment_length * window_size, 5)
            combined = np.concatenate(window, axis=0)
            windowed_samples.append(combined)
        if not windowed_samples:
            print(f"유효한 sliding window가 없습니다: {file_path}")
            continue
        windowed_samples = np.array(windowed_samples)
        labels = np.full((windowed_samples.shape[0],), label, dtype=np.int32)
        
        if subject_id not in subject_data:
            subject_data[subject_id] = {'X': [], 'y': []}
        subject_data[subject_id]['X'].append(windowed_samples)
        subject_data[subject_id]['y'].append(labels)
    
    for subject in subject_data:
        subject_data[subject]['X'] = np.concatenate(subject_data[subject]['X'], axis=0)
        subject_data[subject]['y'] = np.concatenate(subject_data[subject]['y'], axis=0)
        print(f"{subject} - PPG X shape: {subject_data[subject]['X'].shape}, y shape: {subject_data[subject]['y'].shape}")
        
        # 오버샘플링으로 클래스 균형 맞추기
        X_data = subject_data[subject]['X']
        y_data = subject_data[subject]['y']
        unique_classes, counts = np.unique(y_data, return_counts=True)
        max_count = counts.max()
        noise_std = 1e-6
        X_aug_list = [X_data]
        y_aug_list = [y_data]
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
        subject_data[subject]['X'] = np.concatenate(X_aug_list, axis=0)
        subject_data[subject]['y'] = np.concatenate(y_aug_list, axis=0)
        print(f"{subject} after augmentation: PPG X shape: {subject_data[subject]['X'].shape}, y shape: {subject_data[subject]['y'].shape}")
    return subject_data

# =============================================================================
# PPG branch용 CNN (Depthwise Separable Conv 구조)
# =============================================================================
def build_cnn_branch(input_length):
    inputs = Input(shape=(input_length, 1))
    x = tf.keras.layers.Conv1D(8, 7, strides=1, padding='same',
                               activation=tf.nn.relu6,
                               kernel_initializer=tf.keras.initializers.HeNormal(),
                               kernel_regularizer=tf.keras.regularizers.l2(1e-6))(inputs)
    x = tf.keras.layers.SeparableConv1D(16, 7, strides=4, padding='same',
                                        activation=tf.nn.relu6,
                                        depthwise_initializer=tf.keras.initializers.HeNormal(),
                                        pointwise_initializer=tf.keras.initializers.HeNormal(),
                                        depthwise_regularizer=tf.keras.regularizers.l2(1e-6),
                                        pointwise_regularizer=tf.keras.regularizers.l2(1e-6))(x)
    x = tf.keras.layers.SeparableConv1D(32, 7, strides=2, padding='same',
                                        activation=tf.nn.relu6,
                                        depthwise_initializer=tf.keras.initializers.HeNormal(),
                                        pointwise_initializer=tf.keras.initializers.HeNormal(),
                                        depthwise_regularizer=tf.keras.regularizers.l2(1e-6),
                                        pointwise_regularizer=tf.keras.regularizers.l2(1e-6))(x)
    x = tf.keras.layers.SeparableConv1D(64, 7, strides=4, padding='same',
                                        activation=tf.nn.relu6,
                                        depthwise_initializer=tf.keras.initializers.HeNormal(),
                                        pointwise_initializer=tf.keras.initializers.HeNormal(),
                                        depthwise_regularizer=tf.keras.regularizers.l2(1e-6),
                                        pointwise_regularizer=tf.keras.regularizers.l2(1e-6))(x)
    x = tf.keras.layers.SeparableConv1D(128, 7, strides=2, padding='same',
                                        activation=tf.nn.relu6,
                                        depthwise_initializer=tf.keras.initializers.HeNormal(),
                                        pointwise_initializer=tf.keras.initializers.HeNormal(),
                                        depthwise_regularizer=tf.keras.regularizers.l2(1e-6),
                                        pointwise_regularizer=tf.keras.regularizers.l2(1e-6))(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    return Model(inputs=inputs, outputs=x)

# =============================================================================
# PPG 특징 추출기: 각 채널별 branch 처리 후 fc layer
# =============================================================================
def create_ppg_feature_extractor(input_shape, num_branches=5):
    ppg_input = Input(shape=input_shape, name='ppg_input_inner')
    branch_outputs = []
    for i in range(num_branches):
        channel = Lambda(lambda x, i=i: x[:, :, i:i+1])(ppg_input)
        branch = build_cnn_branch(input_shape[0])
        branch_outputs.append(branch(channel))
    concatenated = Concatenate()(branch_outputs)
    initializer = tf.keras.initializers.HeNormal()
    reg = tf.keras.regularizers.l2(1e-6)
    x = Dense(128, activation='relu', kernel_initializer=initializer, kernel_regularizer=reg, name='ppg_fc1')(concatenated)
    x = Dense(64, activation='relu', kernel_initializer=initializer, kernel_regularizer=reg, name='ppg_fc2')(x)
    return Model(inputs=ppg_input, outputs=x, name='ppg_feature_extractor')

# =============================================================================
# EEG branch용 CNN: create_base_network (입력 shape 변경 주의)
# =============================================================================
def create_base_network(input_dim, dropout_rate):
    seq = Sequential()
    seq.add(Conv2D(64, 5, activation='relu', padding='same', name='conv1', input_shape=input_dim))
    seq.add(BatchNormalization())
    seq.add(Dropout(dropout_rate))
    seq.add(Conv2D(128, 4, activation='relu', padding='same', name='conv2'))
    seq.add(BatchNormalization())
    seq.add(Dropout(dropout_rate))
    seq.add(Conv2D(256, 4, activation='relu', padding='same', name='conv3'))
    seq.add(BatchNormalization())
    seq.add(Dropout(dropout_rate))
    seq.add(Conv2D(64, 1, activation='relu', padding='same', name='conv4'))
    seq.add(MaxPooling2D(2, 2, name='pool1'))
    seq.add(BatchNormalization())
    seq.add(Dropout(dropout_rate))
    seq.add(Flatten(name='fla1'))
    seq.add(Dense(512, activation='relu', name='dense1'))
    seq.add(Reshape((1, 512), name='reshape'))
    seq.add(BatchNormalization())
    seq.add(Dropout(dropout_rate))
    return seq

# =============================================================================
# 멀티모달 모델 생성: EEG와 PPG 특징 추출 후 concat → 분류기
# =============================================================================
def create_multimodal_model(eeg_input_shape, ppg_input_shape, dropout_rate=DROPOUT_RATE, num_classes=NUM_CLASSES):
    eeg_input = Input(shape=eeg_input_shape, name='eeg_input')
    eeg_features = Flatten(name='eeg_flat')(create_base_network(eeg_input_shape, dropout_rate)(eeg_input))
    
    ppg_input = Input(shape=ppg_input_shape, name='ppg_input')
    ppg_features = create_ppg_feature_extractor(ppg_input_shape, num_branches=ppg_input_shape[-1])(ppg_input)
    
    concatenated = Concatenate(name='concat_features')([eeg_features, ppg_features])
    x = Dense(256, activation='relu', name='dense1')(concatenated)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation='relu', name='dense2')(x)
    x = Dropout(dropout_rate)(x)
    output = Dense(num_classes, activation='softmax', name='output')(x)
    return Model(inputs=[eeg_input, ppg_input], outputs=output)

# =============================================================================
# EEG & PPG 데이터에 대해 증강하여 클래스 불균형 해결
# =============================================================================
def balance_multimodal_dataset(eeg_X, ppg_X, y, noise_std_eeg=0.01, noise_std_ppg=0.01):
    y_labels = np.argmax(y, axis=1)
    classes = np.unique(y_labels)
    counts = {cls: np.sum(y_labels == cls) for cls in classes}
    max_count = max(counts.values())
    
    eeg_list = [eeg_X]
    ppg_list = [ppg_X]
    y_list = [y]
    
    for cls in classes:
        indices = np.where(y_labels == cls)[0]
        num_needed = max_count - len(indices)
        for _ in range(num_needed):
            idx = np.random.choice(indices)
            eeg_sample = eeg_X[idx]
            ppg_sample = ppg_X[idx]
            eeg_aug = eeg_sample + np.random.normal(0.0, noise_std_eeg, eeg_sample.shape)
            ppg_aug = ppg_sample + np.random.normal(0.0, noise_std_ppg, ppg_sample.shape)
            eeg_list.append(eeg_aug[np.newaxis, ...])
            ppg_list.append(ppg_aug[np.newaxis, ...])
            y_list.append(y[idx][np.newaxis, ...])
    
    eeg_augmented = np.concatenate(eeg_list, axis=0)
    ppg_augmented = np.concatenate(ppg_list, axis=0)
    y_augmented = np.concatenate(y_list, axis=0)
    idxs = np.arange(len(y_augmented))
    np.random.shuffle(idxs)
    return eeg_augmented[idxs], ppg_augmented[idxs], y_augmented[idxs]

# =============================================================================
# 학습 그래프 저장 헬퍼 함수
# =============================================================================
def save_training_history(history, fold_result_dir, subject_id, fold_no):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f"{subject_id} - Fold {fold_no} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f"{subject_id} - Fold {fold_no} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fold_result_dir, "training_history.png"))
    plt.close()

# =============================================================================
# leave-one-subject-out 방식 학습 후, 각 trial 내 sample 다수결로 trial-level 평가
# =============================================================================

def train_multimodal_model_leave_one_subject_out(test_subject_id, train_eeg_X, train_ppg_X, train_y,
                                                 test_eeg_X, test_ppg_X, test_y, test_trial_ids,
                                                 num_classes=NUM_CLASSES, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                                 result_dir_base="/home/bcml1/sigenv/_4주차_eeg+ppg/result_inter_seg1s"):
    result_dir = os.path.join(result_dir_base, f"test_{test_subject_id}")
    os.makedirs(result_dir, exist_ok=True)
    
    eeg_X_train, eeg_X_val, ppg_X_train, ppg_X_val, y_train, y_val = train_test_split(
        train_eeg_X, train_ppg_X, train_y, test_size=0.2, random_state=42, stratify=np.argmax(train_y, axis=1)
    )
    
    eeg_X_train, ppg_X_train, y_train = balance_multimodal_dataset(eeg_X_train, ppg_X_train, y_train, 0.01, 0.01)
    eeg_X_val, ppg_X_val, y_val = balance_multimodal_dataset(eeg_X_val, ppg_X_val, y_val, 0.01, 0.01)
    
    eeg_input_shape = train_eeg_X.shape[1:]
    ppg_input_shape = train_ppg_X.shape[1:]
    model = create_multimodal_model(eeg_input_shape, ppg_input_shape, dropout_rate=DROPOUT_RATE, num_classes=num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    checkpoint_path = os.path.join(result_dir, "best_model.keras")
    checkpoint_cb = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
    earlystop_cb = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
    
    history = model.fit([eeg_X_train, ppg_X_train], y_train,
                        validation_data=([eeg_X_val, ppg_X_val], y_val),
                        epochs=epochs, batch_size=batch_size,
                        callbacks=[checkpoint_cb, earlystop_cb], verbose=1)
    
    save_training_history(history, result_dir, test_subject_id, fold_no=0)
    
    model.load_weights(checkpoint_path)
    y_pred = model.predict([test_eeg_X, test_ppg_X])
    
    y_test_sample = np.argmax(test_y, axis=1)
    y_pred_sample = np.argmax(y_pred, axis=1)
    overall_report = classification_report(y_test_sample, y_pred_sample, digits=4)
    overall_cm = confusion_matrix(y_test_sample, y_pred_sample)
    with open(os.path.join(result_dir, "classification_report_overall.txt"), "w") as f:
        f.write(overall_report)
    with open(os.path.join(result_dir, "confusion_matrix_overall.txt"), "w") as f:
        f.write(np.array2string(overall_cm))
    
    aggregated_y_true = []
    aggregated_y_pred = []
    unique_trials = np.unique(test_trial_ids)
    for trial in unique_trials:
        trial_indices = np.where(test_trial_ids == trial)[0]
        if len(trial_indices) == 0:
            continue
        trial_true = np.argmax(test_y[trial_indices][0])
        # 수정된 부분: .mode[0] 대신 .mode.item() 사용
        trial_pred = mode(np.argmax(y_pred[trial_indices], axis=1)).mode.item()
        aggregated_y_true.append(trial_true)
        aggregated_y_pred.append(trial_pred)
    
    trial_report = classification_report(aggregated_y_true, aggregated_y_pred, digits=4)
    trial_cm = confusion_matrix(aggregated_y_true, aggregated_y_pred)
    with open(os.path.join(result_dir, "classification_report_trials.txt"), "w") as f:
        f.write(trial_report)
    with open(os.path.join(result_dir, "confusion_matrix_trials.txt"), "w") as f:
        f.write(np.array2string(trial_cm))
    
    print("=== Sample-level Evaluation ===")
    print(overall_report)
    print("=== Trial-level Evaluation (aggregated via majority vote) ===")
    print(trial_report)
    
    return model, history


# =============================================================================
# Main 함수: EEG와 PPG 데이터를 로딩 후, leave-one-subject-out 방식으로 학습 및 평가
# =============================================================================
def main():
    eeg_data_dir = "/home/bcml1/2025_EMOTION/DEAP_EEG_2D_DE"
    eeg_label_dir = "/home/bcml1/2025_EMOTION/DEAP_four_labels"
    ppg_data_dir = "/home/bcml1/2025_EMOTION/DEAP_PPG_1s_nonorm"
    
    # PPG 데이터 로딩 (동일한 WINDOW_SIZE, SLIDING_STEP 적용)
    ppg_subject_data = load_ppg_data(ppg_data_dir, ppg_segment_length=PPG_SEGMENT_LENGTH, 
                                     window_size=WINDOW_SIZE, step=SLIDING_STEP)
    
    subjects = [f"s{str(i).zfill(2)}" for i in range(1, 23)]
    subject_data = {}
    for subject in subjects:
        print(f"--- {subject} 데이터 로딩 시작 ---")
        try:
            eeg_X, eeg_y_onehot, trial_ids = load_eeg_data_sliding_window(subject, eeg_data_dir, eeg_label_dir, 
                                                                          window_size=WINDOW_SIZE, step=SLIDING_STEP)
        except ValueError as e:
            print(e)
            continue
        if subject not in ppg_subject_data:
            print(f"{subject}의 PPG 데이터가 없습니다. Skipping.")
            continue
        ppg_data = ppg_subject_data[subject]
        ppg_X = ppg_data['X']
        min_samples = min(eeg_X.shape[0], ppg_X.shape[0], len(trial_ids))
        if min_samples == 0:
            print(f"{subject}의 데이터가 부족합니다. Skipping.")
            continue
        eeg_X, eeg_y_onehot, trial_ids, ppg_X = eeg_X[:min_samples], eeg_y_onehot[:min_samples], trial_ids[:min_samples], ppg_X[:min_samples]
        subject_data[subject] = {'eeg_X': eeg_X, 'ppg_X': ppg_X, 'y': eeg_y_onehot, 'trial_ids': trial_ids}
        print(f"{subject} - Aligned samples: {min_samples}")
    
    result_dir_base = "/home/bcml1/sigenv/_4주차_eeg+ppg/result_inter_seg1s"
    os.makedirs(result_dir_base, exist_ok=True)
    
    for test_subject in subject_data.keys():
        print(f"--- Leave-One-Subject-Out: Test subject {test_subject} ---")
        train_eeg_list, train_ppg_list, train_y_list = [], [], []
        for subject in subject_data:
            if subject == test_subject:
                continue
            train_eeg_list.append(subject_data[subject]['eeg_X'])
            train_ppg_list.append(subject_data[subject]['ppg_X'])
            train_y_list.append(subject_data[subject]['y'])
        train_eeg = np.concatenate(train_eeg_list, axis=0)
        train_ppg = np.concatenate(train_ppg_list, axis=0)
        train_y = np.concatenate(train_y_list, axis=0)
        test_eeg = subject_data[test_subject]['eeg_X']
        test_ppg = subject_data[test_subject]['ppg_X']
        test_y = subject_data[test_subject]['y']
        test_trial_ids = subject_data[test_subject]['trial_ids']
        
        train_multimodal_model_leave_one_subject_out(test_subject, train_eeg, train_ppg, train_y,
                                                      test_eeg, test_ppg, test_y, test_trial_ids,
                                                      num_classes=NUM_CLASSES, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                                      result_dir_base=result_dir_base)
        
if __name__ == "__main__":
    main()
