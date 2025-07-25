import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
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
# EEG 데이터 로딩: 1초짜리 세그먼트를 슬라이딩 윈도우(10초, 1초 step)로 결합
# 각 파일명: sXX_sample_YY_segment_ZZZ_label_..._2D_DE.npy
# label은 sXX_emotion_labels.npy (40 trial, 각 trial당 label)
# 결과: 각 trial에서 10초 길이로 윈도우를 생성 → 51 샘플/trial, 총 2040 샘플
# 여기서 각 슬라이딩 윈도우의 시작 인덱스(0~50)도 함께 반환하도록 함.
# =============================================================================
def load_eeg_data_sliding_window(subject_id, data_dir, label_dir, window_size=10, step=1):
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
        trial_idx = int(sample_str)
        segment_idx = int(segment_str)
        try:
            data = np.load(file)
        except Exception as e:
            print(f"파일 로드 에러 {file}: {e}")
            continue
        if trial_idx not in trial_dict:
            trial_dict[trial_idx] = []
        trial_dict[trial_idx].append((segment_idx, data))
    
    X_list = []
    y_list = []
    start_list = []  # 각 윈도우의 시작 인덱스 (초) 기록
    
    # EEG label 파일 로드 (40개의 trial label)
    label_file = os.path.join(label_dir, f"{subject_id}_emotion_labels.npy")
    try:
        label_data = np.load(label_file, allow_pickle=True)
    except Exception as e:
        raise ValueError(f"{subject_id} 라벨 파일 로드 실패: {e}")
    
    # 각 trial별로 정렬 후 sliding window 적용
    for trial_idx, seg_list in trial_dict.items():
        seg_list_sorted = sorted(seg_list, key=lambda x: x[0])
        segments = [item[1] for item in seg_list_sorted]  # list of arrays
        segments = np.array(segments)  # shape: (num_segments, H, W, C)
        num_segments = segments.shape[0]
        if num_segments < window_size:
            continue  # window_size보다 작은 trial은 건너뜀
        # 슬라이딩 윈도우 적용: step=1
        for start in range(0, num_segments - window_size + 1, step):
            window_segments = segments[start:start+window_size]
            # 각 세그먼트의 채널 차원을 이어붙여 새로운 sample 생성
            # 예: 각 세그먼트 (H, W, C) → 결합 후 (H, W, C * window_size)
            combined = np.concatenate(window_segments, axis=-1)
            X_list.append(combined)
            start_list.append(start)  # 해당 윈도우의 시작 초 기록
            # trial label: label_data[trial_idx]
            label_value = label_data[trial_idx]
            if hasattr(label_value, 'shape') and label_value.shape != ():
                label_value = int(label_value[0])
            else:
                label_value = int(label_value)
            y_list.append(label_value)
    
    if len(X_list) == 0:
        raise ValueError(f"{subject_id}에 유효한 EEG 데이터가 없습니다.")
    
    X = np.array(X_list)
    y = to_categorical(np.array(y_list), num_classes=4)
    starts = np.array(start_list)  # sliding window 시작 인덱스 배열
    print(f"{subject_id}: EEG 데이터 {X.shape} 생성됨 (총 {X.shape[0]} 샘플)")
    return X, y, starts

# =============================================================================
# PPG 데이터 로딩 및 증강
# 각 파일: (51, 5, 1280) → 전치 후 (51, 1280, 5)
# trial별로 그룹화 → 최종적으로 모든 trial의 샘플을 concat (예: 40*51 = 2040 샘플)
# =============================================================================
def load_ppg_data(data_dir='/home/bcml1/2025_EMOTION/DEAP_PPG_10s_1soverlap_4label'):
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
        data = np.load(file_path)  # shape: (51, 5, 1280)
        data = np.transpose(data, (0, 2, 1))  # → (51, 1280, 5)
        if subject_id not in subject_data:
            subject_data[subject_id] = {'X': [], 'y': []}
        subject_data[subject_id]['X'].append(data)
        subject_data[subject_id]['y'].append(np.full((data.shape[0],), label, dtype=np.int32))
    
    for subject in subject_data:
        subject_data[subject]['X'] = np.concatenate(subject_data[subject]['X'], axis=0)
        subject_data[subject]['y'] = np.concatenate(subject_data[subject]['y'], axis=0)
        print(f"{subject} - PPG X shape: {subject_data[subject]['X'].shape}, y shape: {subject_data[subject]['y'].shape}")
        
        # 오버샘플링으로 클래스 균형 맞추기
        X_data = subject_data[subject]['X']
        y_data = subject_data[subject]['y']
        unique_classes, counts = np.unique(y_data, return_counts=True)
        max_count = counts.max()
        noise_std = 1e-6  # 아주 작은 노이즈
        X_aug_list = [X_data]
        y_aug_list = [y_data]
        for cls in unique_classes:
            cls_indices = np.where(y_data == cls)[0]
            cls_count = len(cls_indices)
            if cls_count < max_count:
                diff = max_count - cls_count
                sampled_indices = np.random.choice(cls_indices, diff, replace=True)
                X_samples = X_data[sampled_indices]
                noise = np.random.normal(loc=0.0, scale=noise_std, size=X_samples.shape)
                X_aug = X_samples + noise
                X_aug_list.append(X_aug)
                y_aug_list.append(np.full(diff, cls, dtype=y_data.dtype))
        X_data_balanced = np.concatenate(X_aug_list, axis=0)
        y_data_balanced = np.concatenate(y_aug_list, axis=0)
        subject_data[subject]['X'] = X_data_balanced
        subject_data[subject]['y'] = y_data_balanced
        print(f"{subject} after augmentation: PPG X shape: {X_data_balanced.shape}, y shape: {y_data_balanced.shape}")
    return subject_data

# =============================================================================
# PPG branch용 CNN (Depthwise Separable Conv 구조)
# =============================================================================
def build_cnn_branch(input_length):
    inputs = Input(shape=(input_length, 1))
    x = tf.keras.layers.Conv1D(filters=8, kernel_size=7, strides=1, padding='same',
                               activation=tf.nn.relu6,
                               kernel_initializer=tf.keras.initializers.HeNormal(),
                               kernel_regularizer=tf.keras.regularizers.l2(1e-6))(inputs)
    x = tf.keras.layers.SeparableConv1D(
            filters=16, kernel_size=7, strides=4, padding='same',
            activation=tf.nn.relu6,
            depthwise_initializer=tf.keras.initializers.HeNormal(),
            pointwise_initializer=tf.keras.initializers.HeNormal(),
            depthwise_regularizer=tf.keras.regularizers.l2(1e-6),
            pointwise_regularizer=tf.keras.regularizers.l2(1e-6)
        )(x)
    x = tf.keras.layers.SeparableConv1D(
            filters=32, kernel_size=7, strides=2, padding='same',
            activation=tf.nn.relu6,
            depthwise_initializer=tf.keras.initializers.HeNormal(),
            pointwise_initializer=tf.keras.initializers.HeNormal(),
            depthwise_regularizer=tf.keras.regularizers.l2(1e-6),
            pointwise_regularizer=tf.keras.regularizers.l2(1e-6)
        )(x)
    x = tf.keras.layers.SeparableConv1D(
            filters=64, kernel_size=7, strides=4, padding='same',
            activation=tf.nn.relu6,
            depthwise_initializer=tf.keras.initializers.HeNormal(),
            pointwise_initializer=tf.keras.initializers.HeNormal(),
            depthwise_regularizer=tf.keras.regularizers.l2(1e-6),
            pointwise_regularizer=tf.keras.regularizers.l2(1e-6)
        )(x)
    x = tf.keras.layers.SeparableConv1D(
            filters=128, kernel_size=7, strides=2, padding='same',
            activation=tf.nn.relu6,
            depthwise_initializer=tf.keras.initializers.HeNormal(),
            pointwise_initializer=tf.keras.initializers.HeNormal(),
            depthwise_regularizer=tf.keras.regularizers.l2(1e-6),
            pointwise_regularizer=tf.keras.regularizers.l2(1e-6)
        )(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    model = Model(inputs=inputs, outputs=x)
    return model

# =============================================================================
# PPG 특징 추출기: 각 채널 별 branch 처리 후 fc layer
# =============================================================================
def create_ppg_feature_extractor(input_shape, num_branches=5):
    ppg_input = Input(shape=input_shape, name='ppg_input_inner')
    branch_outputs = []
    for i in range(num_branches):
        channel = Lambda(lambda x, i=i: x[:, :, i:i+1])(ppg_input)
        branch = build_cnn_branch(input_shape[0])
        branch_output = branch(channel)
        branch_outputs.append(branch_output)
    concatenated = Concatenate()(branch_outputs)
    initializer = tf.keras.initializers.HeNormal()
    reg = tf.keras.regularizers.l2(1e-6)
    x = Dense(128, activation='relu', kernel_initializer=initializer, kernel_regularizer=reg, name='ppg_fc1')(concatenated)
    x = Dense(64, activation='relu', kernel_initializer=initializer, kernel_regularizer=reg, name='ppg_fc2')(x)
    model = Model(inputs=ppg_input, outputs=x, name='ppg_feature_extractor')
    return model

# =============================================================================
# EEG branch용 CNN: 기존 create_base_network 사용 (입력 shape 변경에 주의)
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
def create_multimodal_model(eeg_input_shape, ppg_input_shape, dropout_rate=0.2, num_classes=4):
    # EEG branch
    eeg_input = Input(shape=eeg_input_shape, name='eeg_input')
    eeg_branch = create_base_network(eeg_input_shape, dropout_rate)
    eeg_features = eeg_branch(eeg_input)
    eeg_features = Flatten(name='eeg_flat')(eeg_features)
    
    # PPG branch
    ppg_input = Input(shape=ppg_input_shape, name='ppg_input')
    ppg_feature_extractor = create_ppg_feature_extractor(ppg_input_shape, num_branches=ppg_input_shape[-1])
    ppg_features = ppg_feature_extractor(ppg_input)
    
    # 결합 후 fc layers
    concatenated = Concatenate(name='concat_features')([eeg_features, ppg_features])
    x = Dense(256, activation='relu', name='dense1')(concatenated)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation='relu', name='dense2')(x)
    x = Dropout(dropout_rate)(x)
    output = Dense(num_classes, activation='softmax', name='output')(x)
    
    model = Model(inputs=[eeg_input, ppg_input], outputs=output)
    return model

# =============================================================================
# EEG & PPG 함께 증강하여 클래스 불균형 해결하는 함수
# =============================================================================
def balance_multimodal_dataset(eeg_X, ppg_X, y, noise_std_eeg=0.01, noise_std_ppg=0.01):
    """
    eeg_X: EEG 데이터, shape=(N, ...)
    ppg_X: PPG 데이터, shape=(N, ...)
    y: one-hot encoded label, shape=(N, num_classes)
    두 모달리티에 대해 동일한 인덱스의 샘플에 노이즈를 추가하여 부족 클래스를 보충
    """
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
            eeg_aug = eeg_sample + np.random.normal(loc=0.0, scale=noise_std_eeg, size=eeg_sample.shape)
            ppg_aug = ppg_sample + np.random.normal(loc=0.0, scale=noise_std_ppg, size=ppg_sample.shape)
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
def save_training_history(history, result_dir, subject_id, split_tag):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f"{subject_id} - {split_tag} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f"{subject_id} - {split_tag} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    loss_acc_plot_path = os.path.join(result_dir, "training_history.png")
    plt.savefig(loss_acc_plot_path)
    plt.close()

# =============================================================================
# subject별 멀티모달 모델 학습 (슬라이딩 윈도우의 시작 인덱스 기준 고정 분할)
# =============================================================================
def train_multimodal_model_for_subject(subject_id, eeg_X, ppg_X, y, window_starts, num_classes=4, epochs=150, batch_size=32,
                                       result_dir_base="/home/bcml1/sigenv/_4주차_eeg+ppg/result_fixed1"):
    subject_result_dir = os.path.join(result_dir_base, subject_id)
    os.makedirs(subject_result_dir, exist_ok=True)
    
    # 슬라이딩 윈도우의 시작 인덱스를 기준으로 분할
    # train: start 0~30, valid: 31~40, test: 41~50
    train_idx = np.where(window_starts <= 30)[0]
    valid_idx = np.where((window_starts >= 31) & (window_starts <= 40))[0]
    test_idx  = np.where(window_starts >= 41)[0]
    
    print(f"Subject {subject_id} 분할: Train {len(train_idx)} / Valid {len(valid_idx)} / Test {len(test_idx)}")
    
    eeg_X_train, eeg_X_val, eeg_X_test = eeg_X[train_idx], eeg_X[valid_idx], eeg_X[test_idx]
    ppg_X_train, ppg_X_val, ppg_X_test = ppg_X[train_idx], ppg_X[valid_idx], ppg_X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[valid_idx], y[test_idx]
    
    # 두 모달리티 모두 증강하여 클래스 균형 맞추기
    eeg_X_train, ppg_X_train, y_train = balance_multimodal_dataset(
        eeg_X_train, ppg_X_train, y_train, noise_std_eeg=0.01, noise_std_ppg=0.01
    )
    eeg_X_val, ppg_X_val, y_val = balance_multimodal_dataset(
        eeg_X_val, ppg_X_val, y_val, noise_std_eeg=0.01, noise_std_ppg=0.01
    )
    eeg_X_test, ppg_X_test, y_test = balance_multimodal_dataset(
        eeg_X_test, ppg_X_test, y_test, noise_std_eeg=0.01, noise_std_ppg=0.01
    )
    
    # 모델 생성 및 컴파일
    eeg_input_shape = eeg_X.shape[1:]   # 예: (H, W, C*window_size)
    ppg_input_shape = ppg_X.shape[1:]     # 예: (1280, 5)
    model = create_multimodal_model(eeg_input_shape, ppg_input_shape, dropout_rate=0.2, num_classes=num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    split_tag = "fixed_split"
    split_result_dir = os.path.join(subject_result_dir, split_tag)
    os.makedirs(split_result_dir, exist_ok=True)
    checkpoint_path = os.path.join(split_result_dir, "best_model.keras")
    checkpoint_cb = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    earlystop_cb = EarlyStopping(monitor='val_loss', patience=150, verbose=1)
    
    history = model.fit(
        [eeg_X_train, ppg_X_train], y_train,
        validation_data=([eeg_X_val, ppg_X_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_cb, earlystop_cb],
        verbose=1
    )
    
    save_training_history(history, split_result_dir, subject_id, split_tag)
    
    # 테스트 평가
    model.load_weights(checkpoint_path)
    test_eval = model.evaluate([eeg_X_test, ppg_X_test], y_test, verbose=0)
    y_pred = model.predict([eeg_X_test, ppg_X_test])
    y_test_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    report = classification_report(y_test_labels, y_pred_labels, digits=4)
    cm = confusion_matrix(y_test_labels, y_pred_labels)
    
    with open(os.path.join(split_result_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    with open(os.path.join(split_result_dir, "confusion_matrix.txt"), "w") as f:
        f.write(np.array2string(cm))
    
    print(f"Subject {subject_id} training completed with fixed split. Results saved in {subject_result_dir}")

# =============================================================================
# Main 함수: EEG와 PPG 데이터를 로딩 후, 각 subject별로 멀티모달 모델 학습 수행
# =============================================================================
def main():
    eeg_data_dir = "/home/bcml1/2025_EMOTION/DEAP_EEG_2D_DE"
    eeg_label_dir = "/home/bcml1/2025_EMOTION/DEAP_four_labels"
    ppg_data_dir = "/home/bcml1/2025_EMOTION/DEAP_PPG_10s_1soverlap_4label"
    
    # PPG 데이터 전체 로딩 (subject별)
    ppg_subject_data = load_ppg_data(ppg_data_dir)
    
    subjects = [f"s{str(i).zfill(2)}" for i in range(1, 23)]  # s01 ~ s22
    for subject in subjects:
        print(f"--- {subject} training starts ---")
        try:
            # EEG 데이터를 슬라이딩 윈도우로 10초 결합 (각 trial 당 51샘플, 총 2040 샘플 예상)
            eeg_X, eeg_y_onehot, window_starts = load_eeg_data_sliding_window(subject, eeg_data_dir, eeg_label_dir, window_size=10, step=1)
        except ValueError as e:
            print(e)
            continue
        # EEG 라벨: one-hot → 정수 라벨 (필요 시)
        eeg_y = np.argmax(eeg_y_onehot, axis=1)
        
        if subject not in ppg_subject_data:
            print(f"{subject}의 PPG 데이터가 없습니다. Skipping.")
            continue
        
        ppg_data = ppg_subject_data[subject]
        ppg_X = ppg_data['X']  # (예: (2040, 1280, 5))
        ppg_y = ppg_data['y']  # 정수 라벨
        
        # 두 모달리티의 sample 수 맞추기: 여기서는 최소 샘플 수 사용 (보통 2040이 되어야 함)
        min_samples = min(eeg_X.shape[0], ppg_X.shape[0], window_starts.shape[0])
        if min_samples == 0:
            print(f"{subject}의 데이터가 부족합니다. Skipping.")
            continue
        
        eeg_X = eeg_X[:min_samples]
        eeg_y_onehot = eeg_y_onehot[:min_samples]
        window_starts = window_starts[:min_samples]
        ppg_X = ppg_X[:min_samples]
        # 두 라벨은 동일하다고 가정 (여기서는 eeg_y_onehot 사용)
        y = eeg_y_onehot
        
        print(f"{subject} - Aligned samples: {min_samples}")
        
        train_multimodal_model_for_subject(subject, eeg_X, ppg_X, y, window_starts, num_classes=4,
                                           epochs=150, batch_size=32,
                                           result_dir_base="/home/bcml1/sigenv/_4주차_eeg+ppg/result_fixed1")
        
if __name__ == "__main__":
    main()
