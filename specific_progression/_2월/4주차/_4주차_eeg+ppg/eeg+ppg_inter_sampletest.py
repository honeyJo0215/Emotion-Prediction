import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold, train_test_split
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
# 각 샘플이 어느 trial에서 왔는지 기록 (trial_ids 반환)
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
    trial_ids = []  # 각 sample의 trial id 저장
    
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
            combined = np.concatenate(window_segments, axis=-1)
            X_list.append(combined)
            trial_ids.append(trial_idx)  # 해당 sample의 trial id 기록
            # trial label은 파일의 라벨 배열이 0-indexed라고 가정하여 trial_idx - 1 사용
            label_value = label_data[trial_idx - 1]
            if hasattr(label_value, 'shape') and label_value.shape != ():
                label_value = int(label_value[0])
            else:
                label_value = int(label_value)
            y_list.append(label_value)
    
    if len(X_list) == 0:
        raise ValueError(f"{subject_id}에 유효한 EEG 데이터가 없습니다.")
    
    X = np.array(X_list)
    y = to_categorical(np.array(y_list), num_classes=4)
    trial_ids = np.array(trial_ids)
    print(f"{subject_id}: EEG 데이터 {X.shape} 생성됨 (총 {X.shape[0]} 샘플)")
    return X, y, trial_ids

# =============================================================================
# PPG 데이터 로딩 및 증강
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
    eeg_X: EEG 데이터, shape=(N, ...),
    ppg_X: PPG 데이터, shape=(N, ...),
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
    loss_acc_plot_path = os.path.join(fold_result_dir, "training_history.png")
    plt.savefig(loss_acc_plot_path)
    plt.close()

# =============================================================================
# fold별 평균 classification report 생성 함수 (5-fold)
# =============================================================================
def create_average_report(fold_metrics):
    avg_precision = np.mean([m['precision'] for m in fold_metrics], axis=0)
    avg_recall = np.mean([m['recall'] for m in fold_metrics], axis=0)
    avg_f1 = np.mean([m['f1'] for m in fold_metrics], axis=0)
    total_support = np.sum([m['support'] for m in fold_metrics], axis=0)
    avg_accuracy = np.mean([m['accuracy'] for m in fold_metrics])
    
    macro_precision = np.mean(avg_precision)
    macro_recall = np.mean(avg_recall)
    macro_f1 = np.mean(avg_f1)
    
    weighted_precision = np.average(avg_precision, weights=total_support)
    weighted_recall = np.average(avg_recall, weights=total_support)
    weighted_f1 = np.average(avg_f1, weights=total_support)
    
    target_names = ['Excited', 'Relaxed', 'Stressed', 'Bored']
    
    report = "Average Classification Report over 5 folds:\n\n"
    report += f"{'Class':<10}{'Precision':>10}{'Recall':>10}{'F1-score':>10}{'Support':>10}\n"
    for i in range(4):
        report += f"{target_names[i]:<10}{avg_precision[i]:10.4f}{avg_recall[i]:10.4f}{avg_f1[i]:10.4f}{int(total_support[i]):10d}\n"
    report += "\n"
    report += f"{'Accuracy':<10}{avg_accuracy:10.4f}\n"
    report += f"{'Macro avg':<10}{macro_precision:10.4f}{macro_recall:10.4f}{macro_f1:10.4f}{np.sum(total_support):10d}\n"
    report += f"{'Weighted avg':<10}{weighted_precision:10.4f}{weighted_recall:10.4f}{weighted_f1:10.4f}{np.sum(total_support):10d}\n"
    
    return report

# =============================================================================
# [수정] subject별 inter-subject leave-one-subject-out 방식으로 학습 후,
# 테스트 subject에 대해 각 trial 별로 classification report 생성하는 함수  
# =============================================================================
def train_multimodal_model_leave_one_subject_out(test_subject_id, train_eeg_X, train_ppg_X, train_y,
                                                 test_eeg_X, test_ppg_X, test_y, test_trial_ids,
                                                 num_classes=4, epochs=150, batch_size=32,
                                                 result_dir_base="/home/bcml1/sigenv/_4주차_eeg+ppg/result_inter_sample"):
    result_dir = os.path.join(result_dir_base, f"test_{test_subject_id}")
    os.makedirs(result_dir, exist_ok=True)
    
    # train/validation 분할 (80:20, stratify 적용)
    eeg_X_train, eeg_X_val, ppg_X_train, ppg_X_val, y_train, y_val = train_test_split(
        train_eeg_X, train_ppg_X, train_y, test_size=0.2, random_state=42, stratify=np.argmax(train_y, axis=1)
    )
    
    # 두 모달리티의 학습 및 검증 데이터에 대해서만 증강 적용
    eeg_X_train, ppg_X_train, y_train = balance_multimodal_dataset(
        eeg_X_train, ppg_X_train, y_train, noise_std_eeg=0.01, noise_std_ppg=0.01
    )
    eeg_X_val, ppg_X_val, y_val = balance_multimodal_dataset(
        eeg_X_val, ppg_X_val, y_val, noise_std_eeg=0.01, noise_std_ppg=0.01
    )
    
    # 테스트 데이터는 원본 데이터를 그대로 사용 (증강 적용 X)
    # test_eeg_X, test_ppg_X, test_y = balance_multimodal_dataset(
    #     test_eeg_X, test_ppg_X, test_y, noise_std_eeg=0.01, noise_std_ppg=0.01
    # )
    
    # 모델 생성 및 컴파일
    eeg_input_shape = train_eeg_X.shape[1:]
    ppg_input_shape = train_ppg_X.shape[1:]
    model = create_multimodal_model(eeg_input_shape, ppg_input_shape, dropout_rate=0.2, num_classes=num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    checkpoint_path = os.path.join(result_dir, "best_model.keras")
    checkpoint_cb = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
    earlystop_cb = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
    
    history = model.fit(
        [eeg_X_train, ppg_X_train], y_train,
        validation_data=([eeg_X_val, ppg_X_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_cb, earlystop_cb],
        verbose=1
    )
    
    save_training_history(history, result_dir, test_subject_id, fold_no=0)
    
    model.load_weights(checkpoint_path)
    y_pred = model.predict([test_eeg_X, test_ppg_X])
    y_test_labels = np.argmax(test_y, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    # 전체 테스트 결과 저장 (전체 report)
    overall_report = classification_report(y_test_labels, y_pred_labels, digits=4)
    overall_cm = confusion_matrix(y_test_labels, y_pred_labels)
    with open(os.path.join(result_dir, "classification_report_overall.txt"), "w") as f:
         f.write(overall_report)
    with open(os.path.join(result_dir, "confusion_matrix_overall.txt"), "w") as f:
         f.write(np.array2string(overall_cm))
    
    # 각 trial 별로 결과 평가
    unique_trials = np.unique(test_trial_ids)
    for trial in unique_trials:
        trial_indices = np.where(test_trial_ids == trial)[0]
        if len(trial_indices) == 0:
            continue
        y_true_trial = np.argmax(test_y[trial_indices], axis=1)
        y_pred_trial = np.argmax(y_pred[trial_indices], axis=1)
        unique_labels = np.unique(y_true_trial)
        if len(unique_labels) == 1:
            true_label = unique_labels[0]
            report_trial = classification_report(
                y_true_trial, y_pred_trial, labels=[true_label], digits=4
            )
            cm_trial = confusion_matrix(
                y_true_trial, y_pred_trial, labels=[true_label]
            )
        else:
            report_trial = classification_report(y_true_trial, y_pred_trial, digits=4)
            cm_trial = confusion_matrix(y_true_trial, y_pred_trial)
        with open(os.path.join(result_dir, f"classification_report_trial_{trial:02d}.txt"), "w") as f:
            f.write(report_trial)
        with open(os.path.join(result_dir, f"confusion_matrix_trial_{trial:02d}.txt"), "w") as f:
            f.write(np.array2string(cm_trial))
        print(f"Test subject {test_subject_id} - Trial {trial:02d} report:")
        print(report_trial)
    
    return model, history

# =============================================================================
# Main 함수: EEG와 PPG 데이터를 로딩 후, inter-subject leave-one-subject-out 방식으로 모델 학습 수행  
# (테스트 subject에 대해 각 trial별 classification report 생성)
# =============================================================================
def main():
    eeg_data_dir = "/home/bcml1/2025_EMOTION/DEAP_EEG_2D_DE"
    eeg_label_dir = "/home/bcml1/2025_EMOTION/DEAP_four_labels"
    ppg_data_dir = "/home/bcml1/2025_EMOTION/DEAP_PPG_10s_1soverlap_4label"
    
    # PPG 데이터 전체 로딩 (subject별)
    ppg_subject_data = load_ppg_data(ppg_data_dir)
    
    subjects = [f"s{str(i).zfill(2)}" for i in range(1, 23)]  # s01 ~ s22
    subject_data = {}
    for subject in subjects:
        print(f"--- {subject} 데이터 로딩 시작 ---")
        try:
            # EEG 데이터를 슬라이딩 윈도우로 10초 결합 (각 trial 당 여러 샘플 생성) + trial id 반환
            eeg_X, eeg_y_onehot, trial_ids = load_eeg_data_sliding_window(subject, eeg_data_dir, eeg_label_dir, window_size=10, step=1)
        except ValueError as e:
            print(e)
            continue
        if subject not in ppg_subject_data:
            print(f"{subject}의 PPG 데이터가 없습니다. Skipping.")
            continue
        ppg_data = ppg_subject_data[subject]
        ppg_X = ppg_data['X']  # (예: (2040, 1280, 5))
        # 두 모달리티의 sample 수 맞추기 (최소 샘플 수 사용)
        min_samples = min(eeg_X.shape[0], ppg_X.shape[0], len(trial_ids))
        if min_samples == 0:
            print(f"{subject}의 데이터가 부족합니다. Skipping.")
            continue
        eeg_X = eeg_X[:min_samples]
        eeg_y_onehot = eeg_y_onehot[:min_samples]
        trial_ids = trial_ids[:min_samples]
        ppg_X = ppg_X[:min_samples]
        subject_data[subject] = {'eeg_X': eeg_X, 'ppg_X': ppg_X, 'y': eeg_y_onehot, 'trial_ids': trial_ids}
        print(f"{subject} - Aligned samples: {min_samples}")
    
    # leave-one-subject-out 평가 수행: 한 subject를 테스트로 하고, 나머지 subject들을 train으로 사용
    result_dir_base = "/home/bcml1/sigenv/_4주차_eeg+ppg/result_inter_sample"
    os.makedirs(result_dir_base, exist_ok=True)
    
    for test_subject in subject_data.keys():
        print(f"--- Leave-One-Subject-Out: Test subject {test_subject} ---")
        # 나머지 subject 데이터를 트레인셋으로 결합
        train_eeg_list = []
        train_ppg_list = []
        train_y_list = []
        for subject in subject_data:
            if subject == test_subject:
                continue
            train_eeg_list.append(subject_data[subject]['eeg_X'])
            train_ppg_list.append(subject_data[subject]['ppg_X'])
            train_y_list.append(subject_data[subject]['y'])
        train_eeg = np.concatenate(train_eeg_list, axis=0)
        train_ppg = np.concatenate(train_ppg_list, axis=0)
        train_y = np.concatenate(train_y_list, axis=0)
        # Test 데이터: 현재 test_subject의 데이터 (trial_ids 포함)
        test_eeg = subject_data[test_subject]['eeg_X']
        test_ppg = subject_data[test_subject]['ppg_X']
        test_y = subject_data[test_subject]['y']
        test_trial_ids = subject_data[test_subject]['trial_ids']
        
        train_multimodal_model_leave_one_subject_out(
            test_subject, train_eeg, train_ppg, train_y,
            test_eeg, test_ppg, test_y, test_trial_ids,
            num_classes=4, epochs=150, batch_size=32,
            result_dir_base=result_dir_base
        )
        
if __name__ == "__main__":
    main()
