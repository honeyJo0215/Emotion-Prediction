import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, DepthwiseConv1D, Dense, GlobalAveragePooling1D, concatenate, BatchNormalization, Activation, Lambda, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical
import scipy.signal as signal

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
# EMCNN 모델 구성 (입력 shape: (1280, 5))
# =============================================================================
input_shape = (1280, 5)
num_classes = 4  # 예: HVHA, HVLA, LVLA, LVHA

# === MobileNet 기반 Feature Extraction CNN ===
def emcnn_branch(input_layer):
    """ EMCNN의 각 branch에 해당하는 CNN 블록 (MobileNet 기반) """
    x = Conv1D(8, kernel_size=7, strides=1, padding='same', activation='relu',
               kernel_regularizer=l2(0.001))(input_layer)
    
    # Depthwise Separable Convolution 적용
    x = DepthwiseConv1D(kernel_size=7, strides=4, padding='same', depth_multiplier=1,
                        depthwise_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(16, kernel_size=1, strides=1, padding='same', activation='relu',
               kernel_regularizer=l2(0.001))(x)

    x = DepthwiseConv1D(kernel_size=7, strides=2, padding='same', depth_multiplier=1,
                        depthwise_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(32, kernel_size=1, strides=1, padding='same', activation='relu',
               kernel_regularizer=l2(0.001))(x)

    x = DepthwiseConv1D(kernel_size=7, strides=4, padding='same', depth_multiplier=1,
                        depthwise_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(64, kernel_size=1, strides=1, padding='same', activation='relu',
               kernel_regularizer=l2(0.001))(x)

    x = DepthwiseConv1D(kernel_size=7, strides=2, padding='same', depth_multiplier=1,
                        depthwise_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(128, kernel_size=1, strides=1, padding='same', activation='relu',
               kernel_regularizer=l2(0.001))(x)

    # Global Average Pooling 적용
    x = GlobalAveragePooling1D()(x)
    return x

# =============================================================================
# EMCNN 전체 모델 (입력 shape: (1280, 5))
# =============================================================================
def build_emcnn():
    inputs = Input(shape=input_shape)  # 각 sample의 shape: (1280, 5)

    # Lambda Layer를 사용하여 각 채널(branch)를 추출 (axis=2에서 채널 추출)
    branch1 = emcnn_branch(Lambda(lambda x: tf.expand_dims(x[:, :, 0], axis=-1))(inputs))
    branch2 = emcnn_branch(Lambda(lambda x: tf.expand_dims(x[:, :, 1], axis=-1))(inputs))
    branch3 = emcnn_branch(Lambda(lambda x: tf.expand_dims(x[:, :, 2], axis=-1))(inputs))
    branch4 = emcnn_branch(Lambda(lambda x: tf.expand_dims(x[:, :, 3], axis=-1))(inputs))
    branch5 = emcnn_branch(Lambda(lambda x: tf.expand_dims(x[:, :, 4], axis=-1))(inputs))
    
    merged = concatenate([branch1, branch2, branch3, branch4, branch5])
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(merged)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# =============================================================================
# PPG 데이터 로딩 및 슬라이딩 윈도우 적용 함수
# =============================================================================
def sliding_window_ppg(continuous_data, window_size, step):
    """
    continuous_data: numpy array, shape=(total_samples, channels) (예: (7680, 5))
    window_size: 초 단위 (예: 10 -> 10*128 = 1280 샘플)
    step: 초 단위 (예: 1 -> 128 샘플)
    
    반환: numpy array, shape=(num_windows, 1280, channels)
    """
    total_samples = continuous_data.shape[0]
    win_length = window_size * 128
    step_length = step * 128
    windows = []
    for start in range(0, total_samples - win_length + 1, step_length):
        windows.append(continuous_data[start:start+win_length, :])
    return np.array(windows)

def load_ppg_data_with_sliding_window(ppg_data_dir, window_size=10, step=1):
    """
    각 파일의 데이터가 (60, 5, 128)인 경우,
    60개의 1초 segment를 연속 신호(7680, 5)로 결합한 후 슬라이딩 윈도우 적용.
    각 파일은 하나의 trial(sample)에 해당하며, trial별로 윈도우 배열과 라벨을 저장합니다.
    
    최종 subject_data: { subject_id: {'X': [trial1, trial2, ...], 'y': [label1, label2, ...] } }
    각 trial의 shape: (num_windows, 1280, 5)
    """
    subject_data = {}
    file_paths = glob.glob(os.path.join(ppg_data_dir, "*.npy"))
    print(f"총 {len(file_paths)}개의 PPG 파일을 찾았습니다.")
    
    for file_path in file_paths:
        base = os.path.basename(file_path)
        try:
            # 파일명이 예: "s01_trial_01_label_0.npy"라고 가정
            subject_id = base.split('_')[0]
            label_str = base.split('_')[-1].split('.')[0]
            label = int(label_str)
        except Exception as e:
            print("라벨/서브젝트 추출 오류:", file_path, e)
            continue
        
        # 파일 로드 (shape: (60, 5, 128))
        data = np.load(file_path)
        # 60개의 1초 segment를 (60, 5, 128)에서 (60, 128, 5)로 차원 변환
        data = np.transpose(data, (0, 2, 1))
        # 60개의 1초 segment를 연속 신호로 결합 → (60*128, 5) = (7680, 5)
        continuous_data = np.concatenate(data, axis=0)
        # 슬라이딩 윈도우 적용 (예: window_size=10초, 1280 샘플)
        windows = sliding_window_ppg(continuous_data, window_size, step)
        
        if subject_id not in subject_data:
            subject_data[subject_id] = {'X': [], 'y': []}
        # 한 파일이 하나의 trial(sample)에 해당
        subject_data[subject_id]['X'].append(windows)
        subject_data[subject_id]['y'].append(label)
    
    # subject별 trial 수 출력
    for subject in subject_data:
        num_trials = len(subject_data[subject]['X'])
        print(f"{subject} - Trial 수: {num_trials}")
    return subject_data

# =============================================================================
# balance_multimodal_dataset 함수 (여기서는 두 모달리티 대신 동일한 데이터를 2번 전달)
# =============================================================================
# def balance_multimodal_dataset(data_X, _, y, noise_std_eeg=0.01, noise_std_ppg=0.01):
#     # y가 one-hot 인코딩되어 있다고 가정
#     y_labels = np.argmax(y, axis=1)
#     classes = np.unique(y_labels)
#     counts = {cls: np.sum(y_labels == cls) for cls in classes}
#     max_count = max(counts.values())
    
#     X_list = [data_X]
#     y_list = [y]
    
#     for cls in classes:
#         indices = np.where(y_labels == cls)[0]
#         num_needed = max_count - len(indices)
#         for _ in range(num_needed):
#             idx = np.random.choice(indices)
#             sample = data_X[idx]
#             aug_sample = sample + np.random.normal(loc=0.0, scale=noise_std_ppg, size=sample.shape)
#             X_list.append(aug_sample[np.newaxis, ...])
#             y_list.append(y[idx][np.newaxis, ...])
    
#     X_augmented = np.concatenate(X_list, axis=0)
#     y_augmented = np.concatenate(y_list, axis=0)
#     idxs = np.arange(len(y_augmented))
#     np.random.shuffle(idxs)
#     return X_augmented[idxs], X_augmented[idxs], y_augmented[idxs]

def balance_ppg_dataset(data_X, _, y, noise_std_ppg=0.01):
    """ PPG 데이터 증강 적용 """
    y_labels = np.argmax(y, axis=1)
    classes = np.unique(y_labels)
    counts = {cls: np.sum(y_labels == cls) for cls in classes}
    max_count = max(counts.values())
    
    X_list = [data_X]
    y_list = [y]
    
    for cls in classes:
        indices = np.where(y_labels == cls)[0]
        num_needed = max_count - len(indices)
        
        for _ in range(num_needed):
            idx = np.random.choice(indices)
            sample = data_X[idx]
            
            # PPG 데이터 증강 적용
            aug_sample = augment_ppg(sample, noise_std=noise_std_ppg)
            
            X_list.append(aug_sample[np.newaxis, ...])
            y_list.append(y[idx][np.newaxis, ...])
    
    X_augmented = np.concatenate(X_list, axis=0)
    y_augmented = np.concatenate(y_list, axis=0)
    
    idxs = np.arange(len(y_augmented))
    np.random.shuffle(idxs)
    return X_augmented[idxs], X_augmented[idxs], y_augmented[idxs]

def augment_ppg(ppg, noise_std=0.01, shift_max=5, scale_range=(0.9, 1.1), mask_freq=10, mask_factor=0.2, peak_shift=3):
    """ PPG 데이터 증강 함수 (채널별 증강) """
    
    augmented_ppg = np.zeros_like(ppg)

    for ch in range(ppg.shape[1]):  # 각 채널별로 증강 적용
        ppg_ch = ppg[:, ch]

        # 1. 노이즈 추가 (Gaussian Noise)
        ppg_noisy = ppg_ch + np.random.normal(0, noise_std, ppg_ch.shape)

        # 2. 신호 이동 (Shifting)
        shift = np.random.randint(-shift_max, shift_max)
        ppg_shifted = np.roll(ppg_noisy, shift, axis=0)

        # 3. 크기 변환 (Scaling)
        scale_factor = np.random.uniform(*scale_range)
        ppg_scaled = ppg_shifted * scale_factor

        # 4. 주파수 변형 (Frequency Masking)
        fft_signal = np.fft.fft(ppg_scaled)
        fft_signal[:mask_freq] *= (1 - mask_factor)  # 특정 주파수 차단
        ppg_freq = np.real(np.fft.ifft(fft_signal))

        # 5. 피크 변형 (Peak Warping)
        peaks, _ = signal.find_peaks(ppg_freq, distance=30)  # 30 이상의 간격을 가지는 피크 찾기
        if len(peaks) > 0:
            shift_values = np.random.randint(-peak_shift, peak_shift, size=len(peaks))
            for i, shift in zip(peaks, shift_values):
                if 0 <= i + shift < len(ppg_freq):
                    ppg_freq[i] = ppg_freq[i + shift]

        augmented_ppg[:, ch] = ppg_freq  # 변형된 채널 저장
    
    return augmented_ppg

# =============================================================================
# 학습 곡선(accuracy, loss) 플롯 저장 함수
# =============================================================================
def plot_training_curves(history, save_path):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# =============================================================================
# LOSO 평가: 각 테스트 subject에 대해, 각 trial(=sample)을 window 단위로 평가
# - 훈련 데이터: 테스트 subject가 아닌 모든 trial의 window를 합친 후,
#   train_test_split을 통해 80:20 비율(학습/검증)로 분할하고, 각 데이터셋에 증강 적용
# - 테스트: 테스트 subject의 모든 window에 대해 전체 평가 후, trial 단위 평가 진행
# =============================================================================
def train_subject_leave_one_out(subject_data, epochs=300, batch_size=32):
    parent_dir = "/home/bcml1/sigenv/_4주차_ppg/LOSO_result_segsplit_nonorm"
    os.makedirs(parent_dir, exist_ok=True)
    
    subject_ids = sorted(subject_data.keys())
    for test_subj in subject_ids:
        print(f"LOSO - Test Subject: {test_subj}")
        
        # trial 단위로 훈련 subject 데이터 구성 (테스트 subject 제외)
        trial_list = []
        label_list = []
        for subj in subject_ids:
            if subj == test_subj:
                continue
            for trial, label in zip(subject_data[subj]['X'], subject_data[subj]['y']):
                trial_list.append(trial)  # trial shape: (num_windows, 1280, 5)
                label_list.append(label)  # label: 정수
        
        # trial-level split: 전체 trial 리스트에서 80:20으로 학습/검증 분할 (stratify 적용)
        trials_train, trials_val, labels_train, labels_val = train_test_split(
            trial_list, label_list, test_size=0.2, random_state=42, stratify=label_list
        )
        
        # 각 trial 내의 window들을 유지한 채로 훈련 데이터 생성
        X_train = np.concatenate(trials_train, axis=0)  # 모든 훈련 trial의 window들을 하나의 배열로 결합
        y_train = np.concatenate([np.full((trial.shape[0],), label, dtype=np.int32)
                                  for trial, label in zip(trials_train, labels_train)], axis=0)
        
        X_val = np.concatenate(trials_val, axis=0)
        y_val = np.concatenate([np.full((trial.shape[0],), label, dtype=np.int32)
                                for trial, label in zip(trials_val, labels_val)], axis=0)
        
        print("X_train shape:", X_train.shape)
        
        # one-hot encoding
        y_train = to_categorical(y_train, num_classes=num_classes)
        y_val = to_categorical(y_val, num_classes=num_classes)
        
        # 증강 적용 (PPG 단일 모달리티이므로 동일 데이터를 2번 전달)
        X_train, _, y_train = balance_ppg_dataset(X_train, X_train, y_train, noise_std_ppg=0.01)
        X_val, _, y_val = balance_ppg_dataset(X_val, X_val, y_val, noise_std_ppg=0.01)
        
        # 테스트 subject 데이터는 trial 단위로 유지 (평가 시 그룹화를 위해)
        # X_test: 전체 trial의 window들을 하나로 결합 (전체 평가용)
        X_test = np.concatenate(subject_data[test_subj]['X'], axis=0)
        y_test_int = np.array(subject_data[test_subj]['y'])
        y_test = to_categorical(y_test_int, num_classes=num_classes)
        # X_test, _, y_test = balance_ppg_dataset(X_test, X_test, y_test, noise_std_ppg=0.01)
        
        # 모델 생성 (PPG 전용 모델)
        model = build_emcnn()  # PPG 전용 모델은 build_emcnn()을 사용
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.99),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=['accuracy'])
        
        # 학습 (명시적으로 validation 데이터 전달)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(X_val, y_val), verbose=1)
        
        result_dir = os.path.join(parent_dir, f"test_{test_subj}")
        os.makedirs(result_dir, exist_ok=True)
        
        model.save(os.path.join(result_dir, f"{test_subj}_model.keras"))
        np.save(os.path.join(result_dir, f"{test_subj}_history.npy"), history.history)
        
        # 1) 전체 테스트 subject의 윈도우에 대한 평가
        y_pred_test = model.predict(X_test)
        y_test_labels = np.argmax(y_test, axis=1)
        y_pred_labels = np.argmax(y_pred_test, axis=1)
        overall_report = classification_report(y_test_labels, y_pred_labels, digits=4)
        overall_cm = confusion_matrix(y_test_labels, y_pred_labels)
        with open(os.path.join(result_dir, "classification_report_overall.txt"), "w") as f:
            f.write(overall_report)
        with open(os.path.join(result_dir, "confusion_matrix_overall.txt"), "w") as f:
            f.write(np.array2string(overall_cm))
        
        # 2) Trial 단위 평가: 원래 test subject의 trial 단위 정보를 사용
        # 각 trial별로 별도의 평가를 진행 (test subject의 원래 trial들은 subject_data[test_subj]['X']에 리스트로 저장됨)
        for idx, trial in enumerate(subject_data[test_subj]['X']):
            # trial의 shape: (num_windows, 1280, 5)
            # trial 내 각 window에 대해 예측 수행
            trial_preds = model.predict(trial)
            y_true_trial = np.full((trial.shape[0],), subject_data[test_subj]['y'][idx], dtype=np.int32)
            y_true_trial = to_categorical(y_true_trial, num_classes=num_classes)
            y_true_labels = np.argmax(y_true_trial, axis=1)
            y_pred_trial = np.argmax(trial_preds, axis=1)
            report_trial = classification_report(y_true_labels, y_pred_trial, digits=4)
            cm_trial = confusion_matrix(y_true_labels, y_pred_trial)
            with open(os.path.join(result_dir, f"classification_report_trial_{idx:02d}.txt"), "w") as f:
                f.write(report_trial)
            with open(os.path.join(result_dir, f"confusion_matrix_trial_{idx:02d}.txt"), "w") as f:
                f.write(np.array2string(cm_trial))
            print(f"Test subject {test_subj} - Trial {idx:02d} report:")
            print(report_trial)
    
if __name__ == "__main__":
    # data_dir = '/home/bcml1/2025_EMOTION/DEAP_PPG_10s_1soverlap_4label/'
    data_dir = '/home/bcml1/2025_EMOTION/DEAP_PPG_1s_nonorm'
    # window_size=10초, step=1초로 슬라이딩 윈도우 적용 → 각 trial의 shape: (num_windows, 1280, 5)
    subject_data = load_ppg_data_with_sliding_window(data_dir, window_size=10, step=1)
    train_subject_leave_one_out(subject_data, epochs=300, batch_size=32)
    