# 유니모달
# 나는 4초로 해보기.
# DE feature는 전처리되어 있는 거 쓰고
# Raw data는 800 segment단위로 사용해서 코드 작성

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------------------------------------------------------
# GPU 메모리 제한 (필요 시)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Helper 함수: trial 데이터를 시간 슬라이스로 분리 및 라벨 복제
# -----------------------------------------------------------------------------
def split_trial_to_time_slices(trial):
    """
    단일 trial 데이터의 shape: (62, T, 4)
    각 trial의 시간 슬라이스를 (62, 1, 4)로 분리합니다.
    반환 shape: (T, 62, 1, 4)
    """
    channels, T, bands = trial.shape
    slices = []
    for t in range(T):
        slices.append(trial[:, t:t+1, :])  # (62, 1, 4)
    return np.array(slices)

def split_trials_list_to_time_slices(trials_list):
    """
    trials_list: list of trial 배열, 각각 shape: (62, T_i, 4)
    각 trial에 대해 시간 슬라이스 분리 후, 모두 concatenate하여 
    최종 배열 shape: (sum_i(T_i), 62, 1, 4)
    """
    slices_list = [split_trial_to_time_slices(trial) for trial in trials_list]
    if slices_list:
        return np.concatenate(slices_list, axis=0)
    else:
        return None

def replicate_labels_for_trials(trials_list, labels):
    """
    trials_list: list of trial 배열, 각각 shape: (62, T_i, 4)
    labels: array of trial별 one-hot 라벨 (shape: (num_trials, num_classes))
    각 trial의 T_i에 대해 라벨을 복제하여, 최종 shape: (sum_i(T_i), num_classes)로 만듭니다.
    """
    replicated = []
    for trial, label in zip(trials_list, labels):
        T = trial.shape[1]  # trial의 시간 길이
        replicated.append(np.repeat(label[np.newaxis, :], T, axis=0))
    if replicated:
        return np.concatenate(replicated, axis=0)
    else:
        return None

# -----------------------------------------------------------------------------
# SEED-IV 데이터 로드 함수 (eeg_feature_smooth, preprocessed data)
# -----------------------------------------------------------------------------
def load_seediv_data(base_dirs, de_keys=["de_movingAve"], psd_keys=["psd_movingAve"]):
    """
    파일명 예:
      1_20160518_sample_01_de_movingAve_label_1.npy
      1_20160518_sample_01_psd_movingAve_label_1.npy
    각 파일은 원래 shape: (62, T, 5)이며, delta 밴드를 제거하여 (62, T, 4)로 변환합니다.
    DE와 PSD 데이터가 모두 있는 trial만 선택합니다.
    
    Returns:
      de_list: list of DE trial 배열 (각 배열 shape: (62, T, 4))
      psd_list: list of PSD trial 배열 (각 배열 shape: (62, T, 4))
      label_list: list of 라벨 (정수)
      subject_list: list of subject id (문자열)
    """
    data_de = {}
    data_psd = {}
    
    for base_dir in base_dirs:
        file_list = glob.glob(os.path.join(base_dir, "*.npy"))
        for file in file_list:
            filename = os.path.basename(file)
            parts = filename.replace('.npy','').split('_')
            if len(parts) < 8:
                continue
            subject = parts[0]
            trial = parts[3]
            key_name = parts[4] + "_" + parts[5]
            label_str = parts[7]
            try:
                label_val = int(label_str)
            except:
                continue
            arr = np.load(file)  # 원래 shape: (62, T, 5)
            arr = arr[..., 1:]   # delta 제거 → (62, T, 4)
            
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

# -----------------------------------------------------------------------------
# SEED-IV raw 데이터 로드 함수 (eeg_raw_data)
# -----------------------------------------------------------------------------
def load_seediv_raw_data(base_dirs, raw_key="raw"):
    """
    파일명 예:
      1_20160518_sample_01_raw_label_1.npy
    각 파일은 raw data로 shape: (62, T, 4)를 가정합니다.
    각 파일에 대해, 각 초(시간 슬라이스)를 추출하여 map_channels_to_2d() 함수를 적용하면
    최종적으로 (4, 8, 8, 1) shape의 샘플이 생성됩니다.
    
    Returns:
      raw_list: list of raw trial 샘플, 각 샘플 shape: (4, 8, 8, 1)
      label_list: list of 정수 라벨
      subject_list: list of subject id (문자열)
    """
    def map_channels_to_2d(raw_segment):
        """
        raw_segment: numpy array, shape (62, 4)
        각 밴드별로 62개의 값을 0-padding하여 64개(8x8)로 재구성 → (8,8,4)
        """
        num_channels, num_bands = raw_segment.shape
        mapped = np.zeros((8, 8, num_bands))
        for band in range(num_bands):
            vec = raw_segment[:, band]
            vec_padded = np.pad(vec, (0, 64 - num_channels), mode='constant')
            mapped[:, :, band] = vec_padded.reshape(8, 8)
        return mapped

    raw_list = []
    label_list = []
    subject_list = []
    for base_dir in base_dirs:
        file_list = glob.glob(os.path.join(base_dir, "*.npy"))
        for file in file_list:
            filename = os.path.basename(file)
            parts = filename.replace('.npy','').split('_')
            if len(parts) < 8:
                continue
            subject = parts[0]
            trial = parts[3]
            key_name = parts[4]  # raw data 파일은 key에 "raw" 포함
            label_str = parts[7]
            try:
                label_val = int(label_str)
            except:
                continue
            if raw_key not in key_name.lower():
                continue
            try:
                arr = np.load(file)  # raw data, shape: (62, T, 4)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
            T = arr.shape[1]
            # 각 초마다 분리하여 map_channels_to_2d 적용 → (4,8,8,1)
            for t in range(T):
                segment = arr[:, t, :]  # shape: (62, 4)
                mapped_img = map_channels_to_2d(segment)  # (8,8,4)
                mapped_img = np.transpose(mapped_img, (2,0,1))  # (4,8,8)
                mapped_img = np.expand_dims(mapped_img, axis=-1)  # (4,8,8,1)
                raw_list.append(mapped_img)
                label_list.append(label_val)
                subject_list.append(subject)
            print(f"Loaded {T} segments from {file} for subject {subject} with label {label_val}.")
    raw_list = np.array(raw_list)
    label_list = np.array(label_list)
    subject_list = np.array(subject_list)
    print(f"Total raw samples: {raw_list.shape[0]}, sample shape: {raw_list.shape[1:]}")
    return raw_list, label_list, subject_list

# -----------------------------------------------------------------------------
# SEED-IV 파인튜닝용 데이터 전처리 함수 (trial 단위 split 후 시간 슬라이스 적용)
# data_mode: "smooth" (preprocessed data) 또는 "raw" (raw data)
# -----------------------------------------------------------------------------
def load_seediv_data_finetune(base_dirs, test_size=0.2, data_mode="smooth"):
    """
    base_dirs: 데이터가 저장된 경로 리스트.
    data_mode: "smooth"이면 eeg_feature_smooth 데이터를, "raw"이면 raw 데이터를 사용.
    
    전체 trial 단위 데이터를 로드한 후, train/test split을 진행하고,
    각 trial을 시간 슬라이스로 변환합니다.
    
    Returns:
      X_de_train, X_de_test, X_psd_train, X_psd_test: 각각의 배열, 
          shape: (총 샘플수, input_shape) — 
          input_shape는 data_mode에 따라 다름:
              "smooth": (62, 1, 4)
              "raw": (4, 8, 8, 1)
      y_train, y_test: 각 시간 슬라이스에 대응하는 one-hot 라벨, shape: (총 샘플수, 4)
    """
    from tensorflow.keras.utils import to_categorical
    
    if data_mode == "smooth":
        de_trials, psd_trials, label_list, _ = load_seediv_data(base_dirs)
    elif data_mode == "raw":
        raw_trials, label_list, _ = load_seediv_raw_data(base_dirs, raw_key="raw")
        # raw 데이터는 하나의 modality이므로, 동일 데이터를 두 branch로 사용
        de_trials = raw_trials
        psd_trials = raw_trials
    else:
        raise ValueError("data_mode must be 'smooth' or 'raw'")
    
    all_labels = to_categorical(np.array(label_list), num_classes=4)
    
    # trial 단위 train/test split (전체 trial 수 기준)
    indices = np.arange(len(de_trials))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42, stratify=np.array(label_list))
    
    de_train_trials = [de_trials[i] for i in train_idx]
    de_test_trials  = [de_trials[i] for i in test_idx]
    psd_train_trials = [psd_trials[i] for i in train_idx]
    psd_test_trials  = [psd_trials[i] for i in test_idx]
    y_train_trials = all_labels[train_idx]
    y_test_trials  = all_labels[test_idx]
    
    # 각 trial을 시간 슬라이스로 변환 (이미 raw mode: 각 파일은 이미 시간 단위 샘플)
    X_de_train = split_trials_list_to_time_slices(de_train_trials)
    X_de_test  = split_trials_list_to_time_slices(de_test_trials)
    X_psd_train = split_trials_list_to_time_slices(psd_train_trials)
    X_psd_test  = split_trials_list_to_time_slices(psd_test_trials)
    
    # trial별 라벨 복제 (각 trial의 시간 길이에 따라)
    y_train = replicate_labels_for_trials(de_train_trials, y_train_trials)
    y_test  = replicate_labels_for_trials(de_test_trials, y_test_trials)
    
    return X_de_train, X_de_test, X_psd_train, X_psd_test, y_train, y_test

# -----------------------------------------------------------------------------
# SEED-IV용 CNN 모델 생성 (입력 shape 결정: smooth → (62,1,4), raw → (4,8,8,1))
# -----------------------------------------------------------------------------
def create_seediv_cnn_model(input_shape, num_classes=4):
    """
    DE와 PSD 데이터를 각각 입력받아 CNN을 통해 특징을 추출한 후, 
    두 branch를 결합하여 감정을 분류하는 모델 생성.
    입력 shape는 data_mode에 따라 달라집니다.
    """
    # DE branch
    input_de = Input(shape=input_shape, name='input_de')
    x_de = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_first')(input_de)
    x_de = BatchNormalization(axis=1)(x_de)
    x_de = MaxPooling2D(pool_size=(1, 2), data_format='channels_first')(x_de)
    x_de = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_first')(x_de)
    x_de = BatchNormalization(axis=1)(x_de)
    x_de = MaxPooling2D(pool_size=(1, 2), data_format='channels_first')(x_de)
    x_de = GlobalAveragePooling2D(data_format='channels_first')(x_de)
    
    # PSD branch
    input_psd = Input(shape=input_shape, name='input_psd')
    x_psd = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_first')(input_psd)
    x_psd = BatchNormalization(axis=1)(x_psd)
    x_psd = MaxPooling2D(pool_size=(1, 2), data_format='channels_first')(x_psd)
    x_psd = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_first')(input_psd)
    x_psd = BatchNormalization(axis=1)(x_psd)
    x_psd = MaxPooling2D(pool_size=(1, 2), data_format='channels_first')(x_psd)
    x_psd = GlobalAveragePooling2D(data_format='channels_first')(x_psd)
    
    merged = concatenate([x_de, x_psd], name='merged_features')
    fc = Dense(128, activation='relu')(merged)
    fc = Dropout(0.5)(fc)
    output = Dense(num_classes, activation='softmax', name='output')(fc)
    
    model = Model(inputs=[input_de, input_psd], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -----------------------------------------------------------------------------
# SEED-IV 데이터 파인튜닝 (eeg_feature_smooth 또는 raw data) 예시
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # data_mode: "smooth" (preprocessed data) 또는 "raw" (raw data)
    data_mode = "smooth"  # 원하는 모드로 설정 ("smooth" 또는 "raw")
    
    # data_mode에 따라 base_dirs 자동 선택
    if data_mode == "smooth":
        base_dirs = [
            "/home/bcml1/2025_EMOTION/SEED_IV/eeg_feature_smooth/1_npy_sample",
            "/home/bcml1/2025_EMOTION/SEED_IV/eeg_feature_smooth/2_npy_sample",
            "/home/bcml1/2025_EMOTION/SEED_IV/eeg_feature_smooth/3_npy_sample"
        ]
        # smooth 데이터의 경우 trial의 샘플 shape는 (62,1,4)
        model_input_shape = (62, 1, 4)
    elif data_mode == "raw":
        base_dirs = [
            "/home/bcml1/2025_EMOTION/SEED_IV/eeg_raw_data/1_npy_sample",
            "/home/bcml1/2025_EMOTION/SEED_IV/eeg_raw_data/2_npy_sample",
            "/home/bcml1/2025_EMOTION/SEED_IV/eeg_raw_data/3_npy_sample"
        ]
        # raw 데이터의 경우 mapping 후 샘플 shape는 (4,8,8,1)
        model_input_shape = (4, 8, 8, 1)
    else:
        raise ValueError("data_mode must be 'smooth' or 'raw'")
    
    # SEED-IV 파인튜닝용 데이터 로드: 각 trial을 시간 슬라이스로 변환하여 샘플들로 만듦.
    X_de_train, X_de_test, X_psd_train, X_psd_test, y_train, y_test = load_seediv_data_finetune(base_dirs, test_size=0.2, data_mode=data_mode)
    print("SEED-IV 데이터 로드 완료:")
    print("X_de_train.shape:", X_de_train.shape)
    print("X_psd_train.shape:", X_psd_train.shape)
    print("y_train.shape:", y_train.shape)
    
    # 모델 생성 (입력 shape: model_input_shape)
    model_seed = create_seediv_cnn_model(input_shape=model_input_shape, num_classes=4)
    model_seed.summary()
    
    # 모델 학습
    history_seed = model_seed.fit([X_de_train, X_psd_train], y_train,
                                  validation_data=([X_de_test, X_psd_test], y_test),
                                  epochs=20, batch_size=32)
    
    test_loss, test_acc = model_seed.evaluate([X_de_test, X_psd_test], y_test)
    print(f"SEED-IV 테스트 정확도: {test_acc * 100:.2f}%")
