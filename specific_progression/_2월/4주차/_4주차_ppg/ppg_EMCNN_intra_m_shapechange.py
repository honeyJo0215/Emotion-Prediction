# 논문 그대로 구현한 코드
# Epochs: 300, Batch Size: 128, Optimizer: Adam (β1=0.9, β2=0.99)
# Loss Function: Cross-Entropy Loss
# Regularization: L2 Regularization (λ=0.001)
# Learning Rate: 0.001
# Intra-Subject Validation (7:3 비율로 데이터 분할)
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, DepthwiseConv1D, Dense, GlobalAveragePooling1D, concatenate, BatchNormalization, Activation, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

# === EMCNN 입력 데이터 형태 ===
# 이제 각 sample(세그먼트)의 shape은 (1280, 5): 1280 샘플의 시간축, 5개의 변환 채널
input_shape = (1280, 5)
num_classes = 4  # 감정 클래스 (HVHA, HVLA, LVLA, LVHA)

# === MobileNet 기반 Feature Extraction CNN ===
def emcnn_branch(input_layer):
    """ EMCNN의 각 branch에 해당하는 CNN 블록 (MobileNet 기반) """
    x = Conv1D(8, kernel_size=7, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.001))(input_layer)
    
    # Depthwise Separable Convolution 적용
    x = DepthwiseConv1D(kernel_size=7, strides=4, padding='same', depth_multiplier=1, depthwise_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(16, kernel_size=1, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.001))(x)

    x = DepthwiseConv1D(kernel_size=7, strides=2, padding='same', depth_multiplier=1, depthwise_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(32, kernel_size=1, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.001))(x)

    x = DepthwiseConv1D(kernel_size=7, strides=4, padding='same', depth_multiplier=1, depthwise_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(64, kernel_size=1, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.001))(x)

    x = DepthwiseConv1D(kernel_size=7, strides=2, padding='same', depth_multiplier=1, depthwise_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(128, kernel_size=1, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.001))(x)

    # Global Average Pooling 적용
    x = GlobalAveragePooling1D()(x)
    return x

# === EMCNN 전체 모델 ===
def build_emcnn():
    # 기존에는 입력 shape이 (5, 1280)였으나, 여기서는 (1280, 5)로 재설계
    inputs = Input(shape=input_shape)  # 각 sample의 shape: (1280, 5)

    # 이제 Lambda Layer를 사용하여 각 채널(branch)를 추출합니다.
    # 입력의 shape는 (batch, 1280, 5)에서, x[:, :, 0]은 첫 번째 채널 (branch)을 의미
    identity_branch = emcnn_branch(Lambda(lambda x: tf.expand_dims(x[:, :, 0], axis=-1))(inputs))  # (batch, 1280, 1)
    smooth_branch_s2 = emcnn_branch(Lambda(lambda x: tf.expand_dims(x[:, :, 1], axis=-1))(inputs))
    smooth_branch_s3 = emcnn_branch(Lambda(lambda x: tf.expand_dims(x[:, :, 2], axis=-1))(inputs))
    downsample_branch_d2 = emcnn_branch(Lambda(lambda x: tf.expand_dims(x[:, :, 3], axis=-1))(inputs))
    downsample_branch_d3 = emcnn_branch(Lambda(lambda x: tf.expand_dims(x[:, :, 4], axis=-1))(inputs))

    # === Branch Concatenation ===
    merged_features = concatenate([identity_branch, smooth_branch_s2, smooth_branch_s3, downsample_branch_d2, downsample_branch_d3])

    # === Fully Connected Layers (L2 Regularization 포함) ===
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(merged_features)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
    output = Dense(num_classes, activation='softmax')(x)  # 감정 클래스 예측

    # === 모델 생성 ===
    model = Model(inputs=inputs, outputs=output)
    return model

# -------------------------------
# Intra-Subject Cross Validation 학습
# -------------------------------
def train_model_for_subject(subject_id, X, y, num_classes=4, epochs=300, batch_size=128):
    print(f"Subject {subject_id} 학습 시작")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    result_dir = f"EMCNN_intra_Results_reshape/{subject_id}"
    os.makedirs(result_dir, exist_ok=True)

    model = build_emcnn()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.99),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_test, y_test), verbose=1)

    model.save(os.path.join(result_dir, f"{subject_id}_model.keras"))
    np.save(os.path.join(result_dir, f"{subject_id}_history.npy"), history.history)

    # ✅ 예측 수행
    y_pred = np.argmax(model.predict(X_test), axis=1)

    # ✅ Confusion Matrix 저장
    plt.figure()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title(f"{subject_id} Confusion Matrix")
    plt.savefig(os.path.join(result_dir, f"{subject_id}_confusion_matrix.png"))
    plt.close()

    # ✅ Loss 그래프 저장
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(result_dir, f"{subject_id}_loss_output.png"))
    plt.close()

    # ✅ Accuracy 그래프 저장
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(result_dir, f"{subject_id}_accuracy_output.png"))
    plt.close()

    # ✅ Classification Report + Confusion Matrix를 sXX_evaluation.txt에 저장
    eval_path = os.path.join(result_dir, f"{subject_id}_evaluation.txt")
    with open(eval_path, 'w') as f:
        f.write(classification_report(y_test, y_pred))
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, y_pred)))

# -------------------------------
# 피실험자 데이터 로드 및 학습 실행
# -------------------------------
def load_data_for_subject(data_dir, subject_id):
    """ 특정 피실험자의 데이터만 불러오는 함수 """
    file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith(subject_id)]
    X, y = [], []

    for file_path in file_paths:
        data = np.load(file_path)  # 원래 shape: (51, 5, 1280)
        # np.transpose로 축 변경: (51, 5, 1280) -> (51, 1280, 5)
        data = np.transpose(data, (0, 2, 1))
        label = int(file_path.split('_')[-1].split('.')[0])
        X.append(data)
        y.append(np.full((data.shape[0],), label))

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y

# =============================================================================
# PPG 데이터 로딩 및 증강(1초의 데이터를 n초 단위로 만들기)
# =============================================================================
def load_ppg_data_with_sliding_window(ppg_data_dir,
                                        window_size=10, step=1):
    """
    PPG 데이터를 로딩한 후, 각 파일이 (60, 5, 128) (즉, 60개의 1초 segment, 5채널, 128 샘플)라면
    먼저 연속 신호 (7680, 5)로 결합하고, 이를 window_size(초)와 step(초)에 따라 슬라이딩 윈도우로 병합합니다.
    또한, 증강(오버샘플링)을 통해 클래스 균형도 맞춥니다.
    
    Parameters:
    - data_dir: PPG 데이터 파일들이 저장된 디렉토리.
    - window_size: 병합할 segment 길이 (초 단위). 예를 들어 10이면 10초짜리 데이터를 만듦.
    - step: 슬라이딩 윈도우 이동 간격 (초 단위).
    
    Returns:
    - subject_data: dictionary, 각 subject별로 {'X': merged PPG 데이터, 'y': 라벨}를 담고 있음.
      최종 데이터의 shape는 (num_windows, 5, window_size*128)가 됩니다.
    """
    def sliding_window_ppg(continuous_data, window_size, step):
        """
        continuous_data: numpy array, shape=(total_samples, channels)
            예를 들어 (7680, 5) – 연속 신호.
        window_size: 초 단위, 예: 10 → 10*128 = 1280 샘플.
        step: 초 단위, 예: 1 → 128 샘플.
        
        반환: numpy array, shape=(num_windows, window_size*128, channels)
        """
        total_samples = continuous_data.shape[0]
        win_length = window_size * 128  # 예: 10초면 1280 샘플
        step_length = step * 128
        windows = []
        for start in range(0, total_samples - win_length + 1, step_length):
            windows.append(continuous_data[start:start+win_length, :])
        return np.array(windows)
    
    subject_data = {}
    file_paths = glob.glob(os.path.join(ppg_data_dir, "*.npy"))
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
        
        # 파일 로드: 여기서 파일 shape은 (60, 5, 128)로 가정
        data = np.load(file_path)  # shape: (60, 5, 128)
        # 먼저 1초 단위 데이터(60개의 segment)를 연속 신호로 결합
        # axis=0(각 1초 segment를 이어붙임) → (60*128, 5) = (7680, 5)
        continuous_data = np.concatenate(data, axis=0)
        # 슬라이딩 윈도우 적용 (continuous_data는 (7680, 5))
        windows = sliding_window_ppg(continuous_data, window_size, step)
        # windows shape: (num_windows, window_size*128, 5)
        # 필요에 따라 axis 순서를 변경할 수 있는데, 여기서는 채널 수를 그대로 두었습니다.
        
        if subject_id not in subject_data:
            subject_data[subject_id] = {'X': [], 'y': []}
        subject_data[subject_id]['X'].append(windows)
        # 각 윈도우마다 동일한 라벨 할당
        subject_data[subject_id]['y'].append(np.full((windows.shape[0],), label, dtype=np.int32))
    
    # subject별로 데이터 병합 및 클래스 불균형 보정
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

if __name__ == "__main__":
    data_dir = '/home/bcml1/2025_EMOTION/DEAP_PPG_10s_1soverlap_4label/'

    # ✅ 피실험자 s01 ~ s22에 대해 개별적으로 학습 수행
    for subject_id in [f"s{str(i).zfill(2)}" for i in range(1, 23)]:
        X, y = load_data_for_subject(data_dir, subject_id)
        train_model_for_subject(subject_id, X, y)
    