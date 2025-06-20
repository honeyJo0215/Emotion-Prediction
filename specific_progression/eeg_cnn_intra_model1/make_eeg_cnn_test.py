import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# GPU 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # 메모리 동적 할당
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]  # 메모리 제한
            )
    except RuntimeError as e:
        print(e)

def load_data_with_leave_one_out(data_dir):
    """
    데이터를 로드하고 Leave-One-Out 방식으로 학습 및 테스트 데이터를 분리.
    
    Parameters:
        data_dir (str): EEG 데이터 경로.

    Returns:
        data (np.ndarray): EEG 데이터 배열.
        labels (np.ndarray): 라벨 배열.
        groups (list): 개인별 그룹 정보.
    """
    data = []
    labels = []
    groups = []  # 개인 그룹 정보

    for file_name in sorted(os.listdir(data_dir)):
        if file_name.endswith("_bands.npy"):
            file_path = os.path.join(data_dir, file_name)

            # 개인 ID 추출 (예: 파일 이름의 첫 번째 부분)
            subject_id = file_name.split("_")[0]

            # 데이터 로드
            try:
                band_data = np.load(file_path)  # (bands, samples, channels)
                alpha_band = band_data[1]  # Alpha band 데이터만 선택
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
                continue

            label = 1 if "positive" in file_name else 0

            for ch_idx in range(alpha_band.shape[1]):
                data.append(alpha_band[:, ch_idx])  # 채널별 데이터 추가
                labels.append(label)
                groups.append(subject_id)  # 개인 ID 추가

    return np.array(data), np.array(labels), np.array(groups)

def build_cnn_model(input_shape):
    """
    CNN 모델 생성.
    
    Parameters:
        input_shape (tuple): 입력 데이터 형태.

    Returns:
        model (tf.keras.Model): 생성된 CNN 모델.
    """
    model = Sequential([
        Input(shape=input_shape),
        # Convolutional Layer 1
        Conv1D(32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        # Convolutional Layer 2
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        # Convolutional Layer 3
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        # Flatten Layer
        Flatten(),

        # Fully Connected Dense Layers
        Dense(256, activation='relu'),
        Dropout(0.4),

        #결과 layer
        Dense(1, activation='sigmoid')  # 이진 분류
    ])
    return model

# 데이터 경로 및 로드
data_dir = "/home/bcml1/sigenv/eeg_band_split"  # 실제 데이터 경로로 변경
X, y, groups = load_data_with_leave_one_out(data_dir)

# Leave-One-Out Cross-Validation
group_kfold = LeaveOneGroupOut()
for train_idx, test_idx in group_kfold.split(X, y, groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 데이터 정규화
    X_train = X_train / np.max(np.abs(X_train), axis=1, keepdims=True)
    X_test = X_test / np.max(np.abs(X_test), axis=1, keepdims=True)

    # 차원 확장 (CNN 입력 형식)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # 모델 생성 및 학습
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_cnn_model(input_shape)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,  # 에포크 수 조정 가능
        batch_size=32,
        verbose=1
    )
    
    # 학습 결과 저장
model.save("eeg_cnn_model_mj.h5")
print("Model saved!")

 # 결과 평가
model = tf.keras.models.load_model("eeg_cnn_model_mj.h5")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# 테스트 데이터 예측
test_sample = X_test[0]  # 임의의 샘플
test_sample = np.expand_dims(test_sample, axis=0)  # 배치 차원 추가
prediction = model.predict(test_sample)

# 결과 출력
print(f"Prediction (0=negative, 1=positive): {prediction[0][0]}")