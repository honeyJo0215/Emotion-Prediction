import numpy as np
from sklearn.model_selection import LeaveOneOut
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import os

# TensorFlow 로그 레벨 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# GPU 설정
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Failed to set memory growth: {e}")
else:
    print("No GPU devices found. Running on CPU.")

def load_data():
    # 데이터 파일 로드
    data_dir = "/home/bcml1/sigenv/eeg_band_split"
    
    # 파일 로드 및 병합
    files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    X_list, y_list, participants_list = [], [], []

    for file in files:
        data = np.load(os.path.join(data_dir, file))
        # 데이터 구조에 맞춰 추가 (예: 파일명에서 라벨 추출)
        X_list.append(data)
        # y와 participants는 파일명 규칙을 기반으로 생성 필요
        # 여기는 파일명에서 라벨이나 참여자 정보를 추출하는 로직 추가
        # 예: "s01_negative_sample_4_bands.npy" -> participant = s01, label = negative
        participant = file.split('_')[0]  # 첫 번째 정보
        label = file.split('_')[1]       # 두 번째 정보
        participants_list.append(participant)
        y_list.append(label)

    # 배열로 변환
    X = np.concatenate(X_list, axis=0)  # 데이터를 하나의 배열로 병합
    y = np.array(y_list)
    participants = np.array(participants_list)
    
    return X, y, participants

# CNN 모델 정의
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
# 테스트 데이터 예측 함수
def predict_test_samples(model, X_test, y_test):
    for i, test_sample in enumerate(X_test):
        test_sample_expanded = np.expand_dims(test_sample, axis=0)  # 배치 차원 추가
        prediction = model.predict(test_sample_expanded)
        predicted_class = np.argmax(prediction, axis=1)[0]  # 예측된 클래스
        true_class = np.argmax(y_test[i])  # 실제 클래스
        print(f"Sample {i}: Prediction (0=negative, 1=positive): {predicted_class}, True: {true_class}")
    X, y, participants = load_data()

    # 데이터 크기 확인
    print(f"X shape: {X.shape}, y shape: {y.shape}, participants shape: {participants.shape}")

    # participants 배열의 크기가 X와 y의 첫 번째 축과 일치하지 않는 경우 처리
    if participants.shape[0] != X.shape[0] or participants.shape[0] != y.shape[0]:
        raise ValueError("Mismatch in dimensions: 'participants', 'X', and 'y' must have the same number of samples.")

    # 문자열 라벨을 숫자형 라벨로 매핑
    unique_labels = np.unique(y)
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    y_int = np.array([label_to_int[label] for label in y])
    
    num_classes = len(unique_labels)
    input_shape = X.shape[1:]  # 데이터가 올바르게 reshape되었다고 가정
    y_categorical = to_categorical(y_int, num_classes=num_classes)

    loo = LeaveOneOut()
    accuracy_scores = []

    for train_index, test_index in loo.split(np.unique(participants)):
        train_participants = np.unique(participants)[train_index]
        test_participant = np.unique(participants)[test_index][0]

        # 참여자별 데이터 분리
        train_mask = np.isin(participants, train_participants)
        test_mask = participants == test_participant

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y_categorical[train_mask], y_categorical[test_mask]

        # 모델 생성 및 학습
        model = create_cnn_model(input_shape, num_classes)
        history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

        # 학습 과정 확인
        print("Training history:")
        print("Accuracy:", history.history['accuracy'])
        print("Loss:", history.history['loss'])

        # 모델 평가
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        accuracy_scores.append(test_accuracy)

        print(f"Test participant: {test_participant}, Test accuracy: {test_accuracy:.4f}")

        # 테스트 데이터 예측
        predict_test_samples(model, X_test, y_test)

    print(f"Mean accuracy across participants: {np.mean(accuracy_scores):.4f}")
