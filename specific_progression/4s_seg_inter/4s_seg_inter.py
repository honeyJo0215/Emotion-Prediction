import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, f1_score, confusion_matrix
from sklearn.utils import class_weight
from scipy.signal import butter, filtfilt
import seaborn as sns

# TensorFlow 버전 확인
print(f"TensorFlow version: {tf.__version__}")  # TensorFlow 버전을 출력

# GPU 설정
gpus = tf.config.list_physical_devices('GPU')  # 사용 가능한 GPU 장치 확인
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # GPU 메모리 동적 할당 설정
            print(f"Set memory growth for GPU: {gpu}")  # 설정 확인 메시지 출력
    except RuntimeError as e:
        print(f"Failed to set memory growth: {e}")  # 설정 실패 시 에러 메시지 출력
else:
    print("No GPU devices found. Running on CPU.")  # GPU가 없을 경우 CPU로 실행

# Step 1: 데이터 로드 함수 (세션 정보 제거)
def load_eeg_data(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]  # .npy 파일 목록 확인
    if not files:
        raise FileNotFoundError("No .npy files found in the directory.")  # 파일이 없을 경우 에러 발생

    data_list, labels, participants = [], [], []  # 데이터를 저장할 리스트 초기화
    for file in files:
        basename = os.path.basename(file)  # 파일 이름 추출
        parts = basename.split('_')  # 파일 이름 분리
        if len(parts) < 3:
            raise ValueError(f"Unexpected file naming format: {basename}")  # 파일 이름 형식 확인
        
        participant_id = parts[0]  # 참가자 ID 추출
        label_str = parts[1]  # 라벨 추출

        label = 1 if 'positive' in label_str else 0  # 라벨이 'positive'인지 확인
        raw_data = np.load(file)  # .npy 파일 데이터 로드

        if len(raw_data.shape) != 3:
            raise ValueError(f"Unexpected data shape {raw_data.shape} in file: {file}")  # 데이터 형식 확인

        data_list.append(raw_data)  # 데이터를 리스트에 추가
        labels.extend([label] * raw_data.shape[0])  # 라벨 추가
        participants.extend([participant_id] * raw_data.shape[0])  # 참가자 ID 추가

    return np.concatenate(data_list), np.array(labels), np.array(participants)  # 데이터, 라벨, 참가자 ID 반환

# Step 2: 밴드패스 필터링 함수
def bandpass_filter(data, lowcut=4, highcut=45, fs=128, order=5):
    nyquist = 0.5 * fs  # 나이퀴스트 주파수 계산
    low = lowcut / nyquist  # 낮은 주파수 설정
    high = highcut / nyquist  # 높은 주파수 설정
    b, a = butter(order, [low, high], btype='band')  # 필터 계수 계산

    if data.shape[-2] <= max(len(b), len(a)):
        raise ValueError("The length of the input vector must be greater than the filter's padding length.")  # 데이터 길이 확인

    return filtfilt(b, a, data, axis=-2)  # 필터링 수행

# Step 3: 데이터 정규화 함수
def normalize_eeg_data(data):
    return (data - np.mean(data, axis=-2, keepdims=True)) / np.std(data, axis=-2, keepdims=True)  # 평균 0, 표준편차 1로 정규화

# Step 4: CNN 모델 정의
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),  # 입력 데이터 형태 지정
        Conv1D(32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001)),  # 첫 번째 Conv1D 레이어
        BatchNormalization(),  # 배치 정규화
        MaxPooling1D(pool_size=2, strides=2, padding='same'),  # MaxPooling
        Dropout(0.3),  # 드롭아웃 추가

        Conv1D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001)),  # 두 번째 Conv1D 레이어
        BatchNormalization(),  # 배치 정규화
        MaxPooling1D(pool_size=2, strides=2, padding='same'),  # MaxPooling
        Dropout(0.3),  # 드롭아웃 추가

        Flatten(),  # 평탄화
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),  # 첫 번째 Dense 레이어
        Dropout(0.5),  # 드롭아웃 추가
        Dense(num_classes, activation='softmax')  # 출력 레이어
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # 모델 컴파일
    return model  # 모델 반환

# Step 5: 데이터 전처리 함수 (세션 정보 제거)
def preprocess_eeg_data(directory):
    X, y, participants = load_eeg_data(directory)  # 데이터 로드
    try:
        X = bandpass_filter(X)  # 밴드패스 필터 적용
    except ValueError as e:
        print(f"Bandpass filter error: {e}")  # 필터링 에러 출력
        raise
    X = normalize_eeg_data(X)  # 데이터 정규화
    return X, y, participants  # 전처리된 데이터 반환

# Step 6: 모델 학습 함수
def train_model_inter_subject_split(X, y, model_path, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)  # 데이터 분할
    num_classes = len(np.unique(y))  # 클래스 수 계산
    y_train_categorical = to_categorical(y_train, num_classes)  # 라벨을 원-핫 인코딩
    y_test_categorical = to_categorical(y_test, num_classes)  # 테스트 라벨 원-핫 인코딩
    input_shape = X_train.shape[1:]  # 입력 데이터 형태 계산

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)  # 클래스 가중치 계산
    class_weights_dict = dict(enumerate(class_weights))  # 가중치를 딕셔너리로 저장

    model = create_cnn_model(input_shape, num_classes)  # CNN 모델 생성
    from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)  # 조기 종료 설정
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1)  # TensorBoard 설정

    history = model.fit(
        X_train, y_train_categorical, epochs=100, batch_size=32, verbose=1,
        validation_split=0.2, callbacks=[early_stopping, tensorboard], class_weight=class_weights_dict
    )  # 모델 학습

    predictions = model.predict(X_test)  # 테스트 데이터 예측
    test_accuracy = model.evaluate(X_test, y_test_categorical, verbose=0)[1]  # 테스트 정확도 평가
    print(f"Test Accuracy: {test_accuracy:.4f}")  # 정확도 출력

    if num_classes == 2:
        roc_auc = roc_auc_score(y_test, predictions[:, 1])  # 이진 분류일 경우 ROC AUC 계산
        print(f"ROC AUC Score: {roc_auc:.4f}")  # AUC 출력
    else:
        roc_auc = roc_auc_score(y_test_categorical, predictions, multi_class='ovr')  # 다중 클래스일 경우 ROC AUC 계산
        print(f"ROC AUC Score: {roc_auc:.4f}")  # AUC 출력

    y_pred_labels = np.argmax(predictions, axis=1)  # 예측 결과 라벨로 변환
    f1 = f1_score(y_test, y_pred_labels, average='weighted')  # F1 점수 계산
    print(f"F1 Score: {f1:.4f}")  # F1 점수 출력

    cm = confusion_matrix(y_test, y_pred_labels)  # 혼동 행렬 계산
    plt.figure(figsize=(6, 4))  # 플롯 크기 설정
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])  # 혼동 행렬 시각화
    plt.xlabel('Predicted')  # x축 라벨 설정
    plt.ylabel('Actual')  # y축 라벨 설정
    plt.title('Confusion Matrix')  # 플롯 제목 설정
    plt.show()  # 플롯 표시

    # Classification Report 저장
    report = classification_report(y_test, y_pred_labels, target_names=['Negative', 'Positive'], digits=4)
    with open("classification_report.txt", "w") as f:
        f.write("Classification Report\n")
        f.write(report)
        print("Classification report saved to classification_report.txt")

    model.save(model_path)  # 모델 저장
    print(f"Model saved at {model_path}")  # 저장 메시지 출력

# Main execution block
def main():
    data_dir = "/home/bcml1/2025_EMOTION/DEAP_EEG/4s_seg_conv_ch_BPF"  # 데이터 디렉토리 경로
    model_path = "4s_seg_inter/4s_seg_inter_subject.h5"# 모델 저장 경로

    X, y, participants = preprocess_eeg_data(data_dir)  # 데이터 전처리
    train_model_inter_subject_split(X, y, model_path, test_size=0.2)  # 모델 학습 및 평가

if __name__ == "__main__":
    main()
