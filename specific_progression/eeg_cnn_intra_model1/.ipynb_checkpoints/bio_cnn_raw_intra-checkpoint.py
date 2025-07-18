import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, f1_score, confusion_matrix
from sklearn.utils import class_weight
from scipy.signal import butter, filtfilt

# TensorFlow 버전 확인
print(f"TensorFlow version: {tf.__version__}")

# GPU 설정
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            try:
                # 최신 API 사용
                tf.config.set_memory_growth(gpu, True)
                print(f"Set memory growth for GPU: {gpu}")
            except AttributeError:
                # 이전 API 사용
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Set memory growth for GPU using experimental API: {gpu}")
    except RuntimeError as e:
        print(f"Failed to set memory growth: {e}")
else:
    print("No GPU devices found. Running on CPU.")

# Step 1: 데이터 로드 함수 (세션 정보 제거)
def load_eeg_data(directory):
    """
    지정된 디렉토리에서 EEG 데이터를 로드하고, 피험자별로 그룹화합니다.

    Parameters:
        directory (str): .npy 파일이 있는 디렉토리 경로.

    Returns:
        data_dict (dict): 피험자 ID를 키로 하고, 해당 피험자의 데이터와 라벨을 값으로 하는 딕셔너리.
    """
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]
    if not files:
        raise FileNotFoundError("No .npy files found in the directory.")

    data_dict = {}
    for file in files:
        # Extract participant ID and label from filename
        basename = os.path.basename(file)
        parts = basename.split('_')
        if len(parts) < 3:
            raise ValueError(f"Unexpected file naming format: {basename}")

        participant_id = parts[0]
        label_str = parts[1]
        # 'FB'는 세션 식별자가 아니므로 무시

        if 'positive' in label_str.lower():
            label = 1
        elif 'negative' in label_str.lower():
            label = 0
        else:
            raise ValueError(f"File name does not indicate label: {file}")

        raw_data = np.load(file)

        # Check if raw_data has the expected shape (samples, channels, timesteps)
        if len(raw_data.shape) != 3:
            raise ValueError(f"Unexpected data shape {raw_data.shape} in file: {file}")

        # Initialize dictionary for participant if not exists
        if participant_id not in data_dict:
            data_dict[participant_id] = {'data': [], 'labels': []}

        # Append raw data and labels
        data_dict[participant_id]['data'].append(raw_data)
        data_dict[participant_id]['labels'].extend([label] * raw_data.shape[0])

    # Concatenate data for each participant
    for participant in data_dict:
        data_dict[participant]['data'] = np.concatenate(data_dict[participant]['data'])
        data_dict[participant]['labels'] = np.array(data_dict[participant]['labels'])

    return data_dict


# Step 2: 밴드패스 필터링 함수
def bandpass_filter(data, lowcut=4, highcut=45, fs=128, order=5):
    """
    EEG 데이터에 밴드패스 필터를 적용합니다.

    Parameters:
        data (np.ndarray): EEG 데이터.
        lowcut (float): 저주파 컷오프 주파수.
        highcut (float): 고주파 컷오프 주파수.
        fs (float): 샘플링 주파수.
        order (int): 필터 차수.

    Returns:
        filtered_data (np.ndarray): 필터링된 EEG 데이터.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)

# Step 3: 데이터 정규화 함수
def normalize_eeg_data(data):
    """
    EEG 데이터를 정규화합니다.

    Parameters:
        data (np.ndarray): EEG 데이터.

    Returns:
        normalized_data (np.ndarray): 정규화된 EEG 데이터.
    """
    return (data - np.mean(data, axis=-1, keepdims=True)) / np.std(data, axis=-1, keepdims=True)

# Step 4: 데이터 증강 함수
def augment_eeg_data(X, y):
    # Gaussian 노이즈 추가
    noise = np.random.normal(0, 0.01, X.shape)
    X_noise = X + noise

    # 시간 이동 (shift)
    shift = np.roll(X, shift=10, axis=1)  # timesteps 축을 기준으로 10 타임스텝 이동
    X_shift = shift

    # 윈도우 슬라이싱 (윈도우 크기 80%로 자르기)
    X_sliced = X[:, :int(X.shape[1] * 0.8), :]

    # 증강된 데이터와 라벨 결합
    X_augmented = np.concatenate([X, X_noise, X_shift, X_sliced], axis=0)
    y_augmented = np.concatenate([y, y, y, y], axis=0)
    
    return X_augmented, y_augmented

# Step 5: 클래스 불균형 처리 함수 (SMOTE)
def balance_classes(X, y):
    smote = SMOTE(random_state=42)
    X_reshaped = X.reshape(X.shape[0], -1)
    X_balanced, y_balanced = smote.fit_resample(X_reshaped, y)
    X_balanced = X_balanced.reshape(-1, X.shape[1], X.shape[2])
    return X_balanced, y_balanced

# Step 4: CNN 모델 정의 (수정)
def create_cnn_model(input_shape, num_classes):
    """
    CNN 모델을 생성합니다.

    Parameters:
        input_shape (tuple): 입력 데이터 형상 (timesteps, channels).
        num_classes (int): 분류할 클래스 수.

    Returns:
        model (tf.keras.Model): 컴파일된 CNN 모델.
    """
    model = Sequential([
        Input(shape=input_shape),
        
        # Conv1D + BatchNormalization + Dropout
        Conv1D(16, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, strides=2, padding='same'),
        Dropout(0.3),  # 드롭아웃 비율을 0.3로 변경

        Conv1D(32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, strides=2, padding='same'),
        Dropout(0.3),  # 드롭아웃 비율을 0.3로 변경

        # Fully Connected Layer
        Flatten(),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.3),  # Fully Connected Layer에서도 드롭아웃 비율 증가
        Dense(num_classes, activation='softmax')
    ])
    
    # 모델 컴파일
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 5: 데이터 전처리 함수 (세션 정보 제거)
def preprocess_eeg_data(directory):
    """
    Preprocess EEG data: bandpass filter and normalize.

    Parameters:
        directory (str): Path to the directory containing .npy files.

    Returns:
        preprocessed_dict (dict): Dictionary with preprocessed data and labels per participant.
    """
    # 데이터 로드
    data_dict = load_eeg_data(directory)
    print("데이터 로드 완료.")

    # 전처리된 데이터를 저장할 딕셔너리 초기화
    preprocessed_dict = {}
    
    for participant, content in data_dict.items():
        X = content['data']
        y = content['labels']
        
        # 밴드패스 필터링
        X = bandpass_filter(X)

        # 데이터 정규화
        X = normalize_eeg_data(X)

        # 데이터 축 재정렬: (samples, channels, timesteps) -> (samples, timesteps, channels)
        X = np.transpose(X, (0, 2, 1))

        # 전처리된 데이터 저장
        preprocessed_dict[participant] = {'data': X, 'labels': y}
        print(f"Participant {participant} 전처리 완료.")

    return preprocessed_dict

# Step 6: Intra-Session (Intra-Subject) 교차 검증 및 모델 학습
def train_model_intra_subject_split(preprocessed_dict, model_path, test_size=0.2, random_state=42):
    """
    Intra-Session 교차 검증을 사용하여 CNN 모델을 학습합니다.
    각 피험자별로 훈련 세트와 테스트 세트를 분리하여 모델을 학습하고 평가합니다.

    Parameters:
        preprocessed_dict (dict): 전처리된 피험자별 데이터와 라벨.
        model_path (str): 학습된 모델을 저장할 경로.
        test_size (float): 테스트 세트의 비율 (예: 0.3 for 30%).
        random_state (int): 랜덤 시드.
    """
    # 결과 파일 초기화
    result_filename = "bio_cnn_eeg_results_raw_intra.txt"
    with open(result_filename, "w") as result_file:
        result_file.write("Intra-Session(Subject) Cross-Validation Results\n")
    
    # Initialize lists to collect metrics
    all_accuracy = []
    all_roc_auc = []
    all_f1 = []
    all_cm = []
    prediction_details = []

    # Iterate over each participant
    for participant, content in preprocessed_dict.items():
        X = content['data']
        y = content['labels']
        num_classes = len(np.unique(y))
        print(f"\nParticipant: {participant}, Samples: {X.shape[0]}, Classes: {num_classes}")

        # Initialize StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

        for train_index, test_index in sss.split(X, y):
            print(f"  Splitting data: Train size = {len(train_index)}, Test size = {len(test_index)}")

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            y_train_cat = to_categorical(y_train, num_classes)
            y_test_cat = to_categorical(y_test, num_classes)

            # Compute class weights
            class_weights_vals = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            class_weights_dict = dict(enumerate(class_weights_vals))
            print(f"    Class weights: {class_weights_dict}")

            # Create a new model instance
            model = create_cnn_model(input_shape=X_train.shape[1:], num_classes=num_classes)

            # Define callbacks
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True
            )

            # ModelCheckpoint 수정: .h5 형식으로 저장, save_weights_only=True 추가
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{model_path}_participant_{participant}.weights.h5",  # 수정: .weights.h5로 저장
                monitor='val_loss',
                save_best_only=True,
                verbose=1,
                save_weights_only=True  # 가중치만 저장
            )

            # Train the model
            history = model.fit(
                X_train,
                y_train_cat,
                epochs=100,
                batch_size=16,
                verbose=1,  # Set to 1 for more detailed logs
                validation_split=0.2,
                callbacks=[early_stopping, checkpoint],
                class_weight=class_weights_dict
            )

            # Load the best model
            model.load_weights(f"{model_path}_participant_{participant}.weights.h5")
            model.save(f"{model_path}_participant_{participant}.h5")  # 전체 모델을 .h5로 저장
            
            # Evaluate the model
            predictions = model.predict(X_test)
            test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)[1]
            all_accuracy.append(test_accuracy)

            # Calculate ROC AUC
            if num_classes == 2:
                roc_auc = roc_auc_score(y_test, predictions[:,1])
            else:
                roc_auc = roc_auc_score(y_test_cat, predictions, multi_class='ovr')
            all_roc_auc.append(roc_auc)

            # Calculate F1 score
            y_pred_labels = np.argmax(predictions, axis=1)
            f1 = f1_score(y_test, y_pred_labels, average='weighted')
            all_f1.append(f1)

            # Calculate Confusion Matrix
            cm = confusion_matrix(y_test, y_pred_labels)
            all_cm.append(cm)

            # Calculate FPR and TPR for binary classification
            if num_classes == 2:
                if cm.shape == (2, 2):
                    TN, FP, FN, TP = cm.ravel()
                    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
                    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
                else:
                    # Handle cases where one class might be missing in predictions
                    fpr = 'Undefined (One class missing in predictions)'
                    tpr = 'Undefined (One class missing in predictions)'
            else:
                # 다중 클래스의 경우, 평균 FPR, TPR을 계산할 수 없습니다.
                fpr = 'N/A'
                tpr = 'N/A'

            # Generate classification report
            report = classification_report(y_test, y_pred_labels, digits=4)

            # Write to result file
            with open(result_filename, "a") as result_file:
                result_file.write(f"\nParticipant {participant} Classification Report:\n\n")
                result_file.write("Detailed Classification Report:\n")
                result_file.write(report)
                result_file.write(f"\nFalse Positive Rate (FPR): {fpr}\n")
                result_file.write(f"True Positive Rate (TPR): {tpr}\n")

            print(f"    Participant {participant} classification report saved.")
            
    # Aggregate Metrics
    avg_accuracy = np.mean(all_accuracy)
    std_accuracy = np.std(all_accuracy)
    avg_roc_auc = np.mean(all_roc_auc)
    std_roc_auc = np.std(all_roc_auc)
    avg_f1 = np.mean(all_f1)
    std_f1 = np.std(all_f1)

    print("\n--- Intra-Session Cross-Validation Results ---")
    print(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Average ROC AUC Score: {avg_roc_auc:.4f} ± {std_roc_auc:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f} ± {std_f1:.4f}")

    # Aggregate Confusion Matrices
    total_cm = np.sum(all_cm, axis=0)
    print("\nAggregated Confusion Matrix:")
    print(total_cm)

    # Plot Aggregated Confusion Matrix
    plt.figure(figsize=(6,4))
    plt.imshow(total_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Aggregated Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(total_cm.shape[0])
    plt.xticks(tick_marks, ['Negative', 'Positive'], rotation=45)
    plt.yticks(tick_marks, ['Negative', 'Positive'])

    # Loop over data dimensions and create text annotations.
    thresh = total_cm.max() / 2.
    for i in range(total_cm.shape[0]):
        for j in range(total_cm.shape[1]):
            plt.text(j, i, format(total_cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if total_cm[i, j] > thresh else "black")

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

    # Save Aggregated Confusion Matrix as Image
    plt.figure(figsize=(6,4))
    plt.imshow(total_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Aggregated Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(total_cm.shape[0])
    plt.xticks(tick_marks, ['Negative', 'Positive'], rotation=45)
    plt.yticks(tick_marks, ['Negative', 'Positive'])

    # Loop over data dimensions and create text annotations.
    thresh = total_cm.max() / 2.
    for i in range(total_cm.shape[0]):
        for j in range(total_cm.shape[1]):
            plt.text(j, i, format(total_cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if total_cm[i, j] > thresh else "black")

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig("aggregated_confusion_matrix_bio_cnn_raw_intra.png")
    plt.show()

    print(f"Aggregated confusion matrix image saved to 'aggregated_confusion_matrix_bio_cnn_raw_intra.png'")

    # Optionally, append aggregated metrics to the result file
    with open(result_filename, "a") as result_file:
        result_file.write("\n--- Intra-Session Cross-Validation Summary ---\n")
        result_file.write(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}\n")
        result_file.write(f"Average ROC AUC Score: {avg_roc_auc:.4f} ± {std_roc_auc:.4f}\n")
        result_file.write(f"Average F1 Score: {avg_f1:.4f} ± {std_f1:.4f}\n")
        result_file.write(f"\nAggregated Confusion Matrix:\n{total_cm}\n")

    print(f"Aggregated results saved to '{result_filename}'")

# Main execution block
def main():
    # 데이터 경로 설정
    data_dir = "/home/bcml1/2025_EMOTION/DEAP_EEG/ch_BPF"  # 실제 데이터가 있는 디렉토리로 업데이트
    model_path = "cnn_model_intra_session"  # 모델 저장 경로 (확장자 제외)

    print("Loading and preprocessing EEG data...")
    preprocessed_dict = preprocess_eeg_data(data_dir)

    print("Starting Intra-Session Cross-Validation...")
    train_model_intra_subject_split(preprocessed_dict, model_path, test_size=0.2, random_state=42)  # 80:20 분할

if __name__ == "__main__":
    main()

