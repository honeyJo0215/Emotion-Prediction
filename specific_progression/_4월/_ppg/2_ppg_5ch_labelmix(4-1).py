import os
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
import logging

# -------------------------------------------------------------------------
# GPU 메모리 제한
# -------------------------------------------------------------------------
def limit_gpu_memory(memory_limit_mib=8000):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit_mib)]
            )
            logging.info(f"GPU memory limited to {memory_limit_mib} MiB.")
        except RuntimeError as e:
            logging.error(e)
    else:
        logging.info("No GPU available, using CPU.")

limit_gpu_memory(6000)

# -------------------------------------------------------------------------
# 라벨을 무작위로 섞어서 각 클래스가 균등하게 분포되도록 하는 함수 (테스트에서만 적용)
# -------------------------------------------------------------------------
def randomly_mix_labels(y, num_classes=4, seed=42):
    """
    입력 라벨 y의 총 샘플 수를 기준으로, 각 클래스가 균등하게 포함되도록 
    라벨을 재배분한 배열을 반환합니다.
    """
    np.random.seed(seed)
    total_samples = len(y)
    samples_per_class = total_samples // num_classes
    mixed_y = np.repeat(np.arange(num_classes), samples_per_class)
    remainder = total_samples - samples_per_class * num_classes
    if remainder > 0:
        extra = np.random.choice(num_classes, size=remainder, replace=True)
        mixed_y = np.concatenate([mixed_y, extra])
    np.random.shuffle(mixed_y)
    return mixed_y

# -----------------------------
# 1. Feature Extractor 정의 (각 브랜치에 사용)
# -----------------------------
def build_feature_extractor(input_length=None, num_channels=5):
    """
    입력 shape: (L, num_channels) – L: 시퀀스 길이, num_channels 채널 입력
    출력: 128 차원 feature 벡터 (GlobalAveragePooling1D 사용)
    """
    inputs = tf.keras.Input(shape=(input_length, num_channels))
    x = tf.keras.layers.Conv1D(filters=8, kernel_size=7, strides=1, padding='same')(inputs)
    x = tf.keras.layers.ReLU(max_value=6)(x)
    x = tf.keras.layers.SeparableConv1D(filters=16, kernel_size=7, strides=4, padding='same')(x)
    x = tf.keras.layers.ReLU(max_value=6)(x)
    x = tf.keras.layers.SeparableConv1D(filters=32, kernel_size=7, strides=2, padding='same')(x)
    x = tf.keras.layers.ReLU(max_value=6)(x)
    x = tf.keras.layers.SeparableConv1D(filters=64, kernel_size=7, strides=4, padding='same')(x)
    x = tf.keras.layers.ReLU(max_value=6)(x)
    x = tf.keras.layers.SeparableConv1D(filters=128, kernel_size=7, strides=2, padding='same')(x)
    x = tf.keras.layers.ReLU(max_value=6)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model

# -----------------------------
# 2. EMCNN 모델 정의 (tf.keras.Model subclassing)
# -----------------------------
class EMCNN(tf.keras.Model):
    def __init__(self, num_classes=4, smoothing_kernel_size=2, downsample_factor=2, input_length=None, num_channels=5):
        """
        num_classes: 분류 클래스 수  
        smoothing_kernel_size: 이동평균 필터의 커널 크기 (예: 2)  
        downsample_factor: 다운샘플링 비율 (예: 2)  
        input_length: 원본 시퀀스 길이  
        num_channels: 입력 채널 수 (여기서는 5)
        """
        super(EMCNN, self).__init__()
        self.num_classes = num_classes
        self.smoothing_kernel_size = smoothing_kernel_size
        self.downsample_factor = downsample_factor

        # 각 브랜치별 Feature Extractor (가중치는 공유하지 않음)
        self.feature_identity = build_feature_extractor(input_length=input_length, num_channels=num_channels)
        self.feature_smoothing = build_feature_extractor(input_length=input_length, num_channels=num_channels)
        self.feature_downsampling = build_feature_extractor(input_length=None, num_channels=num_channels)

        # Smoothing: 5채널 입력에 대해 각 채널별 이동평균 (groups=5 사용)
        self.smoothing_conv = tf.keras.layers.Conv1D(filters=num_channels, kernel_size=smoothing_kernel_size,
                                                     padding='same', use_bias=False, groups=num_channels)
        smoothing_weight = np.ones((smoothing_kernel_size, 1, num_channels), dtype=np.float32) / smoothing_kernel_size
        self.smoothing_conv.build((None, input_length, num_channels))
        self.smoothing_conv.set_weights([smoothing_weight])
        self.smoothing_conv.trainable = False

        # 분류를 위한 Fully Connected 레이어들
        self.fc1 = tf.keras.layers.Dense(128, activation=lambda x: tf.nn.relu6(x))
        self.fc2 = tf.keras.layers.Dense(64, activation=lambda x: tf.nn.relu6(x))
        self.fc3 = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training=False):
        """
        inputs: shape = (batch, L, num_channels)
        """
        # Transformation Stage:
        branch_identity = inputs
        branch_smooth = self.smoothing_conv(inputs)
        branch_down = inputs[:, ::self.downsample_factor, :]

        # Feature Extraction Stage:
        feat_identity = self.feature_identity(branch_identity)
        feat_smooth = self.feature_smoothing(branch_smooth)
        feat_down = self.feature_downsampling(branch_down)

        # 특징 벡터 연결
        features = tf.concat([feat_identity, feat_smooth, feat_down], axis=1)

        # Classification Stage:
        x = self.fc1(features)
        x = self.fc2(x)
        logits = self.fc3(x)
        return logits

# -----------------------------
# 3. 학습 및 평가 (서브젝트별 5-Fold Cross Validation + 테스트 시 라벨 무작위 재배분)
# -----------------------------
def train_model_for_subject_with_random_labels(subject_id, X, y, num_classes=4, epochs=50, batch_size=8,
                                               smoothing_kernel_size=2, downsample_factor=2,
                                               result_dir_base=''):
    """
    subject_id: 예, 's01'
    X: 입력 데이터, shape = (num_samples, L, num_channels)
    y: 원래의 정수 라벨, shape = (num_samples,)
    여기서는 훈련 시에는 원래 라벨을 그대로 사용하고, 테스트(검증) 단계에서만 라벨을 무작위 재배분하여
    각 클래스가 균등하게 분포되도록 합니다.
    즉, 정상적인 모델이라면 테스트 시 4개 클래스 각각 약 25%의 정확도가 나와야 합니다.
    각 fold의 classification report는 sklearn의 classification_report 형식으로 저장되며,
    모든 fold의 결과를 통합한 global classification report도 동일한 형식으로 저장됩니다.
    저장 폴더 이름은 result_dir_base에 mode (_1s 또는 _10s)가 포함됨.
    """
    subject_result_dir = os.path.join(result_dir_base, subject_id)
    os.makedirs(subject_result_dir, exist_ok=True)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_no = 1

    history_all = {}
    reports_all = {}
    conf_matrices_all = {}
    global_y_true = []
    global_y_pred = []

    for train_index, test_index in kfold.split(X):
        print(f"Subject {subject_id} - Fold {fold_no} 학습 시작")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 훈련 시 원래 라벨, 테스트 시 무작위 재배분 적용
        y_test_random = randomly_mix_labels(y_test, num_classes=num_classes)

        input_length = X.shape[1]  # 예: 128 (1s) 또는 1280 (10s)
        model = EMCNN(num_classes=num_classes, smoothing_kernel_size=smoothing_kernel_size,
                      downsample_factor=downsample_factor, input_length=input_length, num_channels=X.shape[2])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        fold_result_dir = os.path.join(subject_result_dir, f"fold_{fold_no}")
        os.makedirs(fold_result_dir, exist_ok=True)
        checkpoint_path = os.path.join(fold_result_dir, "best_model.keras")
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True,
                                                             monitor='val_loss', mode='min')

        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(X_test, y_test_random), callbacks=[checkpoint_cb], verbose=1)
        history_all[f"fold_{fold_no}"] = history.history

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title(f"{subject_id} Fold {fold_no} Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.title(f"{subject_id} Fold {fold_no} Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fold_result_dir, "training_history.png"))
        plt.close()

        model.load_weights(checkpoint_path)
        test_loss, test_acc = model.evaluate(X_test, y_test_random, verbose=0)
        print(f"Subject {subject_id} Fold {fold_no} -- Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

        y_pred_logits = model.predict(X_test)
        y_pred = np.argmax(y_pred_logits, axis=1)
        fold_report = classification_report(y_test_random, y_pred, digits=2)
        reports_all[f"fold_{fold_no}"] = fold_report
        conf_matrix = confusion_matrix(y_test_random, y_pred)
        conf_matrices_all[f"fold_{fold_no}"] = conf_matrix

        global_y_true.append(y_test_random)
        global_y_pred.append(y_pred)

        with open(os.path.join(fold_result_dir, "fold_results.txt"), 'w') as f:
            f.write(f"Subject {subject_id} Fold {fold_no} -- Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}\n")
            f.write("Classification Report:\n" + fold_report + "\n")
            f.write("Confusion Matrix:\n" + str(conf_matrix) + "\n")
        
        plt.figure(figsize=(6, 5))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f"{subject_id} Fold {fold_no} Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, ['Excited', 'relaxed', 'stressed', 'bored'], rotation=45)
        plt.yticks(tick_marks, ['Excited', 'relaxed', 'stressed', 'bored'])
        thresh = conf_matrix.max() / 2.
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, format(conf_matrix[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if conf_matrix[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(os.path.join(fold_result_dir, "confusion_matrix.png"))
        plt.close()

        fold_no += 1

    global_y_true = np.concatenate(global_y_true, axis=0)
    global_y_pred = np.concatenate(global_y_pred, axis=0)
    global_report = classification_report(global_y_true, global_y_pred, digits=2)

    summary_path = os.path.join(subject_result_dir, "crossval_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Subject {subject_id} 5-Fold Cross Validation Summary\n\n")
        for fold in history_all:
            f.write(f"{fold} Classification Report:\n")
            f.write(reports_all[fold] + "\n")
            f.write("Confusion Matrix:\n" + str(conf_matrices_all[fold]) + "\n\n")
        f.write("Global Classification Report (All Folds):\n")
        f.write(global_report + "\n")
    logging.info(f"Subject {subject_id} 학습 및 평가 완료. 결과는 {subject_result_dir}에 저장되었습니다.")

# -----------------------------
# 5. 데이터 로딩 함수: 두 가지 경우 (DEAP_PPG_1s와 DEAP_PPG_10s_1soverlap_4label)
# -----------------------------
# ch_choice_pair1, ch_choice_pair2 선택 부분은 5채널 모두 사용할 경우 제거함.
def load_data_1s(data_dir):
    """
    DEAP_PPG_1s 폴더 파일 로드
    파일 shape: (60, 5, 128)
    -> 모든 5채널 사용: transpose 후 (60, 128, 5)
    """
    subject_data = {}
    file_paths = glob.glob(os.path.join(data_dir, "*.npy"))
    logging.info(f"총 {len(file_paths)}개의 파일을 찾았습니다. (1s)")
    for file_path in file_paths:
        base = os.path.basename(file_path)
        try:
            subject_id = base.split('_')[0]
            label = int(base.split('_')[-1].split('.')[0])
        except Exception as e:
            logging.error(f"라벨/서브젝트 추출 오류: {file_path}, {e}")
            continue
        data = np.load(file_path)  # (60, 5, 128)
        data = np.transpose(data, (0, 2, 1))  # (60, 128, 5)
        if subject_id not in subject_data:
            subject_data[subject_id] = {'X': [], 'y': []}
        subject_data[subject_id]['X'].append(data)
        subject_data[subject_id]['y'].append(np.full((data.shape[0],), label, dtype=np.int32))
    for subject in subject_data:
        subject_data[subject]['X'] = np.concatenate(subject_data[subject]['X'], axis=0)
        subject_data[subject]['y'] = np.concatenate(subject_data[subject]['y'], axis=0)
        logging.info(f"{subject} - X shape: {subject_data[subject]['X'].shape}, y shape: {subject_data[subject]['y'].shape}")
    return subject_data

def load_data_10s(data_dir):
    """
    DEAP_PPG_10s_1soverlap_4label 폴더 파일 로드
    파일 shape: (51, 5, 1280)
    1. os.listdir를 사용하여 파일 목록 생성
    2. 각 파일을 np.load로 읽은 후, transpose하여 (51, 1280, 5)로 변경
    -> 모든 5채널 사용 (최종 shape: (51, 1280, 5))
    """
    subject_data = {}
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    logging.info(f"총 {len(file_list)}개의 파일을 찾았습니다. (10s)")
    
    for f in file_list:
        file_path = os.path.join(data_dir, f)
        try:
            subject_id = f.split('_')[0]
            label = int(f.split('_')[-1].split('.')[0])
        except Exception as e:
            logging.error(f"라벨/서브젝트 추출 오류: {file_path}, {e}")
            continue
        data = np.load(file_path)  # (51, 5, 1280)
        data = np.transpose(data, (0, 2, 1))  # (51, 1280, 5)
        if subject_id not in subject_data:
            subject_data[subject_id] = {'X': [], 'y': []}
        subject_data[subject_id]['X'].append(data)
        subject_data[subject_id]['y'].append(np.full((data.shape[0],), label, dtype=np.int32))
        
    for subject in subject_data:
        subject_data[subject]['X'] = np.concatenate(subject_data[subject]['X'], axis=0)
        subject_data[subject]['y'] = np.concatenate(subject_data[subject]['y'], axis=0)
        logging.info(f"{subject} - X shape: {subject_data[subject]['X'].shape}, y shape: {subject_data[subject]['y'].shape}")
    return subject_data

# -----------------------------
# 6. 메인: 원하는 폴더 선택 후 전체 서브젝트 학습 실행
# 저장 폴더 이름 끝에 mode (_1s 또는 _10s) 가 붙도록 함
# -----------------------------
if __name__ == "__main__":
    # mode 선택: '1s' 또는 '10s'
    mode = '1s'  # 예: '1s' 또는 '10s'
    if mode == '1s':
        data_dir = '/home/bcml1/2025_EMOTION/DEAP_PPG_1s'
        subject_data = load_data_1s(data_dir)
        result_dir_base = '/home/bcml1/sigenv/_4월/_ppg/EMCNN_LabelMix_Results_1s_5ch'
    elif mode == '10s':
        data_dir = '/home/bcml1/2025_EMOTION/DEAP_PPG_10s_1soverlap_4label/'
        subject_data = load_data_10s(data_dir)
        result_dir_base = '/home/bcml1/sigenv/_4월/_ppg/EMCNN_LabelMix_Results_10s_5ch'
    else:
        raise ValueError("mode는 '1s' 또는 '10s'여야 합니다.")

    num_classes = 4

    # 각 서브젝트에 대해 학습 실행 (테스트 시 라벨은 무작위 재배분되어 균등 분포가 되도록 함)
    for subject_id in sorted(subject_data.keys(), key=lambda x: int(x[1:])):
        X = subject_data[subject_id]['X']  # shape: (total_samples, L, 5)
        y = subject_data[subject_id]['y']  # shape: (total_samples,)
        logging.info(f"--- {subject_id} 학습 시작 ---")
        train_model_for_subject_with_random_labels(subject_id, X, y, num_classes=num_classes, epochs=50, batch_size=8,
                                                   result_dir_base=result_dir_base)
