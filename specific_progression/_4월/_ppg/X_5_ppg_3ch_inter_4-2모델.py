import os
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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

# -----------------------------
# 1. Feature Extractor 정의 (각 브랜치에 사용)
# -----------------------------
def build_feature_extractor(input_length=None):
    """
    입력 shape: (L, 3) – L: 시퀀스 길이, 3채널 입력
    출력: 128 차원 feature 벡터 (GlobalAveragePooling1D 사용)
    """
    inputs = tf.keras.Input(shape=(input_length, 3))
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
    def __init__(self, num_classes=4, smoothing_kernel_size=2, downsample_factor=2, input_length=None):
        """
        num_classes: 분류 클래스 수  
        smoothing_kernel_size: 이동평균 필터의 커널 크기 (예: 2)  
        downsample_factor: 다운샘플링 비율 (예: 2)  
        input_length: 원본 시퀀스 길이
        """
        super(EMCNN, self).__init__()
        self.num_classes = num_classes
        self.smoothing_kernel_size = smoothing_kernel_size
        self.downsample_factor = downsample_factor

        # 각 브랜치별 Feature Extractor (가중치는 공유하지 않음)
        self.feature_identity = build_feature_extractor(input_length=input_length)
        self.feature_smoothing = build_feature_extractor(input_length=input_length)
        self.feature_downsampling = build_feature_extractor(input_length=None)

        # Smoothing: 3채널 입력에 대해 각 채널별 이동평균 (groups=3 사용)
        self.smoothing_conv = tf.keras.layers.Conv1D(filters=3, kernel_size=smoothing_kernel_size,
                                                     padding='same', use_bias=False, groups=3)
        smoothing_weight = np.ones((smoothing_kernel_size, 1, 3), dtype=np.float32) / smoothing_kernel_size
        self.smoothing_conv.build((None, input_length, 3))
        self.smoothing_conv.set_weights([smoothing_weight])
        self.smoothing_conv.trainable = False

        # 분류를 위한 Fully Connected 레이어들
        self.fc1 = tf.keras.layers.Dense(128, activation=lambda x: tf.nn.relu6(x))
        self.fc2 = tf.keras.layers.Dense(64, activation=lambda x: tf.nn.relu6(x))
        self.fc3 = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training=False):
        """
        inputs: shape = (batch, L, 3)
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
# 3. 데이터 로딩 함수: 1초/10초 데이터 (5채널) → 추후 학습 전에 3채널 선택
# -----------------------------
def load_data_1s(data_dir):
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
# 4. Inter-Subject Cross Validation (LOSO) 학습 함수
# -----------------------------
def train_model_inter_subject(subject_data, num_classes=4, epochs=50, batch_size=8,
                              smoothing_kernel_size=2, downsample_factor=2, result_dir_base=''):
    """
    각 피실험자를 테스트셋(LOSO)으로 사용하고, 나머지 피실험자들로 학습하는 inter-subject cross validation.
    입력 subject_data: {subject_id: {'X': numpy array, 'y': numpy array}}
    5채널 데이터에서 3채널 (예: 인덱스 [0,1,2])만 선택.
    """
    global_y_true = []
    global_y_pred = []

    for test_subject in sorted(subject_data.keys(), key=lambda x: int(x[1:])):
        
        if test_subject != 's19':
            continue
        # 테스트 피실험자의 데이터에서 3채널만 선택 (예: 채널 인덱스 0,1,2)
        X_test = subject_data[test_subject]['X'][..., [0, 1, 2]]
        y_test = subject_data[test_subject]['y']

        # 나머지 피실험자들의 데이터를 학습셋으로 결합 (각각 3채널 선택)
        X_train_list = []
        y_train_list = []
        for train_subject in subject_data.keys():
            if train_subject == test_subject:
                continue
            X_train_list.append(subject_data[train_subject]['X'][..., [0, 1, 2]])
            y_train_list.append(subject_data[train_subject]['y'])
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        logging.info(f"LOSO: {test_subject}를 테스트셋으로 사용, 나머지 피실험자들로 학습 (X_train shape: {X_train.shape}, X_test shape: {X_test.shape})")
        input_length = X_train.shape[1]  # 예: 128 (1s) 또는 1280 (10s)
        model = EMCNN(num_classes=num_classes, smoothing_kernel_size=smoothing_kernel_size,
                      downsample_factor=downsample_factor, input_length=input_length)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        subject_result_dir = os.path.join(result_dir_base, test_subject)
        os.makedirs(subject_result_dir, exist_ok=True)
        checkpoint_path = os.path.join(subject_result_dir, "best_model.keras")
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True,
                                                             monitor='val_loss', mode='min')

        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(X_test, y_test), callbacks=[checkpoint_cb], verbose=1)

        # 학습 완료 후 best_model 불러오기
        model.load_weights(checkpoint_path)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        logging.info(f"테스트 피실험자 {test_subject} -- Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

        y_pred_logits = model.predict(X_test)
        y_pred = np.argmax(y_pred_logits, axis=1)

        # 테스트 결과 저장 (평가 리포트 및 혼동행렬)
        report = classification_report(y_test, y_pred, digits=2)
        conf_matrix = confusion_matrix(y_test, y_pred)
        eval_path = os.path.join(subject_result_dir, f"{test_subject}_evaluation.txt")
        with open(eval_path, 'w') as f:
            f.write(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}\n")
            f.write("Classification Report:\n" + report + "\n")
            f.write("Confusion Matrix:\n" + str(conf_matrix) + "\n")

        # 혼동행렬 시각화 저장
        plt.figure(figsize=(6, 5))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f"{test_subject} Confusion Matrix")
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
        plt.savefig(os.path.join(subject_result_dir, "confusion_matrix.png"))
        plt.close()

        global_y_true.append(y_test)
        global_y_pred.append(y_pred)

    # 전체 피실험자에 대한 global 평가 (모든 테스트 데이터를 결합)
    global_y_true = np.concatenate(global_y_true, axis=0)
    global_y_pred = np.concatenate(global_y_pred, axis=0)
    global_report = classification_report(global_y_true, global_y_pred, digits=2)
    summary_path = os.path.join(result_dir_base, "global_evaluation.txt")
    with open(summary_path, 'w') as f:
        f.write("Global Classification Report (LOSO Inter-Subject):\n")
        f.write(global_report + "\n")
    logging.info("Inter-subject cross validation 완료. 전체 평가 결과는 '{}'에 저장되었습니다.".format(summary_path))

# -----------------------------
# 5. 메인: mode 선택 후 전체 피실험자에 대해 inter-subject cross validation 실행
# -----------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mode = '1s'  # '1s' 또는 '10s'
    if mode == '1s':
        data_dir = '/home/bcml1/2025_EMOTION/DEAP_PPG_1s'
        subject_data = load_data_1s(data_dir)
        result_dir_base = '/home/bcml1/sigenv/_4월/_ppg/EMCNN_Inter_Results_1s_3ch'
    elif mode == '10s':
        data_dir = '/home/bcml1/2025_EMOTION/DEAP_PPG_10s_1soverlap_4label/'
        subject_data = load_data_10s(data_dir)
        result_dir_base = '/home/bcml1/sigenv/_4월/_ppg/EMCNN_Inter_Results_10s_3ch'
    else:
        raise ValueError("mode는 '1s' 또는 '10s'여야 합니다.")

    train_model_inter_subject(subject_data, num_classes=4, epochs=50, batch_size=8,
                              smoothing_kernel_size=2, downsample_factor=2, result_dir_base=result_dir_base)
