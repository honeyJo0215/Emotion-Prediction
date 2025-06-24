# Case 2. 3채널 정보를 그대로 활용하는 경우
import os
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# 1. Feature Extractor 정의 (각 브랜치에 사용)
# -----------------------------
def build_feature_extractor(input_length=None):
    """
    입력 shape: (L, 3) – L: 시퀀스 길이, 3채널 입력
    출력: 128 차원 feature 벡터 (GlobalAveragePooling1D 사용)
    """
    inputs = tf.keras.Input(shape=(input_length, 3))  # 3채널 입력
    # Block 1: 일반 Conv1D (filters=8, kernel_size=7, stride=1)
    x = tf.keras.layers.Conv1D(filters=8, kernel_size=7, strides=1, padding='same')(inputs)
    x = tf.keras.layers.ReLU(max_value=6)(x)
    # Block 2: Depthwise separable convolution (filters=16, kernel_size=7, stride=4)
    x = tf.keras.layers.SeparableConv1D(filters=16, kernel_size=7, strides=4, padding='same')(x)
    x = tf.keras.layers.ReLU(max_value=6)(x)
    # Block 3: (filters=32, kernel_size=7, stride=2)
    x = tf.keras.layers.SeparableConv1D(filters=32, kernel_size=7, strides=2, padding='same')(x)
    x = tf.keras.layers.ReLU(max_value=6)(x)
    # Block 4: (filters=64, kernel_size=7, stride=4)
    x = tf.keras.layers.SeparableConv1D(filters=64, kernel_size=7, strides=4, padding='same')(x)
    x = tf.keras.layers.ReLU(max_value=6)(x)
    # Block 5: (filters=128, kernel_size=7, stride=2)
    x = tf.keras.layers.SeparableConv1D(filters=128, kernel_size=7, strides=2, padding='same')(x)
    x = tf.keras.layers.ReLU(max_value=6)(x)
    # Global Average Pooling: 결과를 128차원 벡터로 만듦.
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

        # Smoothing: 3채널 입력에 대해 각 채널별로 이동평균 (groups=3 사용)
        self.smoothing_conv = tf.keras.layers.Conv1D(filters=3, kernel_size=smoothing_kernel_size,
                                                     padding='same', use_bias=False, groups=3)
        # 필터 가중치: shape=(smoothing_kernel_size, 1, 3) 각 채널에 동일하게 적용
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
# 3. 학습 및 평가 (서브젝트별 5-Fold Cross Validation)
# -----------------------------
def train_model_for_subject(subject_id, X, y, num_classes=4, epochs=50, batch_size=32,
                            smoothing_kernel_size=2, downsample_factor=2,
                            result_dir_base='EMCNN_CombinedChannel_Results'):
    """
    subject_id: 예, 's01'
    X: 입력 데이터, shape = (num_samples, 1280, 3) 해당 서브젝트 데이터
    y: 정답 레이블, shape = (num_samples,)
    """
    subject_result_dir = os.path.join(result_dir_base, subject_id)
    os.makedirs(subject_result_dir, exist_ok=True)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_no = 1

    history_all = {}
    reports_all = {}
    conf_matrices_all = {}

    for train_index, test_index in kfold.split(X):
        print(f"Subject {subject_id} - Fold {fold_no} 학습 시작")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        input_length = X.shape[1]  # 예: 1280
        model = EMCNN(num_classes=num_classes, smoothing_kernel_size=smoothing_kernel_size,
                      downsample_factor=downsample_factor, input_length=input_length)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        fold_result_dir = os.path.join(subject_result_dir, f"fold_{fold_no}")
        os.makedirs(fold_result_dir, exist_ok=True)
        checkpoint_path = os.path.join(fold_result_dir, "best_model.keras")
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True,
                                                             monitor='val_loss', mode='min')

        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_split=0.1, callbacks=[checkpoint_cb], verbose=1)
        history_all[f"fold_{fold_no}"] = history.history

        # 학습 그래프 저장 (loss, accuracy)
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

        # 최적 모델 로드 및 평가
        model.load_weights(checkpoint_path)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Subject {subject_id} Fold {fold_no} -- Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

        y_pred_logits = model.predict(X_test)
        y_pred = np.argmax(y_pred_logits, axis=1)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        reports_all[f"fold_{fold_no}"] = report
        conf_matrices_all[f"fold_{fold_no}"] = conf_matrix

        # 텍스트 결과 저장
        with open(os.path.join(fold_result_dir, "fold_results.txt"), 'w') as f:
            f.write(f"Subject {subject_id} Fold {fold_no} -- Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}\n")
            f.write("Classification Report:\n")
            f.write(str(report) + "\n")
            f.write("Confusion Matrix:\n")
            f.write(str(conf_matrix) + "\n")
        
        # 혼동 행렬 이미지 저장
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

    # subject 별 Cross Validation 요약 저장
    summary_path = os.path.join(subject_result_dir, "crossval_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Subject {subject_id} 5-Fold Cross Validation Summary\n")
        for fold in history_all:
            f.write(f"{fold}:\n")
            f.write("History: " + str(history_all[fold]) + "\n")
            f.write("Classification Report: " + str(reports_all[fold]) + "\n")
            f.write("Confusion Matrix: " + str(conf_matrices_all[fold]) + "\n\n")
    print(f"Subject {subject_id} 학습 및 평가 완료. 결과는 {subject_result_dir}에 저장되었습니다.")

# -----------------------------
# 4. 데이터 로딩 및 전체 서브젝트 학습 실행
# -----------------------------
def load_data(data_dir='/home/bcml1/2025_EMOTION/DEAP_PPG_10s_1soverlap_4label'):
    """
    주어진 폴더 내의 모든 .npy 파일을 읽어들입니다.
    각 파일은 (51, 3, 1280) 모양이며, 각 파일에서 51개 전체 segment 대신
    균일하게 6개 segment를 선택하여 (6, 1280, 3)으로 변환한 후,
    파일명에서 서브젝트 ID (예: s01 ~ s22)와 라벨(파일명 마지막의 label_X.npy)을 추출하여
    서브젝트별로 데이터를 그룹화합니다.
    """
    subject_data = {}  # key: subject id, value: dict with 'X' and 'y'
    file_paths = glob.glob(os.path.join(data_dir, "*.npy"))
    print(f"총 {len(file_paths)}개의 파일을 찾았습니다.")
    for file_path in file_paths:
        base = os.path.basename(file_path)
        try:
            subject_id = base.split('_')[0]
            label_str = base.split('_')[-1].split('.')[0]
            label = int(label_str)
        except Exception as e:
            print("라벨/서브젝트 추출 오류:", file_path, e)
            continue
        data = np.load(file_path)  # shape: (51, 3, 1280)
        # 각 segment: (3, 1280) -> (1280, 3)
        data = np.transpose(data, (0, 2, 1))  # (51, 1280, 3)
        # 균일하게 6개 segment 선택 (예: 인덱스 0, 10, 20, 30, 40, 50)
        indices = np.linspace(0, data.shape[0]-1, 6, dtype=int)
        data = data[indices, :, :]  # (6, 1280, 3)
        if subject_id not in subject_data:
            subject_data[subject_id] = {'X': [], 'y': []}
        subject_data[subject_id]['X'].append(data)
        subject_data[subject_id]['y'].append(np.full((data.shape[0],), label, dtype=np.int32))
    
    # 각 서브젝트별로 concatenate
    for subject in subject_data:
        subject_data[subject]['X'] = np.concatenate(subject_data[subject]['X'], axis=0)
        subject_data[subject]['y'] = np.concatenate(subject_data[subject]['y'], axis=0)
        print(f"{subject} - X shape: {subject_data[subject]['X'].shape}, y shape: {subject_data[subject]['y'].shape}")
    return subject_data

if __name__ == "__main__":
    data_dir = '/home/bcml1/2025_EMOTION/DEAP_PPG_10s_1soverlap_4label'
    result_dir_base = '/home/bcml1/sigenv/_4주차_ppg/EMCNN_CombinedChannel_Results'
    subject_data = load_data(data_dir=data_dir)
    num_classes = 4

    # 서브젝트 ID를 s01, s02, ... s22 순으로 정렬하여 학습 실행
    for subject_id in sorted(subject_data.keys(), key=lambda x: int(x[1:])):
        X = subject_data[subject_id]['X']  # (total_samples, 1280, 3)
        y = subject_data[subject_id]['y']  # (total_samples,)
        print(f"--- {subject_id} 학습 시작 ---")
        train_model_for_subject(subject_id, X, y, num_classes=num_classes, epochs=50, batch_size=8,
                                result_dir_base=result_dir_base)
