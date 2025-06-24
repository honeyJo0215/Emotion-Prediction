import os
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix

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

# -------------------------------------------------------------------------
# 설정: 초기화 및 정규화 파라미터
# -------------------------------------------------------------------------
initializer = tf.keras.initializers.HeNormal()
reg = tf.keras.regularizers.l2(0.0005)  # l2 세기 조금 낮춤

# -------------------------------------------------------------------------
# 1. CNN Branch 정의 (Table 1 참고)
#    - Batch Normalization + Dropout 추가
# -------------------------------------------------------------------------
def build_cnn_branch(input_length=None, dropout_rate=0.3):
    """
    input_length: 시퀀스 길이 (None 허용)
    dropout_rate: 과적합 방지를 위한 dropout 비율
    """
    inputs = tf.keras.Input(shape=(input_length, 1))

    # Layer 1: 일반 Conv1D
    x = tf.keras.layers.Conv1D(filters=8, kernel_size=7, strides=1, padding='same',
                               kernel_initializer=initializer,
                               kernel_regularizer=reg,
                               activation=None)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=6.0)(x)
    
    # Layer 2: Depthwise Separable Conv
    x = tf.keras.layers.SeparableConv1D(filters=16, kernel_size=7, strides=4, padding='same',
                                        depthwise_initializer=initializer,
                                        pointwise_initializer=initializer,
                                        depthwise_regularizer=reg,
                                        pointwise_regularizer=reg,
                                        activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=6.0)(x)

    # Layer 3: Depthwise Separable Conv
    x = tf.keras.layers.SeparableConv1D(filters=32, kernel_size=7, strides=2, padding='same',
                                        depthwise_initializer=initializer,
                                        pointwise_initializer=initializer,
                                        depthwise_regularizer=reg,
                                        pointwise_regularizer=reg,
                                        activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=6.0)(x)

    # Layer 4: Depthwise Separable Conv
    x = tf.keras.layers.SeparableConv1D(filters=64, kernel_size=7, strides=4, padding='same',
                                        depthwise_initializer=initializer,
                                        pointwise_initializer=initializer,
                                        depthwise_regularizer=reg,
                                        pointwise_regularizer=reg,
                                        activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=6.0)(x)

    # Layer 5: Depthwise Separable Conv
    x = tf.keras.layers.SeparableConv1D(filters=128, kernel_size=7, strides=2, padding='same',
                                        depthwise_initializer=initializer,
                                        pointwise_initializer=initializer,
                                        depthwise_regularizer=reg,
                                        pointwise_regularizer=reg,
                                        activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(max_value=6.0)(x)

    # Global Average Pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Branch별로 FC 전에는 Dropout 적용(선택)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model

# -------------------------------------------------------------------------
# 2. Transformation Stage
# -------------------------------------------------------------------------
def transform_signal(x, s=2, d=2):
    # Identity mapping
    identity = x
    # Smoothing: moving average (window size s, stride=1)
    smoothing = tf.keras.layers.AveragePooling1D(pool_size=s, strides=1, padding='same')(x)
    # Downsampling: 매 d번째 샘플만 선택
    downsampling = tf.keras.layers.Lambda(lambda z: z[:, ::d, :])(x)
    return [identity, smoothing, downsampling]

# -------------------------------------------------------------------------
# 3. EMCNN 모델 정의
# -------------------------------------------------------------------------
class EMCNN(tf.keras.Model):
    def __init__(self, num_classes=4, input_length=1280, s=2, d=2, dropout_rate=0.3):
        super(EMCNN, self).__init__()
        self.s = s
        self.d = d
        self.num_classes = num_classes

        # 3개 Branch (가중치 공유 X)
        self.branches = []
        for _ in range(3):
            branch = build_cnn_branch(input_length=None, dropout_rate=dropout_rate)
            self.branches.append(branch)

        self.concat = tf.keras.layers.Concatenate()

        # FC Layers
        self.fc1 = tf.keras.layers.Dense(128, kernel_initializer=initializer,
                                         kernel_regularizer=reg, activation=None)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.ReLU(max_value=6.0)

        self.fc2 = tf.keras.layers.Dense(64, kernel_initializer=initializer,
                                         kernel_regularizer=reg, activation=None)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.act2 = tf.keras.layers.ReLU(max_value=6.0)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.fc3 = tf.keras.layers.Dense(num_classes, kernel_initializer=initializer,
                                         kernel_regularizer=reg, activation='softmax')

    def call(self, inputs, training=False):
        # 변환 단계
        transformed = transform_signal(inputs, s=self.s, d=self.d)
        branch_outputs = []
        for i, branch in enumerate(self.branches):
            branch_out = branch(transformed[i])
            branch_outputs.append(branch_out)

        x = self.concat(branch_outputs)

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.dropout(x, training=training)

        out = self.fc3(x)
        return out

# -------------------------------------------------------------------------
# 헬퍼 함수들
# -------------------------------------------------------------------------
def compute_average_classification_report(reports_all):
    avg_report = {}
    first_key = list(reports_all.keys())[0]
    for key in reports_all[first_key].keys():
        if isinstance(reports_all[first_key][key], dict):
            avg_report[key] = {}
            for metric in reports_all[first_key][key]:
                avg_report[key][metric] = np.mean([reports_all[f][key][metric] for f in reports_all])
        else:
            avg_report[key] = np.mean([reports_all[f][key] for f in reports_all])
    return avg_report

def format_classification_report(avg_report):
    lines = []
    header = f"{'':<12}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}"
    lines.append(header)
    lines.append("")
    for key in sorted(avg_report.keys(), key=lambda x: (x not in ['macro avg','weighted avg','accuracy'], x)):
        if key == 'accuracy':
            continue
        metrics = avg_report[key]
        line = f"{key:<12}{metrics['precision']:10.2f}{metrics['recall']:10.2f}{metrics['f1-score']:10.2f}{metrics['support']:10.0f}"
        lines.append(line)
    acc_line = f"\n{'accuracy':<12}{avg_report['accuracy']:>10.2f}"
    lines.append(acc_line)
    return "\n".join(lines)

# -------------------------------------------------------------------------
# 5-Fold Cross Validation 학습 함수
#   - ReduceLROnPlateau 콜백 추가
# -------------------------------------------------------------------------
def train_model_for_subject(subject_id, X, y, num_classes=4, epochs=300, batch_size=128,
                            result_dir_base='./EMCNN_results'):
    """
    subject_id: 예, 's01'
    X: (num_samples, 1280, 1)
    y: (num_samples,)
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

        model = EMCNN(num_classes=num_classes, input_length=X.shape[1], s=2, d=2, dropout_rate=0.3)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.99)

        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])

        fold_result_dir = os.path.join(subject_result_dir, f"fold_{fold_no}")
        os.makedirs(fold_result_dir, exist_ok=True)
        checkpoint_path = os.path.join(fold_result_dir, "best_model.keras")

        # ReduceLROnPlateau: val_loss 개선이 없으면 LR 감소
        reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                            factor=0.5,
                                                            patience=10,
                                                            min_lr=1e-6,
                                                            verbose=1)
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True,
                                                           monitor='val_loss', mode='min')

        history = model.fit(X_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=0.1,
                            callbacks=[checkpoint_cb, reduce_lr_cb],
                            verbose=1)
        history_all[f"fold_{fold_no}"] = history.history

        # 학습 그래프 저장
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

        # Best model 로드 후 테스트
        model.load_weights(checkpoint_path)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Subject {subject_id} Fold {fold_no} -- Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

        y_pred = np.argmax(model.predict(X_test), axis=1)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        reports_all[f"fold_{fold_no}"] = report
        conf_matrices_all[f"fold_{fold_no}"] = conf_matrix

        with open(os.path.join(fold_result_dir, "fold_results.txt"), 'w') as f:
            f.write(f"Subject {subject_id} Fold {fold_no} -- Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}\n")
            f.write("Classification Report:\n")
            f.write(str(report) + "\n")
            f.write("Confusion Matrix:\n")
            f.write(str(conf_matrix) + "\n")

        # 혼동 행렬 시각화
        plt.figure(figsize=(6, 5))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f"{subject_id} Fold {fold_no} Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(num_classes)
        class_names = ['Excited', 'Relaxed', 'Stressed', 'Bored']  # 예시
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
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

    # 5-Fold 평균 report
    avg_report = compute_average_classification_report(reports_all)
    avg_report_str = format_classification_report(avg_report)
    with open(os.path.join(subject_result_dir, "avg_classification_report.txt"), 'w') as f:
        f.write(avg_report_str)

    # 전체 요약
    summary_path = os.path.join(subject_result_dir, "crossval_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Subject {subject_id} 5-Fold Cross Validation Summary\n")
        for fold in history_all:
            f.write(f"{fold}:\n")
            f.write("History: " + str(history_all[fold]) + "\n")
            f.write("Classification Report: " + str(reports_all[fold]) + "\n")
            f.write("Confusion Matrix: " + str(conf_matrices_all[fold]) + "\n\n")

    print(f"Subject {subject_id} 학습 및 평가 완료. 결과는 {subject_result_dir}에 저장되었습니다.")

# -------------------------------------------------------------------------
# 6. 데이터 로딩 함수 (사용자 환경에 맞게 조정)
# -------------------------------------------------------------------------
def load_data(data_dir='/home/bcml1/2025_EMOTION/DEAP_PPG_10s_1soverlap_4label'):
    subject_data = {}
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
        data = np.load(file_path)  # (51, 5, 1280) 가정
        data = np.transpose(data, (0, 2, 1))  # (51, 1280, 5)
        data = data[..., 0:1]  # 첫 번째 채널만
        if subject_id not in subject_data:
            subject_data[subject_id] = {'X': [], 'y': []}
        subject_data[subject_id]['X'].append(data)
        subject_data[subject_id]['y'].append(np.full((data.shape[0],), label, dtype=np.int32))

    for subject in subject_data:
        subject_data[subject]['X'] = np.concatenate(subject_data[subject]['X'], axis=0)
        subject_data[subject]['y'] = np.concatenate(subject_data[subject]['y'], axis=0)
        sample_count = subject_data[subject]['X'].shape[0]
        print(f"{subject} - X shape: {subject_data[subject]['X'].shape}, y shape: {subject_data[subject]['y'].shape}")
        if sample_count != 2040:
            print(f"[DEBUG] 경고: {subject}의 샘플 수가 {sample_count}개입니다. (예상: 2040개)")
        else:
            print(f"[DEBUG] {subject}의 샘플 수가 2040개로 정상 로드되었습니다.")
    return subject_data

# -------------------------------------------------------------------------
# 7. 디버그용
# -------------------------------------------------------------------------
def debug_branches():
    dummy_input = np.random.randn(1, 1280, 1).astype(np.float32)
    model = EMCNN(num_classes=4, input_length=1280, s=2, d=2, dropout_rate=0.3)
    dummy_out = model(dummy_input, training=False)
    print("전체 모델 출력 shape:", dummy_out.shape)
    transformed = transform_signal(dummy_input, s=2, d=2)
    for i, t in enumerate(transformed):
        branch_out = model.branches[i](t)
        print(f"Branch {i} 출력 shape: {branch_out.shape}, 평균: {np.mean(branch_out.numpy()):.6f}")

# 메인 실행 예시
if __name__ == "__main__":
    # 디버그(옵션)
    debug_branches()

    data_dir = '/home/bcml1/2025_EMOTION/DEAP_PPG_10s_1soverlap_4label'
    result_dir_base = '/home/bcml1/sigenv/_4주차_ppg/EMCNN_1result2'
    subject_data = load_data(data_dir=data_dir)
    num_classes = 4

    for subject_id in sorted(subject_data.keys(), key=lambda x: int(x[1:])):
        X = subject_data[subject_id]['X']  # (total_samples, 1280, 1)
        y = subject_data[subject_id]['y']  # (total_samples,)
        print(f"--- {subject_id} 학습 시작 ---")
        train_model_for_subject(subject_id, X, y, num_classes=num_classes, epochs=300, batch_size=128,
                                result_dir_base=result_dir_base)
