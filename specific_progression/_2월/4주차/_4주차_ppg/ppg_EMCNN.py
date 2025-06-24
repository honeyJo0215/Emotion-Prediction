import os
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold  # KFold 대신 StratifiedKFold 사용
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

# -----------------------------
# 설정: 초기화 및 정규화
# -----------------------------
initializer = tf.keras.initializers.HeNormal()
reg = tf.keras.regularizers.l2(1e-6)

# -----------------------------
# 1. 각 채널별 CNN branch 정의 (입력: (1280, 1))
# 논문 Table 1에 따라 Conv1D 및 5단계 Depthwise Separable Conv 적용
# -----------------------------
def build_cnn_branch(input_length):
    inputs = tf.keras.Input(shape=(input_length, 1))
    # Layer 1: 일반 Conv1D
    x = tf.keras.layers.Conv1D(filters=8, kernel_size=7, strides=1, padding='same',
                               activation=tf.nn.relu6,
                               kernel_initializer=initializer,
                               kernel_regularizer=reg)(inputs)
    # Layer 2 ~ 5: Depthwise Separable Conv
    x = tf.keras.layers.SeparableConv1D(
            filters=16, kernel_size=7, strides=4, padding='same',
            activation=tf.nn.relu6,
            depthwise_initializer=initializer,
            pointwise_initializer=initializer,
            depthwise_regularizer=reg,
            pointwise_regularizer=reg
        )(x)
    x = tf.keras.layers.SeparableConv1D(
            filters=32, kernel_size=7, strides=2, padding='same',
            activation=tf.nn.relu6,
            depthwise_initializer=initializer,
            pointwise_initializer=initializer,
            depthwise_regularizer=reg,
            pointwise_regularizer=reg
        )(x)
    x = tf.keras.layers.SeparableConv1D(
            filters=64, kernel_size=7, strides=4, padding='same',
            activation=tf.nn.relu6,
            depthwise_initializer=initializer,
            pointwise_initializer=initializer,
            depthwise_regularizer=reg,
            pointwise_regularizer=reg
        )(x)
    x = tf.keras.layers.SeparableConv1D(
            filters=128, kernel_size=7, strides=2, padding='same',
            activation=tf.nn.relu6,
            depthwise_initializer=initializer,
            pointwise_initializer=initializer,
            depthwise_regularizer=reg,
            pointwise_regularizer=reg
        )(x)
    # Global Average Pooling → (batch, 128)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model

# -----------------------------
# 2. EMCNN 모델 정의 (각 채널을 branch로 분리)
# 입력: (1280, 5)
# -----------------------------
class EMCNN(tf.keras.Model):
    def __init__(self, num_classes=4, input_length=1280, num_branches=5):
        super(EMCNN, self).__init__()
        self.num_branches = num_branches
        self.branches = []
        for i in range(num_branches):
            branch = build_cnn_branch(input_length)
            self.branches.append(branch)
        self.concat = tf.keras.layers.Concatenate()
        self.fc1 = tf.keras.layers.Dense(128, activation=tf.nn.relu6,
                                         kernel_initializer=initializer,
                                         kernel_regularizer=reg)
        self.fc2 = tf.keras.layers.Dense(64, activation=tf.nn.relu6,
                                         kernel_initializer=initializer,
                                         kernel_regularizer=reg)
        self.fc3 = tf.keras.layers.Dense(num_classes, activation='softmax',
                                         kernel_initializer=initializer,
                                         kernel_regularizer=reg)
    
    def call(self, inputs, training=False):
        branch_outputs = []
        for i in range(self.num_branches):
            channel_input = inputs[:, :, i:i+1]  # (batch, 1280, 1)
            branch_out = self.branches[i](channel_input)  # (batch, 128)
            branch_outputs.append(branch_out)
        x = self.concat(branch_outputs)  # (batch, 640)
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.fc3(x)
        return out

# -----------------------------
# 3. 헬퍼 함수: 평균 classification report 계산 및 포맷
# -----------------------------
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

# -----------------------------
# 4. 학습 및 평가 (Stratified 5-Fold Cross Validation)
# -----------------------------
def train_model_for_subject(subject_id, X, y, num_classes=4, epochs=300, batch_size=128,
                            result_dir_base='/home/bcml1/sigenv/_4주차_ppg/EMCNN_1result'):
    """
    subject_id: 예, 's01'
    X: 입력 데이터, shape = (num_samples, 1280, 5)
    y: 정답 레이블, shape = (num_samples,)
    """
    subject_result_dir = os.path.join(result_dir_base, subject_id)
    os.makedirs(subject_result_dir, exist_ok=True)
    
    # StratifiedKFold를 사용하여 클래스 분포를 유지하면서 분할
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_no = 1

    history_all = {}
    reports_all = {}
    conf_matrices_all = {}

    for train_index, test_index in skf.split(X, y):
        print(f"Subject {subject_id} - Fold {fold_no} 학습 시작")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        input_length = X.shape[1]   # 1280
        num_channels = X.shape[2]   # 5
        model = EMCNN(num_classes=num_classes, input_length=input_length, num_branches=num_channels)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.99)
        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])

        fold_result_dir = os.path.join(subject_result_dir, f"fold_{fold_no}")
        os.makedirs(fold_result_dir, exist_ok=True)
        checkpoint_path = os.path.join(fold_result_dir, "best_model.keras")
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True,
                                                             monitor='val_loss', mode='min')

        # 모델 학습 시 validation_split=0.1 사용 (train 내에서도 stratification 효과는 완벽하지 않을 수 있음)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_split=0.1, callbacks=[checkpoint_cb], verbose=1)
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
        
        plt.figure(figsize=(6, 5))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f"{subject_id} Fold {fold_no} Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(num_classes)
        class_names = ['Excited', 'relaxed', 'stressed', 'bored']
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

    avg_report = compute_average_classification_report(reports_all)
    avg_report_str = format_classification_report(avg_report)
    with open(os.path.join(subject_result_dir, "avg_classification_report.txt"), 'w') as f:
        f.write(avg_report_str)

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
# 5. 데이터 로딩 및 증강 (클래스 분포 균일화)
# -----------------------------
def load_data(data_dir='/home/bcml1/2025_EMOTION/DEAP_PPG_10s_1soverlap_4label'):
    """
    각 파일은 (51, 5, 1280) 모양이며, 파일당 모든 51개 segment를 (51, 1280, 5)로 변환.
    파일명에서 서브젝트 ID와 라벨을 추출하여 그룹화한 후,
    부족한 클래스에 대해 작은 노이즈를 추가한 데이터 증강(oversampling)으로 전체 클래스 분포를 균일하게 만듭니다.
    """
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
        data = np.load(file_path)  # (51, 5, 1280)
        data = np.transpose(data, (0, 2, 1))  # (51, 1280, 5)
        if subject_id not in subject_data:
            subject_data[subject_id] = {'X': [], 'y': []}
        subject_data[subject_id]['X'].append(data)
        subject_data[subject_id]['y'].append(np.full((data.shape[0],), label, dtype=np.int32))
    
    for subject in subject_data:
        subject_data[subject]['X'] = np.concatenate(subject_data[subject]['X'], axis=0)
        subject_data[subject]['y'] = np.concatenate(subject_data[subject]['y'], axis=0)
        sample_count = subject_data[subject]['X'].shape[0]
        print(f"{subject} - X shape: {subject_data[subject]['X'].shape}, y shape: {subject_data[subject]['y'].shape}")
        
        # ----- 데이터 증강 (oversampling)으로 클래스 분포 균일화 -----
        X_data = subject_data[subject]['X']
        y_data = subject_data[subject]['y']
        unique_classes, counts = np.unique(y_data, return_counts=True)
        max_count = counts.max()
        noise_std = 1e-6  # 아주 작은 노이즈
        X_aug_list = [X_data]
        y_aug_list = [y_data]
        for cls in unique_classes:
            cls_indices = np.where(y_data == cls)[0]
            cls_count = len(cls_indices)
            if cls_count < max_count:
                diff = max_count - cls_count
                sampled_indices = np.random.choice(cls_indices, diff, replace=True)
                X_samples = X_data[sampled_indices]
                # 작은 노이즈 추가하여 증강
                noise = np.random.normal(loc=0.0, scale=noise_std, size=X_samples.shape)
                X_aug = X_samples + noise
                X_aug_list.append(X_aug)
                y_aug_list.append(np.full(diff, cls, dtype=y_data.dtype))
        X_data_balanced = np.concatenate(X_aug_list, axis=0)
        y_data_balanced = np.concatenate(y_aug_list, axis=0)
        subject_data[subject]['X'] = X_data_balanced
        subject_data[subject]['y'] = y_data_balanced
        print(f"{subject} after augmentation: X shape: {X_data_balanced.shape}, y shape: {y_data_balanced.shape}")
        # ----- 증강 종료 -----
    return subject_data

# -----------------------------
# 디버그: dummy input으로 각 branch 출력 확인
# -----------------------------
def debug_branches():
    dummy_input = np.random.randn(1, 1280, 5).astype(np.float32)
    model = EMCNN(num_classes=4, input_length=1280, num_branches=5)
    dummy_out = model(dummy_input, training=False)
    print("전체 모델 출력 shape:", dummy_out.shape)
    for i in range(5):
        branch_out = model.branches[i](dummy_input[:, :, i:i+1])
        print(f"Branch {i} 출력 shape: {branch_out.shape}, 평균: {np.mean(branch_out.numpy()):.6f}")

if __name__ == "__main__":
    # 디버그: branch 출력 확인
    debug_branches()
    
    data_dir = '/home/bcml1/2025_EMOTION/DEAP_PPG_10s_1soverlap_4label'
    result_dir_base = '/home/bcml1/sigenv/_4주차_ppg/EMCNN_1result'
    subject_data = load_data(data_dir=data_dir)
    num_classes = 4

    # ★ 학습할 서브젝트의 시작 아이디 지정 (예: 's03'부터 학습)
    start_subject = 's01'
    
    # 서브젝트 ID는 s01, s02, ... s22 형태라고 가정
    for subject_id in sorted(subject_data.keys(), key=lambda x: int(x[1:])):
        if int(subject_id[1:]) < int(start_subject[1:]):
            print(f"Skipping {subject_id} (학습 시작 기준: {start_subject})")
            continue
        X = subject_data[subject_id]['X']  # (total_samples, 1280, 5)
        y = subject_data[subject_id]['y']  # (total_samples,)
        print(f"--- {subject_id} 학습 시작 ---")
        train_model_for_subject(subject_id, X, y, num_classes=num_classes, epochs=300, batch_size=128,
                                result_dir_base=result_dir_base)
