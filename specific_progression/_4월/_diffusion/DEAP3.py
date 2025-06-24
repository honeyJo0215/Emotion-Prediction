import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 결과 저장 경로 설정
RESULT_DIR = "/home/bcml1/sigenv/_4월/_diffusion/DEAP3_rx_diffnoise_CNN_diff2"
os.makedirs(RESULT_DIR, exist_ok=True)

# =============================================================================
# GPU 메모리 제한 (필요시 사용)
def limit_gpu_memory(memory_limit_mib=10000):
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
limit_gpu_memory(10000)

# =============================================================================
# DiffuSSMLayer: Dense와 LayerNormalization을 이용한 노이즈 제거 모듈
class DiffuSSMLayer(layers.Layer):
    def __init__(self, hidden_dim=64, output_units=None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_units = output_units
        self.dense1 = layers.Dense(hidden_dim, activation='relu')
        self.norm1 = layers.LayerNormalization()
        self.dense2 = layers.Dense(hidden_dim, activation='relu')
        self.norm2 = layers.LayerNormalization()
        final_units = output_units if output_units is not None else hidden_dim
        self.out_dense = layers.Dense(final_units, activation=None)
        self.norm_out = layers.LayerNormalization()

    def call(self, x, training=False):
        h = self.dense1(x)
        h = self.norm1(h)
        h = self.dense2(h)
        h = self.norm2(h)
        out = self.out_dense(h)
        out = self.norm_out(out)
        return out

# =============================================================================
# 가우시안 노이즈 추가 함수
def add_diffusion_noise(x, stddev=0.05):
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=stddev)
    return x + noise

# =============================================================================
# Frequency branch CNN block with pre- and post-diffusion
def build_freq_branch_with_diffusion(input_shape=(200,8,4), noise_std=0.05):
    inp = layers.Input(shape=input_shape)
    # Pre-CNN: 노이즈 추가 (Residual 연결 없음)
    pre_processed = layers.Lambda(lambda x: add_diffusion_noise(x, stddev=noise_std))(inp)
    # CNN block
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(pre_processed)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    freq_feat = x
    # Post-diffusion block
    noisy_freq_feat = layers.Lambda(lambda x: add_diffusion_noise(x, stddev=noise_std))(freq_feat)
    diffu_freq = DiffuSSMLayer(hidden_dim=64)(noisy_freq_feat)
    freq_res = layers.Add()([freq_feat, diffu_freq])
    branch = models.Model(inputs=inp, outputs=freq_res, name="FreqBranchDiff")
    return branch

# =============================================================================
# Channel branch CNN block with pre- and post-diffusion
def build_chan_branch_with_diffusion(input_shape=(200,4,8), noise_std=0.05):
    inp = layers.Input(shape=input_shape)
    # Pre-CNN: 노이즈 추가 (Residual 연결 없음)
    pre_processed = layers.Lambda(lambda x: add_diffusion_noise(x, stddev=noise_std))(inp)
    # CNN block
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(pre_processed)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    chan_feat = x
    # Post-diffusion block
    noisy_chan_feat = layers.Lambda(lambda x: add_diffusion_noise(x, stddev=noise_std))(chan_feat)
    diffu_chan = DiffuSSMLayer(hidden_dim=64)(noisy_chan_feat)
    chan_res = layers.Add()([chan_feat, diffu_chan])
    branch = models.Model(inputs=inp, outputs=chan_res, name="ChanBranchDiff")
    return branch

# =============================================================================
# 전체 모델 구성: 주파수와 채널 branch 결합
def build_separated_model_with_diffusion(num_classes, noise_std=0.05):
    # 입력 shape: (4,200,8)
    input_csp = layers.Input(shape=(4,200,8), name="CSP_Input")
    # Frequency branch: (batch,4,200,8) -> (batch,200,8,4)
    freq_input = layers.Permute((2,3,1), name="Freq_Permute")(input_csp)
    freq_branch = build_freq_branch_with_diffusion(input_shape=(200,8,4), noise_std=noise_std)
    freq_feat = freq_branch(freq_input)
    # Channel branch: (batch,4,200,8) -> (batch,200,4,8)
    chan_input = layers.Permute((2,1,3), name="Chan_Permute")(input_csp)
    chan_branch = build_chan_branch_with_diffusion(input_shape=(200,4,8), noise_std=noise_std)
    chan_feat = chan_branch(chan_input)
    # 결합 후 분류기
    combined = layers.Concatenate()([freq_feat, chan_feat])
    x = layers.Dense(128, activation='relu')(combined)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=input_csp, outputs=output, name="Separated_CNN_Model_with_Diffusion")
    return model

# =============================================================================
# CSP feature 로드 및 1초 단위 분할, 클래스 불균형 보완 (diffusion noise 증강)
def load_csp_features(csp_feature_dir="/home/bcml1/2025_EMOTION/DEAP_eeg_new_label_CSP"):
    """
    지정 디렉토리의 모든 npy 파일을 로드합니다.
    파일명 형식: folder{num}_subject{num}_sample{num}_label{label}.npy
    각 파일은 (4, T, 8)이며, 1초(200 샘플) 단위로 분할하여 (4,200,8) 샘플 생성.
    반환: X, Y, subjects, file_ids
    """
    X_list, Y_list, subjects_list, files_list = [], [], [], []
    for file_name in os.listdir(csp_feature_dir):
        if not file_name.endswith(".npy"):
            continue
        file_path = os.path.join(csp_feature_dir, file_name)
        try:
            data = np.load(file_path)  # 예상 shape: (4, T, 8)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
        if data.ndim != 3 or data.shape[0] != 4 or data.shape[2] != 8:
            print(f"Unexpected shape in {file_path}: {data.shape}")
            continue

        # label 추출: "label" 뒤의 숫자
        label_match = re.search(r'label(\d+)', file_name, re.IGNORECASE)
        if label_match is None:
            print(f"No label found in {file_name}, skipping file.")
            continue
        label = int(label_match.group(1))
        # 만약 0,1,2,3 중 하나가 아니라면 건너뜁니다.
        if label not in [0, 1, 2, 3]:
            print(f"Label {label} in file {file_name} is not in [0,1,2,3], skipping file.")
            continue

        # subject 추출: "subject" 뒤의 숫자
        subject_match = re.search(r'subject(\d+)', file_name, re.IGNORECASE)
        subject = subject_match.group(1) if subject_match else 'unknown'
        T = data.shape[1]
        n_windows = T // 200  # 1초=200 샘플
        if n_windows < 1:
            continue
        for i in range(n_windows):
            window = data[:, i*200:(i+1)*200, :]  # (4,200,8)
            X_list.append(window)
            Y_list.append(label)
            files_list.append(f"{file_name}_win{i}")
            subjects_list.append(subject)
    X = np.array(X_list)
    Y = np.array(Y_list)
    subjects = np.array(subjects_list)
    file_ids = np.array(files_list)

    print(f"Loaded {X.shape[0]} samples, each shape: {X.shape[1:]}")
    print(f"Unique labels found: {np.unique(Y)}")
    print(f"Unique subjects found: {np.unique(subjects)}")

    # ----- 증강: 전체 데이터에서 각 라벨별 샘플 수를 동일하게 맞춤 -----
    unique_labels, counts = np.unique(Y, return_counts=True)
    max_count = counts.max()
    print(f"Balancing classes to {max_count} samples per label using diffusion noise augmentation.")
    augmented_X = []
    augmented_Y = []
    augmented_subjects = []
    augmented_file_ids = []
    for label in unique_labels:
        label_mask = (Y == label)
        current_count = np.sum(label_mask)
        num_to_augment = max_count - current_count
        if num_to_augment > 0:
            indices = np.where(label_mask)[0]
            for i in range(num_to_augment):
                random_index = np.random.choice(indices)
                sample = X[random_index]
                sample_tensor = tf.convert_to_tensor(sample, dtype=tf.float32)
                augmented_sample = add_diffusion_noise(sample_tensor, stddev=0.05).numpy()
                augmented_X.append(augmented_sample)
                augmented_Y.append(label)
                augmented_subjects.append(subjects[random_index])
                augmented_file_ids.append(file_ids[random_index] + f"_aug{i}")
    if augmented_X:
        X = np.concatenate([X, np.array(augmented_X)], axis=0)
        Y = np.concatenate([Y, np.array(augmented_Y)], axis=0)
        subjects = np.concatenate([subjects, np.array(augmented_subjects)], axis=0)
        file_ids = np.concatenate([file_ids, np.array(augmented_file_ids)], axis=0)
        print(f"After augmentation: {X.shape[0]} samples, each shape: {X.shape[1:]}")
    return X, Y, subjects, file_ids

# =============================================================================
# Intra-subject 학습 및 평가 (샘플 단위 stratified split)
if __name__ == "__main__":
    # 전체 CSP feature 로드 (증강 포함)
    CSP_FEATURE_DIR = "/home/bcml1/2025_EMOTION/DEAP_eeg_new_label_CSP"
    X, Y, subjects, file_ids = load_csp_features(CSP_FEATURE_DIR)
    
    unique_subjects = np.unique(subjects)
    for subj in unique_subjects:
        if not (2 <= int(subj) < 10):
            continue
        print(f"\n========== Intra-subject: Subject = {subj} ==========")
        subj_mask = (subjects == subj)
        X_subj = X[subj_mask]
        Y_subj = Y[subj_mask]
        
        # 만약 해당 subject에 존재하는 라벨이 일부만 있다면, 해당 라벨들만 사용하도록 재매핑합니다.
        unique_labels_subj = np.sort(np.unique(Y_subj))
        label_map = {old: new for new, old in enumerate(unique_labels_subj)}
        Y_subj = np.array([label_map[y] for y in Y_subj])
        num_classes = len(unique_labels_subj)
        print(f"Using labels: {unique_labels_subj} remapped to 0 ~ {num_classes-1}")

        # 샘플 단위 stratified split (파일 단위 split 대신)
        X_train, X_temp, Y_train, Y_temp = train_test_split(
            X_subj, Y_subj, test_size=0.3, random_state=42, stratify=Y_subj)
        X_val, X_test, Y_val, Y_test = train_test_split(
            X_temp, Y_temp, test_size=0.5, random_state=42, stratify=Y_temp)

        print(f"Subject {subj}: Train {X_train.shape[0]} samples, Val {X_val.shape[0]} samples, Test {X_test.shape[0]} samples.")
        print("Train label distribution:", np.unique(Y_train, return_counts=True))
        print("Val label distribution:", np.unique(Y_val, return_counts=True))
        print("Test label distribution:", np.unique(Y_test, return_counts=True))
        
        subj_folder = os.path.join(RESULT_DIR, f"s{subj}")
        os.makedirs(subj_folder, exist_ok=True)
        
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(1000).batch(16)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(16)
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(16)
        
        # 모델 생성 (subject별 실제 num_classes 사용)
        model = build_separated_model_with_diffusion(num_classes=num_classes, noise_std=0.02)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=1e-4, decay_steps=100000, decay_rate=0.9, staircase=True),
            clipnorm=1.0
        )
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=2000, mode='max', restore_best_weights=True)
        model.compile(optimizer=optimizer,
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        model.summary()
        
        history = model.fit(train_dataset, epochs=2000, validation_data=val_dataset, callbacks=[early_stopping])
        
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curve')
        
        plt.subplot(1,2,2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy Curve')
        train_curve_path = os.path.join(subj_folder, "training_curves.png")
        plt.savefig(train_curve_path)
        plt.close()
        print(f"Subject {subj}: Training curves saved to {train_curve_path}")
        
        test_loss, test_acc = model.evaluate(test_dataset)
        print(f"Subject {subj}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        
        y_pred_prob = model.predict(test_dataset)
        y_pred = np.argmax(y_pred_prob, axis=1)
        cm = confusion_matrix(Y_test, y_pred)
        report = classification_report(Y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(report)
        
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        cm_path = os.path.join(subj_folder, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        print(f"Subject {subj}: Confusion matrix saved to {cm_path}")
        
        report_path = os.path.join(subj_folder, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Subject {subj}: Classification report saved to {report_path}")
        
        model_save_path = os.path.join(subj_folder, "model_eeg_cnn_diffussm.keras")
        model.save(model_save_path)
        print(f"Subject {subj}: Model saved to {model_save_path}")
