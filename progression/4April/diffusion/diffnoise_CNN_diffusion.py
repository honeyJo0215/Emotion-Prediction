import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# result directory
RESULT_DIR = "/home/bcml1/sigenv/_4월/_diffusion/A_rx_diffnoise_CNN_diff2"
os.makedirs(RESULT_DIR, exist_ok=True)

# gpu limit
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


class BidirectionalSSMLayer(layers.Layer):
    def __init__(self, units, kernel_size=3, **kwargs):
        """
        units: 입력/출력 차원 (== 피처 개수)
        kernel_size: causal conv의 커널 크기
        """
        super().__init__(**kwargs)
        self.units = units
        self.kernel_size = kernel_size
        self.fwd_conv = layers.Conv1D(
            filters=1,
            kernel_size=kernel_size,
            padding='causal',
            activation=None,
            name="ssm_fwd_conv"
        )
        self.bwd_conv = layers.Conv1D(
            filters=1,
            kernel_size=kernel_size,
            padding='causal',
            activation=None,
            name="ssm_bwd_conv"
        )

    def call(self, x, training=False):

        seq = tf.expand_dims(x, axis=-1)

        
        fwd = self.fwd_conv(seq)  # (batch, units, 1)

        rev_seq = tf.reverse(seq, axis=[1])
        bwd = self.bwd_conv(rev_seq)
        bwd = tf.reverse(bwd, axis=[1])  # (batch, units, 1)

        out = fwd + bwd
        return tf.squeeze(out, axis=-1)

# DiffuSSMLayer
class DiffuSSMLayer(layers.Layer):
    def __init__(self,                 
                 model_dim, # input/output dimension
                 hidden_dim, # bottleneck dimension
                 cond_dim=None, # conditioning input dimension (optional)
                 alpha_init=0.0, # fusion scale initial value
                 **kwargs):
        super().__init__(**kwargs)
        self.norm = layers.LayerNormalization()
        if cond_dim is not None:
            self.cond_mlp = layers.Dense(2 * model_dim, name="cond_mlp")
        else:
            self.cond_mlp = None

        self.hg_down = layers.Dense(hidden_dim, activation='swish', name="hourglass_down")
        self.ssm = BidirectionalSSMLayer(hidden_dim, kernel_size=3)    # → (batch, hidden_dim)
        self.hg_up = layers.Dense(model_dim, activation=None, name="hourglass_up")

        self.alpha = self.add_weight(
            name="fusion_scale_alpha",
            shape=(),
            initializer=tf.keras.initializers.Constant(alpha_init),
            trainable=True)

    def call(self, x, cond=None, training=False):
        # --- 1) Layer norm + (optional) conditioning scale&shift ---
        x_ln = self.norm(x)
        if self.cond_mlp is not None and cond is not None:
            # cond → [γ, β]
            gamma, beta = tf.split(self.cond_mlp(cond), num_or_size_splits=2, axis=-1)
            x_ln = gamma * x_ln + beta

        # --- 2) Hourglass down ---
        h = self.hg_down(x_ln)

        # --- 3) Bidirectional SSM  ---
        h = self.ssm(h)  

        # --- 4) Hourglass up ---
        h = self.hg_up(h)

        # --- 5) Residual with learnable scale α ---
        return x + self.alpha * h

# gausian noise
def add_diffusion_noise(x, stddev=0.05):
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=stddev)
    return x + noise
#####
# Frequency branch CNN block with pre- and post-diffusion
# input shape: (200,8,4)
# Frequency branch CNN block with pre- and post-diffusion
#####
def build_freq_branch_with_diffusion(input_shape=(200,8,4), noise_std=0.05):
    inp = layers.Input(shape=input_shape)

    # ----- Pre-CNN block -----
    pre_processed = layers.Lambda(lambda x: add_diffusion_noise(x, stddev=noise_std))(inp)

    # ----- CNN block -----
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(pre_processed)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)  # -> (64,)
    freq_feat = x

    # ----- Post-diffusion block -----
    noisy_freq_feat = layers.Lambda(lambda x: add_diffusion_noise(x, stddev=noise_std))(freq_feat)

    # explicitly pass model_dim=64, hidden_dim=64 here
    diffu_freq = DiffuSSMLayer(model_dim=64, hidden_dim=64)(noisy_freq_feat)

    freq_res = layers.Add()([freq_feat, diffu_freq])

    return models.Model(inputs=inp, outputs=freq_res, name="FreqBranchDiff")


# Channel branch CNN block with pre- and post-diffusion
def build_chan_branch_with_diffusion(input_shape=(200,4,8), noise_std=0.05):
    inp = layers.Input(shape=input_shape)

    # ----- Pre-CNN block -----
    pre_processed = layers.Lambda(lambda x: add_diffusion_noise(x, stddev=noise_std))(inp)

    # ----- CNN block -----
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(pre_processed)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)  # -> (64,)
    chan_feat = x

    # ----- Post-diffusion block -----
    noisy_chan_feat = layers.Lambda(lambda x: add_diffusion_noise(x, stddev=noise_std))(chan_feat)

    # model_dim=64, hidden_dim=64
    diffu_chan = DiffuSSMLayer(model_dim=64, hidden_dim=64)(noisy_chan_feat)

    chan_res = layers.Add()([chan_feat, diffu_chan])

    return models.Model(inputs=inp, outputs=chan_res, name="ChanBranchDiff")

# # Full model configuration: model combining frequency and channel branches (noise only in pre-CNN, DiffuSSM in post-CNN)
def build_separated_model_with_diffusion(num_classes=4, noise_std=0.05):
    # input shape: (4,200,8)
    input_csp = layers.Input(shape=(4,200,8), name="CSP_Input")
    
    # Frequency branch: Permute to (200,8,4)
    # (batch,4,200,8) -> (batch,200,8,4)
    freq_input = layers.Permute((2,3,1), name="Freq_Permute")(input_csp)
    freq_branch = build_freq_branch_with_diffusion(input_shape=(200,8,4), noise_std=noise_std)
    freq_feat = freq_branch(freq_input)  # (batch, 64)
    
    # Channel branch: Permute to (200,4,8)
    # (batch,4,200,8) -> (batch,200,4,8)
    chan_input = layers.Permute((2,1,3), name="Chan_Permute")(input_csp)
    chan_branch = build_chan_branch_with_diffusion(input_shape=(200,4,8), noise_std=noise_std)
    chan_feat = chan_branch(chan_input)  # (batch, 64)
    
    # two branches
    combined = layers.Concatenate()([freq_feat, chan_feat])  # (batch, 128)
    x = layers.Dense(128, activation='relu')(combined)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=input_csp, outputs=output, name="Separated_CNN_Model_with_Diffusion")
    return model

# load csp
def load_csp_features(csp_feature_dir="/home/bcml1/2025_EMOTION/SEED_IV/eeg_EM-CSP"):
    """
    지정된 디렉토리 내의 모든 npy 파일을 로드합니다.
    파일명 형식: folder{folder_num}_subject{subject}_sample{sample}_label{label}.npy
    각 파일은 (4, T, 8) 형태이며, 1초(200 샘플) 단위로 분할하여 (4,200,8) 샘플 생성.
    반환: X, Y, subjects, file_ids
    """
    X_list, Y_list, subjects_list, files_list = [], [], [], []
    for file_name in os.listdir(csp_feature_dir):
        if not file_name.endswith(".npy"):
            continue
        file_path = os.path.join(csp_feature_dir, file_name)
        try:
            data = np.load(file_path)  # shape: (4, T, 8)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
        if data.ndim != 3 or data.shape[0] != 4 or data.shape[2] != 8:
            print(f"Unexpected shape in {file_path}: {data.shape}")
            continue
        T = data.shape[1]
        n_windows = T // 200  
        if n_windows < 1:
            continue
        for i in range(n_windows):
            window = data[:, i*200:(i+1)*200, :]  # (4,200,8)
            X_list.append(window)
            subject_match = re.search(r'subject(\d+)', file_name, re.IGNORECASE)
            subject = subject_match.group(1) if subject_match else 'unknown'
            label_match = re.search(r'label(\d+)', file_name, re.IGNORECASE)
            label = int(label_match.group(1)) if label_match else -1
            Y_list.append(label)
            files_list.append(f"{file_name}_win{i}")
            subjects_list.append(subject)
    X = np.array(X_list)
    Y = np.array(Y_list)
    subjects = np.array(subjects_list)
    file_ids = np.array(files_list)
    
    valid_mask = Y >= 0
    X = X[valid_mask]
    Y = Y[valid_mask]
    subjects = subjects[valid_mask]
    file_ids = file_ids[valid_mask]
    
    print(f"Loaded {X.shape[0]} samples, each sample shape: {X.shape[1:]}")
    print(f"Unique labels found: {np.unique(Y)}")
    print(f"Unique subjects found: {np.unique(subjects)}")
    return X, Y, subjects, file_ids

# Intra-subject 
if __name__ == "__main__":
    CSP_FEATURE_DIR = "/home/bcml1/2025_EMOTION/SEED_IV/eeg_EM-CSP"
    X, Y, subjects, file_ids = load_csp_features(CSP_FEATURE_DIR)
    
    unique_subjects = np.unique(subjects)
    for subj in unique_subjects:
        if not (1 <= int(subj) < 16):
            continue
        print(f"\n========== Intra-subject: Subject = {subj} ==========")
        subj_mask = (subjects == subj)
        X_subj = X[subj_mask]
        Y_subj = Y[subj_mask]
        F_subj = file_ids[subj_mask]
        
        unique_files = np.unique(F_subj)
        if len(unique_files) < 2:
            print(f"Skip because there are not enough files in Subject {subj}.")
            continue
        
        train_files, test_files = train_test_split(unique_files, test_size=0.3, random_state=42)
        train_files_final, val_files = train_test_split(train_files, test_size=0.25, random_state=42)
        
        train_mask = np.isin(F_subj, train_files_final)
        val_mask = np.isin(F_subj, val_files)
        test_mask = np.isin(F_subj, test_files)
        
        X_train = X_subj[train_mask]
        Y_train = Y_subj[train_mask]
        X_val = X_subj[val_mask]
        Y_val = Y_subj[val_mask]
        X_test = X_subj[test_mask]
        Y_test = Y_subj[test_mask]
        
        print(f"Subject {subj}: Train {X_train.shape[0]} samples, Val {X_val.shape[0]} samples, Test {X_test.shape[0]} samples.")
        
        subj_folder = os.path.join(RESULT_DIR, f"s{subj}")
        os.makedirs(subj_folder, exist_ok=True)
        
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(1000).batch(64)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(64)
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)
        
        # 모델 생성: 입력 shape = (4,200,8)
        model = build_separated_model_with_diffusion(num_classes=4, noise_std=0.05)
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
