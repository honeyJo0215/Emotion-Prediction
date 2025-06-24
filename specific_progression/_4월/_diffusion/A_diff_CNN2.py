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
RESULT_DIR = "/home/bcml1/sigenv/_4월/_diffusion/A_diff_CNN2_improved"
os.makedirs(RESULT_DIR, exist_ok=True)

# =============================================================================
# GPU 메모리 제한
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
# Bidirectional SSM (1D causal conv forward+backward)
class BidirectionalSSMLayer(layers.Layer):
    def __init__(self, units, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_size = kernel_size
        self.fwd_conv = layers.Conv1D(filters=1, kernel_size=kernel_size,
                                      padding='causal', activation=None)
        self.bwd_conv = layers.Conv1D(filters=1, kernel_size=kernel_size,
                                      padding='causal', activation=None)

    def call(self, x, training=False):
        # x: (batch, channels)
        seq = tf.expand_dims(x, -1)              # → (batch, channels, 1)
        fwd = self.fwd_conv(seq)                 # → (batch, channels, 1)
        rev = tf.reverse(seq, axis=[1])
        bwd = self.bwd_conv(rev)
        bwd = tf.reverse(bwd, axis=[1])          # → (batch, channels, 1)
        out = fwd + bwd
        return tf.squeeze(out, -1)               # → (batch, channels)

# =============================================================================
# DiffuSSM Layer: Hourglass down↑SSM↑Hourglass up + residual
class DiffuSSMLayer(layers.Layer):
    def __init__(self, model_dim, hidden_dim, alpha_init=0.0, **kwargs):
        super().__init__(**kwargs)
        self.model_dim    = model_dim
        self.hidden_dim   = hidden_dim
        # LayerNorm + Hourglass + SSM + Hourglass + residual‐scale
        self.norm    = layers.LayerNormalization()
        self.hg_down = layers.Dense(hidden_dim, activation='swish')
        self.ssm     = BidirectionalSSMLayer(hidden_dim, kernel_size=3)
        self.hg_up   = layers.Dense(model_dim, activation=None)
        self.alpha   = self.add_weight(
            name="fusion_alpha", shape=(), 
            initializer=tf.keras.initializers.Constant(alpha_init),
            trainable=True
        )

    def call(self, x, training=False):
        # x: (batch, feature_dim)
        x_ln = self.norm(x)
        h    = self.hg_down(x_ln)
        h    = self.ssm(h, training=training)
        h    = self.hg_up(h)
        return x + self.alpha * h

    def compute_output_shape(self, input_shape):
        # TimeDistributed 에서 input_shape=(batch, timesteps, feature_dim)
        # 내부적으로 call() 에 전달되는 x 의 shape 는 (..., feature_dim)
        # 따라서 반환 shape 도 그대로 input_shape 으로 유지
        return input_shape

# =============================================================================
# Gaussian Noise
def add_diffusion_noise(x, stddev=0.05):
    return x + tf.random.normal(shape=tf.shape(x), stddev=stddev)

# =============================================================================
# Frequency branch: Conv Hourglass + TimeDistributed(DiffuSSM) + ConvTranspose → CNN
def build_freq_branch_with_diffusion(input_shape=(200,8,4), noise_std=0.05):
    inp = layers.Input(shape=input_shape)
    
    # 1) noisy input
    noisy = layers.Lambda(lambda x: add_diffusion_noise(x, stddev=noise_std))(inp)
    
    # 2) Conv-down (시공간 ↓)
    h = layers.Conv2D(32, (3,3), strides=(2,2), activation='relu', padding='same')(noisy)  # (100,4,32)
    h = layers.Conv2D(64, (3,3), strides=(2,2), activation='relu', padding='same')(h)      # (50,2,64)

    # 3) TimeDistributed DiffuSSM across 채널 dimension
    #    reshape to (batch, seq_len, features)
    seq_len = np.prod(h.shape[1:3], dtype=int)  # 50*2=100
    h_flat = layers.Reshape((seq_len, 64))(h)    # (batch,100,64)
    h_rec  = layers.TimeDistributed(
                 DiffuSSMLayer(model_dim=64, hidden_dim=32),
                 name="Freq_TD_DiffuSSM"
             )(h_flat)                          # (batch,100,64)
    h2     = layers.Reshape((50,2,64))(h_rec)    # (batch,50,2,64)

    # 4) ConvTranspose↑ to 원크기
    h2 = layers.Conv2DTranspose(32, (3,3), strides=(2,2), activation='relu', padding='same')(h2)  # (100,4,32)
    h2 = layers.Conv2DTranspose(4,  (3,3), strides=(2,2), activation='relu', padding='same')(h2)  # (200,8,4)
    
    # 5) residual 복원
    restored = layers.Add()([noisy, h2])
    
    # 6) CNN feature extraction
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(restored)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    feat = layers.GlobalAveragePooling2D()(x)  # (64,)

    return models.Model(inputs=inp, outputs=feat, name="FreqBranch")

# =============================================================================
# Channel branch: 동일 구조, 다른 input_shape
def build_chan_branch_with_diffusion(input_shape=(200,4,8), noise_std=0.05):
    inp = layers.Input(shape=input_shape)
    noisy = layers.Lambda(lambda x: add_diffusion_noise(x, stddev=noise_std))(inp)
    
    h = layers.Conv2D(32, (3,3), strides=(2,2), activation='relu', padding='same')(noisy)  # (100,2,32)
    h = layers.Conv2D(64, (3,3), strides=(2,2), activation='relu', padding='same')(h)      # (50,1,64)

    seq_len = np.prod(h.shape[1:3], dtype=int)  # 50*1=50
    h_flat = layers.Reshape((seq_len, 64))(h)    # (batch,50,64)
    h_rec  = layers.TimeDistributed(
                 DiffuSSMLayer(model_dim=64, hidden_dim=32),
                 name="Chan_TD_DiffuSSM"
             )(h_flat)                          # (batch,50,64)
    h2     = layers.Reshape((50,1,64))(h_rec)    # (batch,50,1,64)

    h2 = layers.Conv2DTranspose(32, (3,3), strides=(2,2), activation='relu', padding='same')(h2)  # (100,2,32)
    h2 = layers.Conv2DTranspose(8,  (3,3), strides=(2,2), activation='relu', padding='same')(h2)  # (200,4,8)

    restored = layers.Add()([noisy, h2])

    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(restored)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    feat = layers.GlobalAveragePooling2D()(x)  # (64,)

    return models.Model(inputs=inp, outputs=feat, name="ChanBranch")

# =============================================================================
# 전체 모델: 두 브랜치 결합 후 분류
def build_separated_model_with_diffusion(num_classes=4, noise_std=0.05):
    input_csp = layers.Input(shape=(4,200,8), name="CSP_Input")

    # permute for freq (4→channel dim)
    freq_in  = layers.Permute((2,3,1))(input_csp)  # (200,8,4)
    chan_in  = layers.Permute((2,1,3))(input_csp)  # (200,4,8)

    freq_feat = build_freq_branch_with_diffusion((200,8,4), noise_std)(freq_in)
    chan_feat = build_chan_branch_with_diffusion((200,4,8), noise_std)(chan_in)

    x = layers.Concatenate()([freq_feat, chan_feat])  # (128,)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs=input_csp, outputs=out, name="Separated_CNN_DiffuSSM")

# =============================================================================
# CSP feature 로드 (이전과 동일)
def load_csp_features(csp_feature_dir):
    X_list, Y_list, subjects, file_ids = [], [], [], []
    for fn in os.listdir(csp_feature_dir):
        if not fn.endswith(".npy"): continue
        data = np.load(os.path.join(csp_feature_dir, fn))
        if data.ndim!=3 or data.shape[0]!=4 or data.shape[2]!=8: continue
        T = data.shape[1]
        for i in range(T//200):
            X_list.append(data[:,i*200:(i+1)*200,:])
            lbl = int(re.search(r'label(\d+)', fn).group(1))
            subjects.append(re.search(r'subject(\d+)', fn).group(1))
            Y_list.append(lbl)
            file_ids.append(f"{fn}_win{i}")
    return (np.stack(X_list), np.array(Y_list),
            np.array(subjects), np.array(file_ids))

# =============================================================================
# 메인: Intra-subject 학습/평가 (이전과 동일 구조)
if __name__ == "__main__":
    CSP_DIR = "/home/bcml1/2025_EMOTION/SEED_IV/eeg_EM-CSP"
    X, Y, subjects, fids = load_csp_features(CSP_DIR)

    for subj in np.unique(subjects):
        if not (2 <= int(subj) < 10): continue
        mask = (subjects==subj)
        Xs, Ys, Fs = X[mask], Y[mask], fids[mask]
        if len(np.unique(Fs))<2: continue

        tr, te = train_test_split(np.unique(Fs), test_size=0.3, random_state=42)
        tr, va = train_test_split(tr, test_size=0.25, random_state=42)
        train_mask = np.isin(Fs, tr)
        val_mask   = np.isin(Fs, va)
        test_mask  = np.isin(Fs, te)

        subj_folder = os.path.join(RESULT_DIR, f"s{subj}")
        os.makedirs(subj_folder, exist_ok=True)

        X_train = Xs[train_mask]
        Y_train = Ys[train_mask]
        X_val = Xs[val_mask]
        Y_val = Ys[val_mask]
        X_test = Xs[test_mask]
        Y_test = Ys[test_mask]
        
        train_dataset = tf.data.Dataset.from_tensor_slices((Xs[train_mask], Ys[train_mask])).shuffle(1000).batch(64)
        val_dataset = tf.data.Dataset.from_tensor_slices((Xs[val_mask],   Ys[val_mask])).batch(64)
        test_dataset = tf.data.Dataset.from_tensor_slices((Xs[test_mask],  Ys[test_mask])).batch(64)

        model = build_separated_model_with_diffusion()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4, clipnorm=1.0),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        es = EarlyStopping("val_accuracy", patience=200, restore_best_weights=True)
        history = model.fit(train_dataset, epochs=1000, validation_data=val_dataset, callbacks=[es])
        
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
