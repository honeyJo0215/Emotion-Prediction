import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Dense, LayerNormalization, Dropout, Flatten, Input, Lambda, Concatenate, TimeDistributed, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Directory
RESULT_DIR = "/home/bcml1/sigenv/_6주차_eeg_test_ft/eeg_test_ft_5"
os.makedirs(RESULT_DIR, exist_ok=True)

# GPU limit
def limit_gpu_memory(memory_limit_mib=5000):
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

limit_gpu_memory(5000)

# Preprocess
class PreprocessLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        x = tf.where(tf.math.is_nan(inputs), tf.zeros_like(inputs), inputs)
        x = tf.where(tf.math.is_inf(x), tf.zeros_like(x), x)
        axis = list(range(1, inputs.shape.rank))
        min_val = tf.reduce_min(x, axis=axis, keepdims=True)
        max_val = tf.reduce_max(x, axis=axis, keepdims=True)
        range_val = max_val - min_val + 1e-8
        return (x - min_val) / range_val


def map_channels_to_2d(segment):
    num_channels, num_bands = segment.shape
    mapped = np.zeros((8, 8, num_bands))
    for band in range(num_bands):
        vec = segment[:, band]
        vec_padded = np.pad(vec, (0, 64 - num_channels), mode='constant')
        mapped[:, :, band] = vec_padded.reshape(8, 8)
    return mapped

def subject_normalize(X, S):
    X_norm = np.copy(X)
    unique_subjects = np.unique(S)
    for subj in unique_subjects:
        subj_idx = np.where(S == subj)[0]
        de_data = X[subj_idx, :, 0, ...]
        mean_de = np.mean(de_data)
        X_norm[subj_idx, :, 0, ...] = X_norm[subj_idx, :, 0, ...] - mean_de
        psd_data = X[subj_idx, :, 1, ...]
        mean_psd = np.mean(psd_data)
        X_norm[subj_idx, :, 1, ...] = X_norm[subj_idx, :, 1, ...] - mean_psd
    return X_norm

# Load data
def load_all_data(de_base_dir="/home/bcml1/2025_EMOTION/SEED_IV/eeg_DE",
                  psd_base_dir="/home/bcml1/2025_EMOTION/SEED_IV/eeg_PSD",
                  window_size=10, step_size=1):
    X, Y, S, F = [], [], [], []
    for subfolder in ["1", "2", "3"]:
        de_folder = os.path.join(de_base_dir, subfolder)
        psd_folder = os.path.join(psd_base_dir, subfolder)
        if not os.path.isdir(de_folder) or not os.path.isdir(psd_folder):
            continue
        for file_name in os.listdir(de_folder):
            if not file_name.endswith(".npy"):
                continue
            de_file_path = os.path.join(de_folder, file_name)
            psd_file_path = os.path.join(psd_folder, file_name)
            try:
                de_data = np.load(de_file_path)   # (62, n_seconds, 4)
                psd_data = np.load(psd_file_path)  # (62, n_seconds, 4)
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
                continue
            try:
                label = int(file_name.split('_')[-1].split('.')[0])
            except:
                print(f"Label extraction failed for {file_name}")
                continue
            subject_id = file_name.split('_')[0]
            n_seconds = de_data.shape[1]
            segments = []
            for t in range(n_seconds):
                de_segment = de_data[:, t, :]
                psd_segment = psd_data[:, t, :]
                de_mapped = map_channels_to_2d(de_segment)
                psd_mapped = map_channels_to_2d(psd_segment)
                de_img = np.transpose(de_mapped, (2, 0, 1))
                de_img = np.expand_dims(de_img, axis=-1)
                psd_img = np.transpose(psd_mapped, (2, 0, 1))
                psd_img = np.expand_dims(psd_img, axis=-1)
                sample = np.stack([de_img, psd_img], axis=0)  # (2, 4,8,8,1)
                segments.append(sample)
            if len(segments) >= window_size:
                for i in range(0, len(segments) - window_size + 1, step_size):
                    window = np.stack(segments[i:i+window_size], axis=0)  # (10,2,4,8,8,1)
                    X.append(window)
                    Y.append(label)
                    S.append(subject_id)
                    F.append(file_name)  
            print(f"Loaded {n_seconds} seconds from {file_name} with label {label} and subject {subject_id}.")
    X = np.array(X)
    Y = np.array(Y)
    S = np.array(S)
    F = np.array(F)
    print(f"Total samples (windows): {X.shape[0]}, each sample shape: {X.shape[1:]}")
    return X, Y, S, F

X, Y, subjects, file_ids = load_all_data(window_size=10, step_size=1)
X = subject_normalize(X, subjects)

# Model configuration (here the same CNN+Transformer structure as before)
def build_single_segment_model(d_model=64):
    input_seg = Input(shape=(2, 4, 8, 8, 1))
    de_input = Lambda(lambda x: x[:, 0, ...], name="DE_input")(input_seg)
    psd_input = Lambda(lambda x: x[:, 1, ...], name="PSD_input")(input_seg)
    cnn_de = CNNBranchLayer(d_model, name="CNN_DE")
    cnn_psd = CNNBranchLayer(d_model, name="CNN_PSD")
    feature_de = cnn_de(de_input)
    feature_psd = cnn_psd(psd_input)
    combined_features = Concatenate()([feature_de, feature_psd])
    combined_features = Dense(d_model, activation="relu", kernel_regularizer=l2(1e-5))(combined_features)
    model = Model(inputs=input_seg, outputs=combined_features)
    return model

class CNNBranchLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, **kwargs):
        super(CNNBranchLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.preprocess = PreprocessLayer()
        self.attention = SpatialSpectralAttention()
        self.conv1 = SpatialSpectralConvModule(8, kernel_size=(3,3), strides=(3,3))
        self.conv2 = SpatialSpectralConvModule(16, kernel_size=(1,1), strides=(1,1))
        self.conv3 = SpatialSpectralConvModule(32, kernel_size=(1,1), strides=(1,1))
        self.flatten = Flatten()
        self.dense = Dense(d_model, activation="relu", kernel_regularizer=l2(1e-5))
    
    def call(self, x):
        x = self.preprocess(x)
        x = self.attention(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.d_model)

class SpatialSpectralConvModule(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides):
        super(SpatialSpectralConvModule, self).__init__()
        self.spatial_conv = Conv3D(filters, kernel_size=(1, *kernel_size), strides=(1, *strides),
                                   padding="same", activation="relu",
                                   kernel_regularizer=l2(1e-5))
        self.spectral_conv = Conv3D(filters, kernel_size=(kernel_size[0], 1, 1), strides=(1, *strides),
                                    padding="same", activation="relu",
                                    kernel_regularizer=l2(1e-5))
    
    def call(self, inputs):
        spatial_features = self.spatial_conv(inputs)
        spectral_features = self.spectral_conv(inputs)
        return spatial_features + spectral_features
    
    def get_config(self):
        return {"filters": self.spatial_conv.filters,
                "kernel_size": self.spatial_conv.kernel_size,
                "strides": self.spatial_conv.strides}

class SpatialSpectralAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(SpatialSpectralAttention, self).__init__()
        self.spatial_squeeze = Conv3D(1, kernel_size=(1,1,1), activation="sigmoid")
        self.spectral_squeeze = Conv3D(1, kernel_size=(1,1,1), activation="sigmoid")
    
    def call(self, inputs):
        spatial_mask = self.spatial_squeeze(inputs)
        spectral_mask = self.spectral_squeeze(inputs)
        return inputs * spatial_mask + inputs * spectral_mask
    
    def get_config(self):
        return {}

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.5):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(d_ff, activation=tf.nn.gelu, kernel_regularizer=l2(1e-5)),
            Dense(d_model, kernel_regularizer=l2(1e-5))
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
    
    def call(self, inputs, training=False):
        attn_output = self.mha(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        return {"d_model": self.mha.key_dim * self.mha.num_heads,
                "n_heads": self.mha.num_heads,
                "d_ff": self.ffn.layers[0].units,
                "dropout_rate": self.dropout1.rate}

# The model uses the TimeDistributed + Transformer structure as before here
def build_model(input_shape=(10, 2, 4, 8, 8, 1), n_layers=2, n_heads=4, d_ff=512, p_drop=0.5, d_model=64):
    inputs = Input(shape=input_shape)  # (batch, 10, 2,4,8,8,1)
    single_seg_model = build_single_segment_model(d_model)
    features_seq = TimeDistributed(single_seg_model, name="TimeDistributed_CNN")(inputs)  # (batch, 10, d_model)
    x = features_seq
    for _ in range(n_layers):
        x = TransformerEncoderLayer(d_model, n_heads, d_ff, dropout_rate=p_drop)(x)
    x = Flatten()(x)
    outputs = Dense(4, activation="softmax", kernel_regularizer=l2(1e-5))(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Fine tuning separation function: For each test file (sample), the middle 20% of the total window (e.g. 40-60%) as fine tuning,
# Separate the rest as test data (window order is maintained within the file)
def split_finetune_test_by_file(X_test, Y_test, F_test, ft_ratio=0.2):
    X_ft_list, X_test_final_list = [], []
    y_ft_list, y_test_final_list = [], []
    unique_files = np.unique(F_test)
    for f in unique_files:
        indices = np.where(F_test == f)[0]
        # indices is assumed to be ordered in the file
        n = len(indices)
        start = int(n * 0.4)
        end = int(n * 0.6)
        ft_idx = indices[start:end]
        test_idx = np.concatenate([indices[:start], indices[end:]])
        X_ft_list.append(X_test[ft_idx])
        y_ft_list.append(Y_test[ft_idx])
        X_test_final_list.append(X_test[test_idx])
        y_test_final_list.append(Y_test[test_idx])
    X_ft = np.concatenate(X_ft_list, axis=0)
    y_ft = np.concatenate(y_ft_list, axis=0)
    X_test_final = np.concatenate(X_test_final_list, axis=0)
    y_test_final = np.concatenate(y_test_final_list, axis=0)
    return X_ft, X_test_final, y_ft, y_test_final

# Inter-Subject learning (LOSO) + fine tuning for test subjects
X_all, Y_all, S_all, F_all = X, Y, subjects, file_ids
unique_subjects = np.unique(S_all)

BATCH_SIZE = 64
initial_lr = 1e-4
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=100000,
    decay_rate=0.9,
    staircase=True
)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, mode='max', restore_best_weights=True)

for subj in unique_subjects:
    print(f"\n========== Inter-Subject: Test subject = {subj} ==========")
    test_mask = (S_all == subj)
    train_valid_mask = (S_all != subj)
    
    X_test = X_all[test_mask]
    Y_test = Y_all[test_mask]
    F_test = F_all[test_mask]
    X_train_valid = X_all[train_valid_mask]
    Y_train_valid = Y_all[train_valid_mask]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_valid, Y_train_valid, test_size=0.25, random_state=42, stratify=Y_train_valid
    )
    
    # Here we separate fine tuning and test data for each test file
    X_ft, X_test_final, y_ft, y_test_final = split_finetune_test_by_file(X_test, Y_test, F_test, ft_ratio=0.2)
    
    subj_folder = os.path.join(RESULT_DIR, f"s{subj.zfill(2)}")
    os.makedirs(subj_folder, exist_ok=True)
    
    with tf.device('/CPU:0'):
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(BATCH_SIZE)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
    
    model = build_model(input_shape=(10, 2, 4, 8, 8, 1))
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()
    
    history = model.fit(train_dataset, epochs=150, validation_data=val_dataset, callbacks=[early_stopping])
    
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
    
    # Fine tuning steps: Use the fine tuning data isolated from the test files
    ft_dataset = tf.data.Dataset.from_tensor_slices((X_ft, y_ft)).shuffle(1000).batch(BATCH_SIZE)
    print(f"Subject {subj}: Fine tuning with {X_ft.shape[0]} samples.")
    model.fit(ft_dataset, epochs=20)
    
    # Final test evaluation: Using the remaining data
    test_dataset_final = tf.data.Dataset.from_tensor_slices((X_test_final, y_test_final)).batch(BATCH_SIZE)
    test_loss, test_acc = model.evaluate(test_dataset_final)
    print(f"Subject {subj}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    y_pred_prob = model.predict(test_dataset_final)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    cm = confusion_matrix(y_test_final, y_pred)
    report = classification_report(y_test_final, y_pred)
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
    
    model_save_path = os.path.join(subj_folder, "model_eeg_cnn_transformer.keras")
    model.save(model_save_path)
    print(f"Subject {subj}: Model saved to {model_save_path}")
