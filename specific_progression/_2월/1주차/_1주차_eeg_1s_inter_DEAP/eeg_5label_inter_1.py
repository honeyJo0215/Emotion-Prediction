import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import LeaveOneGroupOut
from tensorflow.keras.layers import Conv3D, Dense, Flatten, LayerNormalization, Dropout, GlobalAveragePooling3D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set GPU Memory Limits
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
limit_gpu_memory()

# Data Path
DATA_PATH = "/home/bcml1/sigenv/_1주차_data_preprocess/test_DEAP_EEG_de_features_2D_mapping"
SAVE_PATH = "/home/bcml1/sigenv/_1주차_eeg_1s_inter/test_result"
os.makedirs(SAVE_PATH, exist_ok=True)

# Load Data with Session Information
def load_data(data_path):
    data, labels, groups = [], [], []
    subject_ids = [f"s{str(i).zfill(2)}" for i in range(1, 32)]
    
    for subject in subject_ids:
        subject_data_path = os.path.join(data_path, subject)
        if not os.path.exists(subject_data_path):
            continue
        
        # Assume each subject has multiple sessions, e.g., session_01 to session_05
        session_ids = [f"session_{str(s).zfill(2)}" for s in range(1, 6)]  # 예: 5개의 세션
        
        for session in session_ids:
            session_data_path = os.path.join(subject_data_path, session)
            if not os.path.exists(session_data_path):
                print(f"Session path does not exist: {session_data_path}")
                continue
            
            for label in range(5):  # Emotion classes
                for segment_idx in range(3, 63):  # Segment range
                    for sample_idx in range(40):  # 40 samples per segment
                        file_name = f"{subject}_{session}_sample_{str(sample_idx).zfill(2)}_segment_{str(segment_idx).zfill(3)}_label_{label}_2D.npy"
                        file_path = os.path.join(session_data_path, file_name)
                        
                        if os.path.exists(file_path):
                            try:
                                de_features = np.load(file_path)
                                de_features = np.expand_dims(de_features, axis=-1)
                                data.append(de_features)
                                labels.append(label)
                                groups.append(f"{subject}_{session}")  # 세션별 그룹 식별자
                            except Exception as e:
                                print(f"Error loading {file_path}: {e}")
    return np.array(data), np.array(labels), np.array(groups)

# Load EEG Data
data, labels, groups = load_data(DATA_PATH)

# Convert labels to integers
labels = labels.astype(np.int32)

# Data Preprocessing Function
def preprocess_data(data):
    data = tf.where(tf.math.is_nan(data), tf.zeros_like(data), data)  # Replace NaN with 0
    data = tf.where(tf.math.is_inf(data), tf.zeros_like(data), data)  # Replace Inf with 0
    # Uncomment the following lines if normalization is needed
    # min_val = tf.reduce_min(data)
    # max_val = tf.reduce_max(data)
    # range_val = max_val - min_val + 1e-8  # Prevent division by zero
    # data = (data - min_val) / range_val
    return data

# Spatial-Spectral Convolution Module
class SpatialSpectralConvModule(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides):
        super(SpatialSpectralConvModule, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.spatial_conv = Conv3D(filters, kernel_size=(1, kernel_size[1], kernel_size[2]), 
                                   strides=strides, padding="same", activation="relu")
        self.spectral_conv = Conv3D(filters, kernel_size=(kernel_size[0], 1, 1), 
                                   strides=strides, padding="same", activation="relu")

    def call(self, inputs):
        if len(inputs.shape) == 4:
            inputs = tf.expand_dims(inputs, axis=-1)
        spatial_features = self.spatial_conv(inputs)
        spectral_features = self.spectral_conv(inputs)
        return spatial_features + spectral_features
    
    def get_config(self):
        config = super(SpatialSpectralConvModule, self).get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
        })
        return config

# Spatial and Spectral Attention Branch
class SpatialSpectralAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(SpatialSpectralAttention, self).__init__()
        self.spatial_squeeze = Conv3D(1, kernel_size=(1, 1, 1), activation="sigmoid")
        self.spectral_squeeze = Conv3D(1, kernel_size=(1, 1, 1), activation="sigmoid")

    def call(self, inputs):
        if len(inputs.shape) == 4:  # (batch_size, depth, height, width)
            inputs = tf.expand_dims(inputs, axis=-1)  # (batch_size, depth, height, width, channels)
    
        # Spatial attention
        spatial_mask = self.spatial_squeeze(inputs)
        spatial_output = inputs * spatial_mask

        # Spectral attention
        spectral_mask = self.spectral_squeeze(inputs)
        spectral_output = inputs * spectral_mask

        # Combine spatial and spectral outputs
        combined_output = spatial_output + spectral_output
        # Debugging output
        print(f"Combined output shape: {combined_output.shape}")
        return combined_output
    
    def get_config(self):
        return super(SpatialSpectralAttention, self).get_config()

# Transformer Encoder Layer
class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(d_ff, activation=tf.nn.gelu),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, inputs, training=False):
        if len(inputs.shape) == 2:  # (batch_size, features)
            inputs = tf.expand_dims(inputs, axis=1)  # (batch_size, 1, features)
    
        attn_output = self.mha(inputs, inputs, training=training)  # Self-attention
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # Residual connection

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)  # Residual connection
    
    def get_config(self):
        config = super(TransformerEncoderLayer, self).get_config()
        config.update({
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "dropout_rate": self.dropout_rate,
        })
        return config

# Transformer Encoder Model
class TransformerEncoder(tf.keras.Model):
    def __init__(self, input_dim, n_layers=6, n_heads=8, d_ff=2048, p_drop=0.5, d_model=64):
        super(TransformerEncoder, self).__init__()
        # Instance variables
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.p_drop = p_drop
        self.d_model = d_model

        # Model components
        self.conv_block1 = SpatialSpectralConvModule(8, kernel_size=(1, 3, 3), strides=(1, 3, 3))
        self.conv_block2 = SpatialSpectralConvModule(16, kernel_size=(4, 1, 1), strides=(4, 1, 1))
        self.conv_block3 = SpatialSpectralConvModule(32, kernel_size=(1, 2, 2), strides=(1, 2, 2))
        self.attention = SpatialSpectralAttention()
        self.flatten = Flatten()
        self.dense_projection = Dense(d_model, activation="relu")
        self.encoder_layers = [
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout_rate=p_drop)
            for _ in range(n_layers)
        ]
        self.output_dense = Dense(5, activation="softmax")

    def build(self, input_shape):
        """Explicitly build the model."""
        super(TransformerEncoder, self).build(input_shape)

    def call(self, inputs, training=False):
        x = preprocess_data(inputs)
        x = self.attention(x)  # Apply attention first
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = self.dense_projection(x)
        for layer in self.encoder_layers:
            x = layer(x, training=training)
        return tf.squeeze(self.output_dense(x), axis=1)
    
    def get_config(self):
        config = super(TransformerEncoder, self).get_config()
        config.update({
            "input_dim": self.input_dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "p_drop": self.p_drop,
            "d_model": self.d_model,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Function to average model weights
def average_model_weights(model, model_paths):
    weights = []
    for path in model_paths:
        try:
            temp_model = TransformerEncoder(
                input_dim=model.input_dim,
                n_layers=model.n_layers,
                n_heads=model.n_heads,
                d_ff=model.d_ff,
                p_drop=model.p_drop,
                d_model=model.d_model
            )
            temp_model.build((None, *model.input_dim))  # Adjust input shape as needed
            temp_model.load_weights(path)
            weights.append(temp_model.get_weights())
        except Exception as e:
            print(f"Error loading weights from {path}: {e}")
            continue

    if len(weights) == 0:
        raise ValueError("No valid models were loaded. Cannot average weights.")

    avg_weights = [np.mean([weight[i] for weight in weights], axis=0) for i in range(len(weights[0]))]
    model.set_weights(avg_weights)
    return model

# Inter-Session Cross-Validation (LOSO)
logo = LeaveOneGroupOut()

for fold, (train_idx, test_idx) in enumerate(logo.split(data, labels, groups=groups)):
    train_data, test_data = data[train_idx], data[test_idx]
    train_labels, test_labels = labels[train_idx], labels[test_idx]
    
    unique_train_groups = set([groups[i] for i in train_idx])
    unique_test_groups = set([groups[i] for i in test_idx])
    print(f"Fold {fold+1}: Training on {len(unique_train_groups)} sessions, Testing on {len(unique_test_groups)} sessions")
    
    # Create Datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).shuffle(1000).batch(8)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(8)
    
    # Define Model
    input_shape = train_data.shape[1:]  # Example: (4, 6, 6, 1)
    model = TransformerEncoder(input_dim=input_shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    
    # Train Model
    model.fit(train_dataset, epochs=50, validation_data=test_dataset, callbacks=[early_stopping, lr_scheduler])
    
    # Evaluate Model
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Fold {fold+1} Test Accuracy: {test_acc:.4f}")
    
    # Save Model Weights
    # Since test_idx may contain multiple sessions, save weights per session
    test_groups = set([groups[i] for i in test_idx])
    for test_group in test_groups:
        subject_save_path = os.path.join(SAVE_PATH, f"group_{test_group}_model.h5")
        model.save_weights(subject_save_path)
        print(f"Model saved at {subject_save_path}")
    
    # Clear session to free memory
    tf.keras.backend.clear_session()

print("Inter-Session Cross-Validation Completed!")
