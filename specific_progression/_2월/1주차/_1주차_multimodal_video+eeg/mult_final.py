
import os
import re
import cv2
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Lambda, Reshape, Input, Dense, Flatten, Concatenate, Conv3D, BatchNormalization, Dropout, LayerNormalization
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# GPU memory limit
def limit_gpu(memory_limit_mib=8000):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit_mib)]
            )
            print(f"GPU memory limited: {memory_limit_mib}")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU available.")

limit_gpu(8000)

#DATA label maping
EEG_DATA_PATH = "/home/bcml1/2025_EMOTION/DEAP_EEG_2D_DE"
EYE_CROP_PATH = "/home/bcml1/2025_EMOTION/eye_crop"
SAVE_PATH = "/home/bcml1/myenv/DEAP_Multimodal_final"
os.makedirs(SAVE_PATH, exist_ok=True)

EMOTION_MAPPING = {
    "Excited": 0,
    "Relaxed": 1,
    "Stressed": 2,
    "Bored": 3,
    "Neutral": 4
}

# Positional encoding
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model

    def build(self, input_shape):
        # build
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(self.sequence_length, self.d_model),
            initializer="uniform",
            trainable=True
        )
        super(PositionalEncoding, self).build(input_shape)

    def call(self, inputs):
        # inputs: (batch, sequence_length, d_model)
        return inputs + self.pos_embedding

# model

# Spatial-Spectral Convolution Module
class SpatialSpectralConvModule(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides):
        super(SpatialSpectralConvModule, self).__init__()
        self.spatial_conv = Conv3D(filters, kernel_size=(1, 3, 3), strides=strides,
                                   padding="same", activation="relu")
        self.spectral_conv = Conv3D(filters, kernel_size=(4, 1, 1), strides=strides,
                                    padding="same", activation="relu")
        self.bn = BatchNormalization()

    def call(self, inputs):
        if len(inputs.shape) == 4:
            inputs = tf.expand_dims(inputs, axis=-1)
        spatial_features = self.spatial_conv(inputs)
        spectral_features = self.spectral_conv(inputs)
        combined = spatial_features + spectral_features
        return self.bn(combined)

# Spatial and Spectral Attention 
class SpatialSpectralAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(SpatialSpectralAttention, self).__init__()
        self.spatial_squeeze = Conv3D(1, kernel_size=(1, 1, 1), activation="sigmoid")
        self.spectral_squeeze = Conv3D(1, kernel_size=(1, 1, 1), activation="sigmoid")

    def call(self, inputs):
        if len(inputs.shape) == 4:
            inputs = tf.expand_dims(inputs, axis=-1)
        spatial_mask = self.spatial_squeeze(inputs)
        spectral_mask = self.spectral_squeeze(inputs)
        spatial_output = inputs * spatial_mask
        spectral_output = inputs * spectral_mask
        return spatial_output + spectral_output

# EEG CNN
class EEGCNN(tf.keras.Model):
    def __init__(self, d_model):
        super(EEGCNN, self).__init__()
        self.conv_block1 = SpatialSpectralConvModule(8, kernel_size=(1, 3, 3), strides=(1, 3, 3))
        self.conv_block2 = SpatialSpectralConvModule(16, kernel_size=(4, 1, 1), strides=(4, 1, 1))
        self.conv_block3 = SpatialSpectralConvModule(32, kernel_size=(1, 2, 2), strides=(1, 2, 2))
        self.attention = SpatialSpectralAttention()
        self.flatten = Flatten()
        self.dense_projection = Dense(d_model, activation="relu")
        self.dropout = Dropout(0.3)

    def call(self, inputs):
        x = self.attention(inputs)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = self.dense_projection(x)
        return self.dropout(x)

# Eye Crop CNN
def build_eye_crop_model(d_model=64):
    eye_input = Input(shape=(50, 8, 64, 3))  # 입력 크기 (프레임 수, 높이, 너비, 채널)
    x = Reshape((50, 8, 64, 3))(eye_input)
    x = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(d_model, activation='relu')(x)
    x = Dropout(0.3)(x)
    return Model(eye_input, x, name="EyeCrop_CNN")

# Transformer Encoder Layer
class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
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
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, axis=1)
        attn_output = self.mha(query=inputs, key=inputs, value=inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Cross-Modal Transformer Layer
class CrossModalTransformer(tf.keras.layers.Layer):
    def __init__(self, input_dim, hidden_dim=128, num_heads=4):
        super(CrossModalTransformer, self).__init__()
        self.cross_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_dim)
        self.ffn = tf.keras.Sequential([
            Dense(hidden_dim, activation='relu'),
            Dense(input_dim)
        ])
        self.layer_norm1 = LayerNormalization()
        self.layer_norm2 = LayerNormalization()

    def call(self, query_input, key_value_input):
        attention_output = self.cross_attention(query=query_input,
                                                  key=key_value_input,
                                                  value=key_value_input)
        attention_output = self.layer_norm1(query_input + attention_output)
        ffn_output = self.ffn(attention_output)
        return self.layer_norm2(attention_output + ffn_output)

# Transformer Encoder build
def build_transformer_encoder(input_dim, n_layers=2, n_heads=4, d_ff=512,
                              dropout_rate=0.1, d_model=64, name="Transformer_Encoder"):
    inputs = Input(shape=(input_dim, d_model))
    x = inputs
    for _ in range(n_layers):
        x = TransformerEncoderLayer(d_model=d_model, n_heads=n_heads,
                                    d_ff=d_ff, dropout_rate=dropout_rate)(x)
    return Model(inputs, x, name=name)

# model build
def build_multimodal_model(eeg_input_shape, eye_input_shape=(50, 8, 64, 3)):
    
    eeg_input = Input(shape=eeg_input_shape)
    eye_input = Input(shape=eye_input_shape)

    
    eeg_cnn_model = EEGCNN(d_model=64)
    eeg_features = eeg_cnn_model(eeg_input)  # (batch, 64)

    
    eye_cnn_model = build_eye_crop_model(d_model=64)
    eye_features = eye_cnn_model(eye_input)  # (batch, 64)

    
    expand_dims_layer = Lambda(lambda x: tf.tile(x[:, None, :], [1, 64, 1]))
    eeg_seq = expand_dims_layer(eeg_features)  # (batch, 64, 64)
    eye_seq = expand_dims_layer(eye_features)  # (batch, 64, 64)

    # Positional Encoding
    eeg_seq = PositionalEncoding(sequence_length=64, d_model=64)(eeg_seq)
    eye_seq = PositionalEncoding(sequence_length=64, d_model=64)(eye_seq)

    # Intra-modality Transformer
    eeg_transformer = build_transformer_encoder(input_dim=64, name="Transformer_Encoder_EEG")
    eye_transformer = build_transformer_encoder(input_dim=64, name="Transformer_Encoder_EYE")

    eeg_transformed = eeg_transformer(eeg_seq)  # (batch, 64, 64)
    eye_transformed = eye_transformer(eye_seq)  # (batch, 64, 64)

    # Cross-Modal Transformer
    cross_modal_eeg_q_eye_kv = CrossModalTransformer(input_dim=64)(eeg_transformed, eye_transformed)
    cross_modal_eye_q_eeg_kv = CrossModalTransformer(input_dim=64)(eye_transformed, eeg_transformed)


    inter_input = Concatenate(axis=-1)([eeg_transformed, eye_transformed,
                                         cross_modal_eeg_q_eye_kv, cross_modal_eye_q_eeg_kv])
    # Classification Head
    x = Dense(128, activation="relu")(inter_input)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(5, activation="softmax")(x)
    # Mean Pooling
    output = Lambda(lambda x: tf.reduce_mean(x, axis=1))(x)

    model = Model(inputs=[eeg_input, eye_input], outputs=output, name="Multimodal_CrossTransformer_Model")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss="sparse_categorical_crossentropy", 
                  metrics=["accuracy"])
    return model

# functions
def downsample_eye_frame(frame):
    return cv2.resize(frame, (32, 8), interpolation=cv2.INTER_AREA)

def reshape_eye_frame(data):
   
    if len(data.shape) == 4 and data.shape[0] > 0:
        reshaped_data = np.mean(data, axis=0)
        return downsample_eye_frame(reshaped_data)
    elif len(data.shape) == 3 and data.shape == (32, 64, 3):
        return downsample_eye_frame(data)
    else:
        raise ValueError(f"Unexpected shape: {data.shape}, expected (N, 32, 64, 3) or (32, 64, 3)")

def load_multimodal_data(subject):

    eeg_data, eye_data, labels = [], [], []
    eeg_pattern = re.compile(rf"{subject}_sample_(\d+)_segment_(\d+)_label_(\w+)_2D_DE.npy")

    for sample_index in range(1):  # Sample 000 ~ Sample 039
        sample_number = f"{sample_index:02d}"
        print(f"\nProcessing {subject} - Sample {sample_number}")

        eeg_files = [f for f in os.listdir(EEG_DATA_PATH) if eeg_pattern.match(f) and f"sample_{sample_number}" in f]
        if not eeg_files:
            print(f"No EEG file found for {subject} - Sample {sample_number}")
            continue

        for file_name in eeg_files:
            match = eeg_pattern.match(file_name)
            if not match:
                continue

            segment_index = int(match.group(2))
            emotion_label = match.group(3)

            # ignore empty signal
            if segment_index < 3:
                continue
            segment_index -= 3

            eeg_file_path = os.path.join(EEG_DATA_PATH, file_name)
            eeg_segment = np.load(eeg_file_path)

            eye_subject_path = os.path.join(EYE_CROP_PATH, subject)
            if not os.path.exists(eye_subject_path):
                print(f"Subject folder not found: {eye_subject_path}")
                continue

            trial_number = sample_index + 1
            start_frame = segment_index * 50
            end_frame = start_frame + 50

            print(f"  🔹 Segment {segment_index}: Expected frames {start_frame} to {end_frame}, Matching Trial {trial_number:02d}")

            frame_indices = set()
            file_mapping = {}
            for f in os.listdir(eye_subject_path):
                try:
                    if not f.startswith(subject) or not f.endswith(".npy"):
                        continue
                    match = re.search(r"trial(\d+).avi_frame(\d+)", f)
                    if not match:
                        print(f"⚠ Skipping invalid file name: {f}")
                        continue
                    file_trial_number = int(match.group(1))
                    frame_number = int(match.group(2))
                    if file_trial_number == trial_number:
                        frame_indices.add(frame_number)
                        file_mapping[frame_number] = os.path.join(eye_subject_path, f)
                except ValueError as e:
                    print(f"Error processing file {f}: {e}")
                    continue

            frame_indices = sorted(frame_indices)
            if frame_indices:
                print(f"Available frame indices: {frame_indices[:10]} ... {frame_indices[-10:]}")
            else:
                print("No frame indices found.")

            selected_frames = sorted([frame for frame in frame_indices if start_frame <= frame < end_frame])
            if len(selected_frames) == 0:
                print(f"Warning: No frames found for segment {segment_index}. Skipping Eye Crop.")
                eye_data.append(None)
                eeg_data.append(eeg_segment)
                labels.append(EMOTION_MAPPING[emotion_label])
                continue

            if len(selected_frames) < 50:
                print(f"Warning: Found only {len(selected_frames)} frames for segment {segment_index}")
                while len(selected_frames) < 50:
                    selected_frames.append(selected_frames[-1])
                    print("frame copied")

            eye_frame_files = []
            for frame in selected_frames:
                if frame in file_mapping:
                    eye_frame_files.append(file_mapping[frame])
                if len(eye_frame_files) == 50:
                    break

            eye_frame_stack = []
            for f in eye_frame_files:
                frame_data = np.load(f)
                frame_data = reshape_eye_frame(frame_data)
                if frame_data.shape[-2] == 32:
                    pad_width = [(0, 0)] * frame_data.ndim
                    pad_width[-2] = (16, 16)
                    frame_data = np.pad(frame_data, pad_width, mode='constant', constant_values=0)
                eye_frame_stack.append(frame_data)

            if len(eye_frame_stack) == 50:
                eye_data.append(np.stack(eye_frame_stack, axis=0))
                eeg_data.append(eeg_segment)
                labels.append(EMOTION_MAPPING[emotion_label])
            else:
                print(f"⚠ Warning: Found only {len(eye_frame_stack)} matching frames for segment {segment_index}")

    print(f"✅ Total EEG Samples Loaded: {len(eeg_data)}")
    print(f"✅ Total Eye Crop Samples Loaded: {len([e for e in eye_data if e is not None])}")
    print(f"✅ Labels Loaded: {len(labels)}")
    eye_data = np.array([e if e is not None else np.zeros((50, 8, 64, 3)) for e in eye_data])
    return np.array(eeg_data), eye_data, np.array(labels)

# Test

def train_multimodal():
    subjects = [f"s{str(i).zfill(2)}" for i in range(9, 23)]
    
    for subject in subjects:
        print(f"\n===== Training subject: {subject} =====")

        
        eeg_data, eye_data, labels = load_multimodal_data(subject)

        unique_samples = np.arange(len(eeg_data))
        train_samples, test_samples = train_test_split(unique_samples, test_size=0.2, random_state=42)
        train_samples, valid_samples = train_test_split(train_samples, test_size=0.2, random_state=42)

        train_eeg, train_eye, train_labels = eeg_data[train_samples], eye_data[train_samples], labels[train_samples]
        valid_eeg, valid_eye, valid_labels = eeg_data[valid_samples], eye_data[valid_samples], labels[valid_samples]
        test_eeg, test_eye, test_labels = eeg_data[test_samples], eye_data[test_samples], labels[test_samples]

        checkpoint_dir = f"/home/bcml1/myenv/DEAP_Multimodal_final/checkpoint/{subject}"
        checkpoint_path = os.path.join(checkpoint_dir, "cp.weights.h5")
        os.makedirs(checkpoint_dir, exist_ok=True)  # 디렉토리 없으면 생성

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            verbose=1
        )

        model = build_multimodal_model(eeg_input_shape=train_eeg.shape[1:])
        print(model.summary())

        if os.path.exists(checkpoint_path + ".index"):
            print(f"✅ Checkpoint found for {subject}, loading model...")
            model.load_weights(checkpoint_path)

        train_labels = np.expand_dims(train_labels, axis=-1)
        valid_labels = np.expand_dims(valid_labels, axis=-1)

        start_epoch = 0
        max_epochs = 50
        batch_size = 2
        max_retries = 3 

        for epoch in range(start_epoch, max_epochs):
            retries = 0
            while retries < max_retries:
                try:
                    print(f"\n🚀 Training {subject} - Epoch {epoch+1}/{max_epochs} (Retry: {retries})...")
                    model.fit(
                        [train_eeg, train_eye], train_labels,
                        validation_data=([valid_eeg, valid_eye], valid_labels),
                        epochs=1, batch_size=batch_size,
                        callbacks=[checkpoint_callback]
                    )
                    break  
                except tf.errors.ResourceExhaustedError:
                    print(f"OOM! Checkpoint saving.. (Retry: {retries+1})...")

                    try:
                        model.save_weights(checkpoint_path)
                    except tf.errors.ResourceExhaustedError:
                        print("Check point OOM")
                    

                    tf.keras.backend.clear_session()
                    gc.collect()
                    
                    model = build_multimodal_model(eeg_input_shape=train_eeg.shape[1:])
                    if os.path.exists(checkpoint_path + ".index"):
                        model.load_weights(checkpoint_path)
                    
                    retries += 1
                    tf.keras.backend.sleep(1)
            else:
                print(f"!!epoch {epoch+1} interupped. Next subject processing...")
                break  

        # model save
        subject_save_path = os.path.join(SAVE_PATH, subject)
        os.makedirs(subject_save_path, exist_ok=True)
        weight_path = os.path.join(subject_save_path, f"{subject}_multimodal_model.weights.h5")
        model.save_weights(weight_path)
        print(f"model saved: {weight_path}")

        # test
        predictions = model.predict([test_eeg, test_eye])
        predicted_labels = np.argmax(predictions, axis=-1)
        test_report = classification_report(
            test_labels, predicted_labels,
            target_names=["Excited", "Relaxed", "Stressed", "Bored", "Neutral"],
            labels=[0, 1, 2, 3, 4],
            zero_division=0
        )
        print(f"\nTest Report for {subject}")
        print(test_report)

        report_path = os.path.join(subject_save_path, f"{subject}_test_report.txt")
        with open(report_path, "w") as f:
            f.write(test_report)
        print(f"✅ Test report: {report_path}")

# =============================================================================
# 메인 실행부
# =============================================================================
if __name__ == "__main__":
    train_multimodal()
