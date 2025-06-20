import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, KFold
from sklearn.preprocessing import StandardScaler
import os

# Define Multi-Head Self Attention
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads

        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)
        self.dropout = layers.Dropout(0.5)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        weights = self.dropout(weights)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, _ = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output

# Define Feed Forward Network (FFN)
class FeedForwardNetwork(layers.Layer):
    def __init__(self, embed_dim, dense_units, **kwargs):
        super(FeedForwardNetwork, self).__init__(**kwargs)
        self.dense1 = layers.Dense(dense_units, activation=tf.nn.gelu)
        self.dropout = layers.Dropout(0.5)
        self.dense2 = layers.Dense(embed_dim)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dropout(x)
        x = self.dense2(x)
        return x

# Define Temporal Encoding Layer
class TemporalEncodingLayer(layers.Layer):
    def __init__(self, embed_dim, num_heads, ffn_units, **kwargs):
        super(TemporalEncodingLayer, self).__init__(**kwargs)
        self.mhsa = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_units)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        attn_output = self.mhsa(inputs)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

# Define the model
class ExpandLayer(layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super(ExpandLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)


# Define Model
def build_cnn(input_shape, embed_dim):
    inputs = layers.Input(shape=input_shape)

    # Spatial-Spectral CNN
    x = layers.Conv3D(16, kernel_size=(1, 3, 3), activation='relu', padding='same')(inputs)
    #x = layers.MaxPooling3D(pool_size=(1, 1, 2), padding='same')(x)
    x = layers.Conv3D(32, kernel_size=(5, 1, 1), activation='relu', padding='same')(x)
    #x = layers.MaxPooling3D(pool_size=(1, 1, 2), padding='same')(x)
    x = layers.Conv3D(64, kernel_size=(1, 3, 3), activation='relu', padding='same')(x)
    #x = layers.MaxPooling3D(pool_size=(1, 1, 2), padding='same')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(embed_dim, activation='')(x)

    # # Transformer Layers
    # for _ in range(3):
    #     x = TemporalEncodingLayer(embed_dim, num_heads, ffn_units)(x)

    # x = layers.GlobalAveragePooling1D()(x)
    # x = layers.Dense(66, activation=tf.nn.gelu)(x)
    # x = layers.Dropout(0.5)(x)
    # outputs = layers.Dense(num_classes, activation='softmax')(x)

    # model = models.Model(inputs, outputs)
    return x

def build_model(input_shape, num_classes, embed_dim, num_heads, ffn_units):
    inputs = layers.Input(shape=input_shape)


    x = build_cnn(input_shape,embed_dim)
    
    # Transformer Layers
    for _ in range(3):
        x = TemporalEncodingLayer(embed_dim, num_heads, ffn_units)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(66, activation=tf.nn.gelu)(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    
    return model


# Load Data
def load_data(data_dir):
    X, y, subjects = [], [], []
    print(f"Loading data from: {data_dir}")

    for file_name in os.listdir(data_dir):
        if file_name.endswith("_FB.npy"):
            file_path = os.path.join(data_dir, file_name)
            label = 1 if "positive" in file_name else 0
            subject_id = file_name.split("_")[0]  # Extract subject ID

            try:
                data = np.load(file_path)
                reshaped_data = np.transpose(data, (0, 2, 1)) if data.ndim == 3 else data
                X.append(reshaped_data)
                y.append(label)
                subjects.append(subject_id)
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

    if len(X) != len(y):
        raise ValueError(f"Data mismatch: X has {len(X)} samples, y has {len(y)} labels.")

    print(f"Loaded {len(X)} samples. Labels: {np.bincount(y)}")
    return np.array(X), np.array(y), np.array(subjects)

# Train and Evaluate with Intra-Subject Strategy
def train_and_evaluate_intra_subject(data_dir, model_save_path,k=10):
    X, y, subjects = load_data(data_dir)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # # 피험자별 정규화
    # unique_subjects = np.unique(subjects)
    # for subject in unique_subjects:
    #     idx = (subjects == subject)
    #     scaler = StandardScaler()
    #     X[idx] = scaler.fit_transform(X[idx].reshape(len(X[idx]), -1)).reshape(X[idx].shape)

    # logo = LeaveOneGroupOut()
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_model(input_shape=X_train.shape[1:], num_classes=2, embed_dim=64, num_heads=2, ffn_units=128)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=1)

              
        save_path = os.path.join(model_save_path, "1s_intra_subject.h5")
        model.save(save_path)
        print(f"Model saved at {save_path}")


        
    print("Intra-Subject Training Completed.")

# 실행
data_directory = "/path/to/your/data"  # 데이터 경로 수정
model_save_file = "intra_subject_model.h5"

train_and_evaluate_intra_subject(data_directory, model_save_file)
