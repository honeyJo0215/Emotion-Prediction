import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report
import os

# tf.config.optimizer.set_jit(False)

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
        print(f"transpose 이후 attention의 shape: {attention.shape}")
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
        print(f"ffn의 input의 모양:{inputs.shape}")
        x = self.dense1(inputs)
        print(f"dense1 layer이후:{inputs.shape}")
        x = self.dropout(x)
        x = self.dense2(x)
        print(f"dense2 layer이후:{inputs.shape}")
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
        print(f"transformer layer에 들어오는 input shape: {inputs.shape}")
        attn_output = self.mhsa(inputs)
        print(f"mhsa layer이후 shape: {attn_output.shape}")
        out1 = self.layernorm1(inputs + attn_output) # Residual Connection
        print(f"norm1 layer이후 shape: {out1.shape}")
        ffn_output = self.ffn(out1)
        print(f"ffn layer이후 shape: {ffn_output.shape}")
        out2 = self.layernorm2(out1 + ffn_output)   # Residual Connection
        print(f"norm2 layer이후 shape: {out2.shape}")
        return out2

# Define the model
class ExpandLayer(layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super(ExpandLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)


def build_model(input_shape, num_classes, embed_dim, num_heads, ffn_units):
    inputs = layers.Input(shape=input_shape)
    print(f"Input shape: {inputs.shape}")

    # Spatial-Spectral CNN blocks using Conv3D
    x = ExpandLayer(axis=-1)(inputs)  # Add channel dimension #차원 확장
    print(f"After ExpandLayer: {x.shape}")
    x = layers.Conv3D(32, kernel_size=(3, 3, 1), activation=tf.nn.gelu, padding='same')(x) #3D 합성곱 연산
    print(f"After Conv3D(32): {x.shape}")
    x = layers.MaxPooling3D(pool_size=(1, 1, 2), padding='same')(x) #최대 풀링을 수행
    print(f"After MaxPooling3D(pool_size=(1, 1, 2)): {x.shape}")
    x = layers.Conv3D(64, kernel_size=(1, 1, 5), activation=tf.nn.gelu, padding='same')(x) #3D 합성곱 연산
    print(f"After Conv3D(64): {x.shape}")
    x = layers.MaxPooling3D(pool_size=(1, 1, 2), padding='same')(x)
    print(f"After MaxPooling3D(pool_size=(1, 1, 2)): {x.shape}")
    x = layers.Conv3D(128, kernel_size=(3, 3, 1), activation=tf.nn.gelu, padding='same')(x)
    print(f"After Conv3D(128): {x.shape}")
    x = layers.MaxPooling3D(pool_size=(1, 1, 2), padding='same')(x)
    print(f"After MaxPooling3D(pool_size=(1, 1, 2)): {x.shape}")
    x = layers.Dropout(0.5)(x)  #
    print(f"After Dropout(0.5): {x.shape}")

    # Flatten and prepare for transformer layers
    x = layers.Flatten()(x) #다차원 텐서를 1차원 벡터로 변환(transformer에 입력하기 위해서)
    print(f"After Flatten: {x.shape}")
    x = layers.Dense(embed_dim, activation=tf.nn.gelu)(x) #1차원 벡터(평탄화된 벡터)를 embed_dim 차원의 벡터로 변환 (relu->gelu로 활성화함수 변경)
    print(f"After Dense(embed_dim): {x.shape}")
    
    # Global Average Pooling instead of Flatten
#    x = layers.GlobalAveragePooling3D()(x)  # Dimension reduction
#    print(f"After GlobalAveragePooling3D: {x.shape}")  # e.g., (None, 128)
    
    # Dense layer with BatchNormalization and GELU for embedding
#    x = layers.Dense(embed_dim, activation=None)(x)
#    x = layers.BatchNormalization()(x)
#    x = layers.Activation(tf.nn.gelu)(x)
#    print(f"After Dense(embed_dim) with BatchNormalization and GELU: {x.shape}")

    
    # Emotion classification head before Transformer
    emotion_logits = layers.Dense(num_classes, activation='softmax', name="emotion_logits")(x)
    print(f"After Dense(num_classes): {emotion_logits.shape}")

    # Concatenate emotion predictions to transformer input
    emotion_features = layers.Concatenate(axis=-1)([x, emotion_logits])
    print(f"After Concatenate: {emotion_features.shape}")
    emotion_features = ExpandLayer(axis=1)(emotion_features)  # Expand for temporal dimension
    print(f"After ExpandLayer for temporal dimension: {emotion_features.shape}")

    # Transformer layers
    for i in range(3):  #2->3으로 변경
        emotion_features = TemporalEncodingLayer(embed_dim + num_classes, num_heads, ffn_units)(emotion_features)
        print(f"After TemporalEncodingLayer {i + 1}: {emotion_features.shape}")

    # Classification head
    x = layers.GlobalAveragePooling1D()(emotion_features)
    print(f"After GlobalAveragePooling1D: {x.shape}")
    x = layers.Dense(66, activation=tf.nn.gelu)(x)  #64->66으로 변경
    print(f"After Dense(66): {x.shape}")
    x = layers.Dropout(0.5)(x)
    print(f"After Dropout(0.5): {x.shape}")
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    print(f"Output shape: {outputs.shape}")

    model = models.Model(inputs, outputs)
    return model

# Load Data
def load_data(data_dir):
    X, y = [], []
    unique_files = set()
    print(f"Loading data from: {data_dir}")

    for file_name in os.listdir(data_dir):
        if file_name.endswith("_2D.npy"):
            file_path = os.path.join(data_dir, file_name)
            label = 1 if "positive" in file_name else 0

            try:
                data = np.load(file_path)
                reshaped_data = np.transpose(data, (0, 2, 1)) if data.ndim == 3 else data
                X.append(reshaped_data)
                y.append(label)
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

    if len(X) != len(y):
        raise ValueError(f"Data mismatch: X has {len(X)} samples, y has {len(y)} labels.")

    print(f"Loaded {len(X)} samples. Labels: {np.bincount(y)}")
    return np.array(X), np.array(y)

# K-Fold Training
def train_and_evaluate_kfold(data_dir, model_save_path, k=5):
    X, y = load_data(data_dir)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_metrics = []

    # Class weights for imbalance handling
    class_counts = np.bincount(y)
    class_weights = {i: max(class_counts) / c for i, c in enumerate(class_counts)}
    print(f"Class weights: {class_weights}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Starting Fold {fold + 1}/{k}...")

        X_fold_train, X_val = X[train_idx], X[val_idx]
        y_fold_train, y_val = y[train_idx], y[val_idx]
        print(f"Training samples: {len(y_fold_train)}, Validation samples: {len(y_val)}")

        # Initialize a fresh model for each fold
        model = build_model(
            input_shape=X_fold_train.shape[1:],
            num_classes=2,
            embed_dim=64,
            num_heads= 2,   #2(o) 3(x) 6(o)
            ffn_units=128
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

        # Learning Rate Scheduler
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.0005 * (0.9 ** epoch))

        # Early Stopping
        #early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

        # Train the model
        history = model.fit(
            X_fold_train, y_fold_train,
            epochs=100,
            batch_size=16,
            class_weight=class_weights,
            validation_data=(X_val, y_val),
            callbacks=[lr_scheduler], #, early_stopping],
            verbose=1
        )

        # Evaluate the model
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        fold_metrics.append({"fold": fold + 1, "val_loss": val_loss, "val_accuracy": val_accuracy})
        print(f"Fold {fold + 1}: Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Save the final model
    model.save(model_save_path, save_format="h5")
    print(f"Model saved at {model_save_path}")

    # Print average metrics
    avg_loss = np.mean([m["val_loss"] for m in fold_metrics])
    avg_accuracy = np.mean([m["val_accuracy"] for m in fold_metrics])
    print(f"Average Validation Loss: {avg_loss:.4f}, Average Validation Accuracy: {avg_accuracy:.4f}")

def test_model(data_dir, model_path):
    X, y = load_data(data_dir)

    # 모델 아키텍처를 재구성합니다.
    model = build_model(
        input_shape=X.shape[1:],  # 데이터 형태에 맞춤
        num_classes=2,
        embed_dim=64,
        num_heads=2,    #2(o) 3(x) 6(o)
        ffn_units=128
    )

    # 저장된 가중치를 불러옵니다.
    model.load_weights(model_path)

    # 입력 데이터의 차원이 모델에 맞는지 확인
    input_shape = model.input_shape[1:]
    if X.shape[1:] != input_shape:
        print(f"Adjusting test data from {X.shape[1:]} to {input_shape}...")
        X = tf.reshape(X, (-1,) + input_shape)

    # 예측 수행
    preds = model.predict(X)
    y_pred = np.argmax(preds, axis=1)

    # 분류 리포트 생성 및 출력
    report = classification_report(y, y_pred, target_names=["Negative", "Positive"])
    print("Test Results:")
    print(report)

    # 분류 보고서 저장
    report_path = "4s_test_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")


# Example usage
data_directory = "/home/bcml1/2025_EMOTION/DEAP_EEG/overlap4s_seg_conv_ch_BPF+DE/train"  # Data path
model_save_file = "4s_conv3d_model.h5"

# Train and Evaluate
train_and_evaluate_kfold(data_directory, model_save_file, k=5)

# Test
test_directory = "/home/bcml1/2025_EMOTION/DEAP_EEG/overlap4s_seg_conv_ch_BPF+DE/test"  # Test data path
test_model(test_directory, model_save_file)