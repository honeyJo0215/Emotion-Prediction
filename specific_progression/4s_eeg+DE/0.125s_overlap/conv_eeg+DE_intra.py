#attention 논문의 내용을 적용하기.
# Define Multi-Head Self Attention
#intra subject cross validation

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# GPU 설정
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Failed to set memory growth: {e}")
else:
    print("No GPU devices found. Running on CPU.")


# **Multi-Head Self Attention**
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

# **Feed Forward Network (FFN)**
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

# **Temporal Encoding Layer**
class TemporalEncodingLayer(layers.Layer):
    def __init__(self, embed_dim, num_heads, ffn_units, **kwargs):
        super(TemporalEncodingLayer, self).__init__(**kwargs)
        self.mhsa = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_units)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        attn_output = self.mhsa(inputs)
        out1 = self.layernorm1(inputs + attn_output)  # Residual Connection
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + ffn_output)  # Residual Connection
        return out2


# Define the model
class ExpandLayer(layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super(ExpandLayer, self).__init__(**kwargs)
        self.axis = axis
        
    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

# **Model Definition**
def build_model(input_shape, num_classes, embed_dim, num_heads, ffn_units):
    inputs = layers.Input(shape=input_shape)

    # Spatial-Spectral CNN blocks using Conv3D
    x = ExpandLayer(axis=-1)(inputs)  # Add channel dimension #차원 확장

    # CNN 적용
    #지금은 raw + DE를 하나의 npy로 만든 것을 input으로 해서 intra CNN 모델을 구현하지만,
    #다음 코드는 DE를 합성곱으로 하는 코드를 구현하자.
    x = layers.Conv3D(32, kernel_size=(3, 3, 1), activation=tf.nn.gelu, padding='same')(x)  #3D 합성곱 연산
    x = layers.MaxPooling3D(pool_size=(1, 1, 2), padding='same')(x) #최대 풀링을 수행
    x = layers.Conv3D(64, kernel_size=(1, 1, 5), activation=tf.nn.gelu, padding='same')(x)  #3D 합성곱 연산
    x = layers.MaxPooling3D(pool_size=(1, 1, 2), padding='same')(x)
    x = layers.Conv3D(128, kernel_size=(3, 3, 1), activation=tf.nn.gelu, padding='same')(x) #3D 합성곱 연산
    x = layers.MaxPooling3D(pool_size=(1, 1, 2), padding='same')(x)
    x = layers.Dropout(0.5)(x)

    # Flatten and Prepare for Transformer
    x = layers.Flatten()(x) #다차원 텐서를 1차원 벡터로 변환(transformer에 입력하기 위해서)
    x = layers.Dense(embed_dim, activation=tf.nn.gelu)(x)   #1차원 벡터(평탄화된 벡터)를 embed_dim 차원의 벡터로 변환 (relu->gelu로 활성화함수 변경)

    
    # Emotion classification head before Transformer
    emotion_logits = layers.Dense(num_classes, activation='softmax', name="emotion_logits")(x)
    # Concatenate emotion predictions to transformer input
    emotion_features = layers.Concatenate(axis=-1)([x, emotion_logits])
    emotion_features = ExpandLayer(axis=1)(emotion_features)  # Expand for temporal dimension
    
    # Transformer Layers
    for _ in range(3):
        emotion_features = TemporalEncodingLayer(embed_dim + num_classes, num_heads, ffn_units)(emotion_features)
        # x = TemporalEncodingLayer(embed_dim, num_heads, ffn_units)(x)

    # Classification head
    x = layers.GlobalAveragePooling1D()(emotion_features)
    # x = layers.Dense(66, activation=tf.nn.gelu)(x)  #!!!다음번 코드에서는 구현 안함!!!
    x = layers.Dropout(0.5)(x)  
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

# **Load Data with Subject IDs**
def load_data(data_dir):
    X, y, subjects = [], [], []
    # unique_files = set()
    print(f"Loading data from: {data_dir}")

    for file_name in os.listdir(data_dir):
        if file_name.endswith("_FB.npy"):
            file_path = os.path.join(data_dir, file_name)

            # 피험자 ID 추출 (예: "s11_negative_segment_0_sample_0_FB.npy" → "s11")
            subject_id = file_name.split("_")[0]
            label = 1 if "positive" in file_name else 0

            try:
                data = np.load(file_path)
                reshaped_data = np.transpose(data, (0, 2, 1)) if data.ndim == 3 else data
                X.append(reshaped_data)
                y.append(label)
                subjects.append(subject_id)
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

    return np.array(X), np.array(y), np.array(subjects)

# **K-Fold Training**   
# **Intra-Subject Cross Validation**
def train_and_evaluate_kfold(X, y, subjects, model_save_path, k=10):  #k-fold 수를 10으로 설정!!!
    unique_subjects = np.unique(subjects)
    all_fold_metrics = []  # 전체 피험자의 메트릭을 저장할 리스트 추가
    # X, y = load_data(data_dir)
    
    for subject in unique_subjects:
        print(f"Training for Subject: {subject}")

        # 해당 피험자의 데이터만 추출
        subject_idx = (subjects == subject)
        X_subject, y_subject = X[subject_idx], y[subject_idx]

        # 8:2로 Train/Test 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X_subject, y_subject, test_size=0.2, random_state=42, stratify=y_subject
        )

        if len(X_subject) < k:
            print(f"Not enough data for Subject {subject}. Skipping...")
            continue
        
        k = min(k, len(X_train))  # K 값이 샘플 수보다 크지 않도록 조정
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        fold_metrics = [] #추가

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            print(f"Starting Fold {fold + 1}/{k}...")

            X_fold_train, X_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_val = y_train[train_idx], y_train[val_idx]
            print(f"Training samples: {len(y_fold_train)}, Validation samples: {len(y_val)}")

            # 추가: Class weights for imbalance handling
            class_counts = np.bincount(y_subject)
            class_weights = {i: max(class_counts) / c for i, c in enumerate(class_counts)}
            print(f"Class weights: {class_weights}")
            
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
            
            # model.fit(X_fold_train, y_fold_train, epochs=50, batch_size=16, validation_data=(X_val, y_val), verbose=1)

            # Train the model
            history = model.fit(
                X_fold_train, y_fold_train,
                epochs=10,
                batch_size=4,
                class_weight=class_weights,
                validation_data=(X_val, y_val),
                callbacks=[lr_scheduler],
                verbose=1
            )
            
            # Evaluate the model
            val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
            fold_metrics.append({"fold": fold + 1, "val_loss": val_loss, "val_accuracy": val_accuracy})
            print(f"Fold {fold + 1}: Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
            
        # 평균 정확도와 손실 계산 (수정된 부분)
        avg_loss = np.mean([m["val_loss"] for m in fold_metrics])
        avg_accuracy = np.mean([m["val_accuracy"] for m in fold_metrics])
        
        print(f"✅ Subject {subject} Average Accuracy: {avg_accuracy:.4f}")
        
        # 전체 메트릭에 추가
        all_fold_metrics.extend(fold_metrics)
            
        # 각 피험자의 모델 저장
        model_save_path_subject = f"{model_save_path}_subject_{subject}.h5"
        model.save(model_save_path_subject)  # save_format 제거
        
        # model.save(model_save_path_subject, save_format="h5")
        
        print(f"Model saved at {model_save_path_subject}")

        #테스트
        test_model(X_test, y_test, model_save_path_subject)
        
    print("✅ Intra-Subject Training Completed.")
    # 전체 평균 메트릭 계산
    if all_fold_metrics:
        overall_avg_loss = np.mean([m["val_loss"] for m in all_fold_metrics])
        overall_avg_accuracy = np.mean([m["val_accuracy"] for m in all_fold_metrics])
        print(f"Overall Average Validation Loss: {overall_avg_loss:.4f}, Overall Average Validation Accuracy: {overall_avg_accuracy:.4f}")
    else:
        print("No metrics to average.")
    
def test_model(X, y, model_path):
    if len(X) == 0:
        print(" No test data available. Skipping evaluation.")
        return
    
    # X, y = load_data(data_dir)
    
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
    
    # 모델 컴파일
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    # 입력 데이터의 차원이 모델에 맞는지 확인
    input_shape = model.input_shape[1:]
    if X.shape[1:] != input_shape:
        print(f"Adjusting test data from {X.shape[1:]} to {input_shape}...")
        X = tf.reshape(X, (-1,) + input_shape)
        
    test_loss, test_accuracy = model.evaluate(X, y, verbose=1)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # 예측 수행
    preds = model.predict(X)
    y_pred = np.argmax(preds, axis=1)
    
    # 분류 리포트 생성 및 출력
    report = classification_report(y, y_pred, target_names=["Negative", "Positive"])
    print("Test Results:")
    print(report)

    # 분류 보고서 저장
    # report_path = "4s_test_classification_intra_report.txt"
    
    report_path = f"{model_path.replace('.h5', '')}_classification_report.txt"
    
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")
    
# **Execute Training**
data_directory = "/home/bcml1/2025_EMOTION/DEAP_EEG/overlap4s_0.125_seg_conv_ch_BPF/segment_sample_2D/no_split"
X, y, subjects = load_data(data_directory)

model_save_file = "4s_0.125overlap_intra_model"
train_and_evaluate_kfold(X, y, subjects, model_save_file, k=10)
# test_model(X, y, model_save_file)
