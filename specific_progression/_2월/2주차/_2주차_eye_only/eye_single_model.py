import os
#cpu 사용
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # GPU 메모리 자동 증가 방지
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # TensorFlow 로그 최소화
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dense, Flatten, LayerNormalization, Dropout, TimeDistributed, GlobalAveragePooling1D, Lambda
from tensorflow.keras.models import Model

# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 메모리 제한 설정
# def limit_gpu_memory(memory_limit_mib=10000):
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     if gpus:
#         try:
#             tf.config.experimental.set_virtual_device_configuration(
#                 gpus[0],
#                 [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit_mib)]
#             )
#             print(f"GPU memory limited to {memory_limit_mib} MiB.")
#         except RuntimeError as e:
#             print(e)
#     else:
#         print("No GPU available, using CPU.")

# limit_gpu_memory(10000)

# ---------------------------
# 경로 설정
EYE_CROP_PATH = "/home/bcml1/2025_EMOTION/eye_crop"  # 예: /home/bcml1/2025_EMOTION/eye_crop/s01, s02, ..., s22
LABELS_PATH   = "/home/bcml1/2025_EMOTION/DEAP_three_labels"  # 예: s01_three_labels.npy, s02_three_labels.npy, ...

SAVE_PATH = "/home/bcml1/sigenv/_2주차_eye_only/result1"
os.makedirs(SAVE_PATH, exist_ok=True)

# ---------------------------
# 감정 라벨 매핑 (이미 label 파일에 저장되어 있으므로 별도 매핑은 필요없음)
# 다만 label은 negative:0, positive:1, neutral:2

# ---------------------------
# 공통 전처리 함수 (입력 데이터의 NaN, Inf 처리 및 min-max scaling)
def preprocess_data(data):
    # 입력 데이터를 float32로 변환
    data = tf.cast(data, tf.float32)
    data = tf.where(tf.math.is_nan(data), tf.zeros_like(data), data)
    data = tf.where(tf.math.is_inf(data), tf.zeros_like(data), data)
    min_val = tf.reduce_min(data)
    max_val = tf.reduce_max(data)
    range_val = max_val - min_val + 1e-8  # 분모가 0이 되는 것을 방지
    data = (data - min_val) / range_val
    return data


# ---------------------------
# 가중치 평균화 함수 (모델 가중치를 평균 내어 최종 모델에 적용)
def average_model_weights(model, model_paths):
    weights = []
    for path in model_paths:
        try:
            temp_model = EyeCropTransformer(
                window_length=model.window_length,
                n_layers=model.n_layers,
                n_heads=model.n_heads,
                d_ff=model.d_ff,
                p_drop=model.p_drop,
                d_model=model.d_model
            )
            # 모델 빌드: 입력 shape은 (window_length, 2,32,64,3)
            temp_model.build((None, model.window_length, 2, 32, 64, 3))
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

# ---------------------------
# Transformer Encoder Layer (기존과 동일)
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
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

    def call(self, inputs, training=False):
        # 입력이 (batch, seq_len, d_model)라면 그대로 진행
        attn_output = self.mha(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "dropout_rate": self.dropout_rate,
        }

# ---------------------------
# EyeCropTransformer 모델
class EyeCropTransformer(tf.keras.Model):
    def __init__(self, window_length=500, n_layers=2, n_heads=4, d_ff=512, p_drop=0.1, d_model=128):
        """
        Args:
            window_length: 슬라이딩 윈도우 길이 (500 프레임)
            n_layers: Transformer encoder layer 수
            n_heads: Multi-head attention 헤드 수
            d_ff: Feed-forward 네트워크 차원
            p_drop: 드롭아웃 비율
            d_model: Transformer 임베딩 차원 (여기서는 128)
        """
        super(EyeCropTransformer, self).__init__()
        self.window_length = window_length
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.p_drop = p_drop
        self.d_model = d_model

        # Feature Extractor: 각 프레임의 eye-crop 데이터가 원래 (2,32,64,3) 형태이나, 
        # call()에서 (32,64,6)으로 변환한 후 CNN 계층에 넣습니다.
        self.feature_extractor = tf.keras.Sequential([
            TimeDistributed(Conv2D(16, kernel_size=(3,3), activation="relu", padding="same")),
            TimeDistributed(Conv2D(32, kernel_size=(3,3), activation="relu", padding="same")),
            TimeDistributed(AveragePooling2D(pool_size=(2,2))),
            TimeDistributed(Flatten()),
            TimeDistributed(Dense(128, activation="relu"))
        ])
        # 각 프레임의 feature를 d_model 차원으로 투사
        self.dense_projection = TimeDistributed(Dense(d_model, activation="relu"))
        # Transformer Encoder Layers
        self.encoder_layers = [
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout_rate=p_drop)
            for _ in range(n_layers)
        ]
        # Global pooling over time axis
        self.global_pool = GlobalAveragePooling1D()
        # 최종 분류 (3개 클래스)
        self.output_dense = Dense(3, activation="softmax")

    def build(self, input_shape):
        # 입력 shape: (batch, window_length, 2, 32, 64, 3)
        super(EyeCropTransformer, self).build(input_shape)

    def call(self, inputs, training=False):
        # inputs: (batch, window_length, 2, 32, 64, 3)
        #x = preprocess_data(inputs)  # preprocess_data에서 float32로 캐스팅 등 전처리 수행
        x=inputs
        # --- 채널 합치기: (2,32,64,3) → (32,64,6) ---
        # 1. 입력 x의 shape는 (batch, window_length, 2, 32, 64, 3)
        shape = tf.shape(x)
        batch_size = shape[0]
        time_steps = shape[1]
        # 2. 배치와 시간 축을 merge하여 각 프레임 단위로 만듦
        x_reshaped = tf.reshape(x, (-1, 2, 32, 64, 3))  # (batch*time, 2, 32, 64, 3)
        # 3. 각 프레임의 텐서를 (32,64,6)으로 변환하기 위해 먼저 transpose:
        #    (2,32,64,3) → (32,64,2,3)
        x_transposed = tf.transpose(x_reshaped, perm=[0, 2, 3, 1, 4])  # (batch*time, 32, 64, 2, 3)
        # 4. 마지막 두 차원을 합쳐 (32,64,6)으로 reshape
        x_combined = tf.reshape(x_transposed, (-1, 32, 64, 2*3))  # (batch*time, 32, 64, 6)
        # 5. 원래 배치와 시간 축으로 복원: (batch, window_length, 32, 64, 6)
        x = tf.reshape(x_combined, (batch_size, time_steps, 32, 64, 6))
        # --- 여기까지 각 프레임의 채널을 합쳐 (32,64,6) 모양으로 만들었습니다. ---

        # 이제 CNN 계층에 넣어 feature를 추출합니다.
        x = self.feature_extractor(x)       # → (batch, window_length, 128)
        x = self.dense_projection(x)          # → (batch, window_length, d_model)
        for layer in self.encoder_layers:
            x = layer(x, training=training)   # Transformer 인코더 적용
        x = self.global_pool(x)               # 시퀀스 차원 축소 → (batch, d_model)
        return self.output_dense(x)           # 최종 분류 결과 (batch, 3)

    def get_config(self):
        return {
            "window_length": self.window_length,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "p_drop": self.p_drop,
            "d_model": self.d_model,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# ---------------------------
# 디버깅용 함수 (모델 출력 확인)
def debug_model_outputs(model, dataset):
    for data, labels in dataset.take(1):
        outputs = model(data, training=True)
        print(f"Model output NaN: {np.isnan(outputs.numpy()).any()}, Inf: {np.isinf(outputs.numpy()).any()}")
        print(f"Model output shape: {outputs.shape}, Labels shape: {labels.shape}")

def debug_loss(model, dataset):
    for data, labels in dataset.take(1):
        outputs = model(data, training=True)
        print(f"Model outputs: {outputs.numpy()}")
        print(f"Labels: {labels.numpy()}")

def load_subject_eye_crop_data(subject, window_length=500, stride=50):
    """avi_frame0000
    - subject: 예, "s01", "s02", ... "s22"
    - 각 subject 폴더 내에 40 trial의 npy 파일들이 있으며,
      각 trial는 파일명이 예: s01_trial01..npy, s01_trial01.avi_frame0001.npy, ... 로 구성됨.
    - 각 npy 파일은 어떤 형식이든 reshape하여 (2,32,64,3)로 만듦.
    - 각 trial는 총 3000 프레임(예상)이며, 슬라이딩 윈도우 (window_length=500, stride=50)로 sample을 생성.
    - 해당 subject의 label 파일은 LABELS_PATH/sXX_three_labels.npy (배열 shape: (40,))에 저장됨.
    """
    subject_folder = os.path.join(EYE_CROP_PATH, subject)
    # label 파일 경로 (예: /home/bcml1/2025_EMOTION/DEAP_three_labels/s01_three_labels.npy)
    label_file = os.path.join(LABELS_PATH, f"{subject}_three_labels.npy")
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"Label file not found: {label_file}")
    trial_labels = np.load(label_file)  # shape: (40,)

    all_windows = []
    all_labels = []
    # trial 번호: 1 ~ 40
    for trial in range(1, 41):
        trial_str = f"trial{trial:02d}"
        # 파일명 예: s01_trial01.avi_frame0000.npy
        trial_files = [f for f in os.listdir(subject_folder) if f.startswith(f"{subject}_{trial_str}.avi_frame") and f.endswith(".npy")]
        if len(trial_files) == 0:
            print(f"Trial {trial_str} in {subject} has no files. Skipping...")
            continue
        # 정렬: 파일명에서 "frame" 이후 숫자를 추출하여 정렬
        def get_frame_index(filename):
            try:
                part = filename.split("frame")[-1]
                # part는 "0000.npy"와 같이 되어 있으므로 제거
                return int(part.split(".")[0])
            except:
                return 0
        trial_files = sorted(trial_files, key=get_frame_index)
        
        trial_frames = []
        for file_name in trial_files:
            file_path = os.path.join(subject_folder, file_name)
            try:
                frame = np.load(file_path)
                # 기본: 한 채널 당 원소 수
                base = 32 * 64 * 3  # 6144
                if frame.size % base != 0:
                    print(f"Unexpected frame size {frame.size} in {file_path}. Skipping this frame.")
                    continue
                channels = frame.size // base
                if channels == 1:
                    # 1채널 데이터인 경우 복사하여 2채널로 만듦
                    frame = np.reshape(frame, (1, 32, 64, 3))
                    frame = np.concatenate([frame, frame], axis=0)
                elif channels >= 2:
                    frame = np.reshape(frame, (channels, 32, 64, 3))
                    frame = frame[:2, :, :, :]
                else:
                    print(f"Not enough channels in {file_path} (found {channels}). Skipping this frame.")
                    continue
                trial_frames.append(frame)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        # trial_frames → array shape: (num_frames, 2,32,64,3)
        trial_frames = np.array(trial_frames)
        num_frames = trial_frames.shape[0]
        if num_frames < window_length:
            print(f"Trial {trial_str} in {subject} has less than {window_length} frames. Skipping...")
            continue
        # 슬라이딩 윈도우 생성: 시작 인덱스 0부터 (num_frames - window_length)까지 stride 간격으로
        for start in range(0, num_frames - window_length + 1, stride):
            window = trial_frames[start:start + window_length]  # shape: (500, 2,32,64,3)
            all_windows.append(window)
            # trial의 라벨은 trial_labels[trial-1]
            all_labels.append(trial_labels[trial - 1])
        print(f"trial {trial} loaded")
    if len(all_windows) == 0:
        raise ValueError(f"No valid windows found for {subject}.")
    data_array = np.array(all_windows)
    labels_array = np.array(all_labels)
    print(f"{subject}: Generated {data_array.shape[0]} samples from eye-crop data.")
    return data_array, labels_array
# ---------------------------
# 학습 및 평가 (Stratified K-Fold)
def train_eye_crop_with_stratified_kfold():
    # 서브젝트 s01 ~ s22
    subjects = [f"s{str(i).zfill(2)}" for i in range(1, 23)]
    final_model_path = os.path.join(SAVE_PATH, "final_model")
    os.makedirs(final_model_path, exist_ok=True)

    for subject in subjects:
        print(f"\n========== Training subject: {subject} ==========")
        subject_save_path = os.path.join(SAVE_PATH, subject)
        os.makedirs(subject_save_path, exist_ok=True)

        try:
            data, labels = load_subject_eye_crop_data(subject, window_length=500, stride=50)
        except Exception as e:
            print(f"Error loading data for {subject}: {e}")
            continue

        # NaN/Inf 처리 (이미 preprocess_data가 모델 내부에서 적용되지만 추가 전처리 가능)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        labels = labels.astype(np.int32)

        # train/test 분리
        train_data, test_data, train_labels, test_labels = train_test_split(
            data, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Stratified K-Fold (10-fold)
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        fold_model_paths = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(train_data, train_labels)):
            print(f"\n--- Fold {fold+1} ---")
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (train_data[train_idx], train_labels[train_idx])
            ).shuffle(1000).batch(8)
            val_dataset = tf.data.Dataset.from_tensor_slices(
                (train_data[val_idx], train_labels[val_idx])
            ).batch(8)

            model = EyeCropTransformer(window_length=500, n_layers=2, n_heads=4, d_ff=512, p_drop=0.1, d_model=128)
            # 모델 빌드: 입력 shape (batch, 500, 2,32,64,3)
            model.build((None, 500, 2, 32, 64, 3))
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )

            debug_model_outputs(model, train_dataset)
            debug_loss(model, train_dataset)

            model.fit(train_dataset, epochs=50, validation_data=val_dataset)

            fold_model_path = os.path.join(subject_save_path, f"{subject}_fold_{fold+1}.weights.h5")
            model.save_weights(fold_model_path)
            fold_model_paths.append(fold_model_path)

        # 가중치 평균화 후 최종 모델 생성
        final_model = EyeCropTransformer(window_length=500, n_layers=2, n_heads=4, d_ff=512, p_drop=0.1, d_model=128)
        final_model.build((None, 500, 2, 32, 64, 3))
        final_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        final_model = average_model_weights(final_model, fold_model_paths)

        # 전체 train 데이터로 추가 학습
        full_train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).shuffle(1000).batch(8)
        final_model.fit(full_train_dataset, epochs=50)

        final_model.save_weights(os.path.join(subject_save_path, "final_model.weights.h5"))
        print(f"Final model weights saved for subject {subject}.")

        # 테스트 데이터 평가
        test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(32)
        predictions = final_model.predict(test_dataset)
        predicted_labels = tf.argmax(predictions, axis=-1).numpy()

        test_report = classification_report(
            test_labels, predicted_labels,
            target_names=["negative", "positive", "neutral"],
            labels=[0, 1, 2], zero_division=0
        )
        print(f"\nTest Report for {subject}")
        print(test_report)

        with open(os.path.join(final_model_path, f"{subject}_test_report.txt"), "w") as f:
            f.write(test_report)

# ---------------------------
# 실행
if __name__ == "__main__":
    train_eye_crop_with_stratified_kfold()
