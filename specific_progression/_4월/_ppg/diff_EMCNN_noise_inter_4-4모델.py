import os
import glob
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import logging
import seaborn as sns

# -------------------------------------------------------------------------
# GPU 메모리 제한
# -------------------------------------------------------------------------
def limit_gpu_memory(memory_limit_mib=8000):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit_mib)]
            )
            logging.info(f"GPU memory limited to {memory_limit_mib} MiB.")
        except RuntimeError as e:
            logging.error(e)
    else:
        logging.info("No GPU available, using CPU.")

limit_gpu_memory(5000)

# =============================================================================
# DiffuSSMLayer: Dense와 LayerNormalization을 이용한 diffusion block
# =============================================================================
class DiffuSSMLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_dim=384, output_units=None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_units = output_units
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.dense2 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.norm2 = tf.keras.layers.LayerNormalization()
        final_units = output_units if output_units is not None else hidden_dim
        self.out_dense = tf.keras.layers.Dense(final_units, activation=None)
        self.norm_out = tf.keras.layers.LayerNormalization()

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
# =============================================================================
def add_diffusion_noise_numpy(x, stddev=0.05):
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=stddev, dtype=x.dtype)
    return x + noise

# -----------------------------
# 1. Feature Extractor 정의 (각 브랜치에 사용)
# -----------------------------
def build_feature_extractor(input_length=None):
    """
    입력 shape: (L, 3) – L: 시퀀스 길이, 3채널 입력
    출력: 128 차원 feature 벡터 (GlobalAveragePooling1D 사용)
    """
    inputs = tf.keras.Input(shape=(input_length, 3))
    
    noise_inputs = layers.Lambda(lambda x: add_diffusion_noise_numpy(x, stddev=0.07))(inputs)
    # x = inputs + noise_inputs
    pre_processed = layers.Add()([noise_inputs, inputs])
    
    # x = tf.keras.layers.Conv1D(filters=8, kernel_size=7, strides=1, padding='same')(pre_processed)
    # x = tf.keras.layers.ReLU(max_value=6)(x)
    x = tf.keras.layers.SeparableConv1D(filters=16, kernel_size=7, strides=4, padding='same')(noise_inputs)
    x = tf.keras.layers.ReLU(max_value=6)(x)
    # x = tf.keras.layers.SeparableConv1D(filters=32, kernel_size=7, strides=2, padding='same')(x)
    # x = tf.keras.layers.ReLU(max_value=6)(x)
    # x = tf.keras.layers.SeparableConv1D(filters=64, kernel_size=7, strides=4, padding='same')(x)
    # x = tf.keras.layers.ReLU(max_value=6)(x)
    x = tf.keras.layers.SeparableConv1D(filters=128, kernel_size=7, strides=2, padding='same')(x)
    x = tf.keras.layers.ReLU(max_value=6)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model

# -----------------------------
# 2. EMCNN 모델 정의 (tf.keras.Model subclassing, diffusion 추가)
# -----------------------------
class EMCNN(tf.keras.Model):
    def __init__(self, num_classes=4, smoothing_kernel_size=2, downsample_factor=2, input_length=None):
        """
        num_classes: 분류 클래스 수  
        smoothing_kernel_size: 이동평균 필터의 커널 크기  
        downsample_factor: 다운샘플링 비율  
        input_length: 원본 시퀀스 길이
        """
        super(EMCNN, self).__init__()
        self.num_classes = num_classes
        self.smoothing_kernel_size = smoothing_kernel_size
        self.downsample_factor = downsample_factor

        # 각 브랜치별 Feature Extractor (가중치는 공유하지 않음)
        self.feature_identity = build_feature_extractor(input_length=input_length)
        self.feature_smoothing = build_feature_extractor(input_length=input_length)
        self.feature_downsampling = build_feature_extractor(input_length=None)

        # Smoothing: 3채널 입력에 대해 각 채널별 이동평균 (groups=3 사용)
        self.smoothing_conv = tf.keras.layers.Conv1D(filters=3, kernel_size=smoothing_kernel_size,
                                                     padding='same', use_bias=False, groups=3)
        smoothing_weight = np.ones((smoothing_kernel_size, 1, 3), dtype=np.float32) / smoothing_kernel_size
        self.smoothing_conv.build((None, input_length, 3))
        self.smoothing_conv.set_weights([smoothing_weight])
        self.smoothing_conv.trainable = False

        # Diffusion block: concatenated feature의 차원은 128*3 = 384
        self.diffusion_layer = DiffuSSMLayer(hidden_dim=384, output_units=384)

        # 분류를 위한 Fully Connected 레이어들
        self.fc1 = tf.keras.Sequential([
            tf.keras.layers.Dense(128),
            tf.keras.layers.ReLU(max_value=6)
        ])
        # self.dropout1 = tf.keras.layers.Dropout(0.3)
        self.fc2 = tf.keras.Sequential([
            tf.keras.layers.Dense(64),
            tf.keras.layers.ReLU(max_value=6)
        ])
        # self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.fc3 = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training=False):
        """
        inputs: shape = (batch, L, 3)
        """
        # Transformation Stage:
        branch_identity = inputs
        branch_smooth = self.smoothing_conv(inputs)
        branch_down = inputs[:, ::self.downsample_factor, :]

        # Feature Extraction Stage:
        feat_identity = self.feature_identity(branch_identity)
        feat_smooth = self.feature_smoothing(branch_smooth)
        feat_down = self.feature_downsampling(branch_down)

        # 특징 벡터 연결 (예상 feature dimension: 128*3 = 384)
        features = tf.concat([feat_identity, feat_smooth, feat_down], axis=1)
        # Diffusion 적용 (residual connection)
        diffused_features = features + self.diffusion_layer(features)

        # Classification Stage:
        # x = self.fc1(diffused_features)
        # x = self.fc2(x)
        # logits = self.fc3(x)
        x = self.fc1(diffused_features)
        # x = self.dropout1(x, training=training)
        x = self.fc2(x)
        # x = self.dropout2(x, training=training)
        logits = self.fc3(x)
        return logits

# -----------------------------
# 3. 데이터 로딩 함수: 1s와 10s 버전 (원본 5채널 데이터에서 원하는 3채널 선택 및 노이즈 추가)
# -----------------------------
def load_data_1s(data_dir, ch_choice_pair1=1, ch_choice_pair2=3, noise_std=0.05):
    """
    DEAP_PPG_1s 폴더 파일 로드  
    파일 shape: (60, 5, 128)  
    선택: 첫 번째 채널은 항상 사용 (원본), 
           두번째와 세번째 채널 중 ch_choice_pair1 (예: 1 또는 2),
           네번째와 다섯번째 채널 중 ch_choice_pair2 (예: 3 또는 4)
    최종 shape: (60, 128, 3)
    그리고 로드된 데이터에 가우시안 노이즈를 추가합니다.
    """
    subject_data = {}
    file_paths = glob.glob(os.path.join(data_dir, "*.npy"))
    logging.info(f"총 {len(file_paths)}개의 파일을 찾았습니다. (1s)")
    for file_path in file_paths:
        base = os.path.basename(file_path)
        try:
            subject_id = base.split('_')[0]
            label = int(base.split('_')[-1].split('.')[0])
        except Exception as e:
            logging.error(f"라벨/서브젝트 추출 오류: {file_path}, {e}")
            continue
        data = np.load(file_path)  # (60, 5, 128)
        # 입력 데이터에 가우시안 노이즈 추가
        # data = add_diffusion_noise_numpy(data, stddev=noise_std)
        # 채널 선택: 항상 0번 채널, 그리고 ch_choice_pair1, ch_choice_pair2 선택
        data = data[:, [0, ch_choice_pair1, ch_choice_pair2], :]  # (60, 3, 128)
        data = np.transpose(data, (0, 2, 1))  # (60, 128, 3)
        if subject_id not in subject_data:
            subject_data[subject_id] = {'X': [], 'y': []}
        subject_data[subject_id]['X'].append(data)
        subject_data[subject_id]['y'].append(np.full((data.shape[0],), label, dtype=np.int32))
    for subject in subject_data:
        subject_data[subject]['X'] = np.concatenate(subject_data[subject]['X'], axis=0)
        subject_data[subject]['y'] = np.concatenate(subject_data[subject]['y'], axis=0)
        logging.info(f"{subject} - X shape: {subject_data[subject]['X'].shape}, y shape: {subject_data[subject]['y'].shape}")
    return subject_data

def load_data_10s(data_dir, ch_choice_pair1=1, ch_choice_pair2=3, noise_std=0.05):
    """
    DEAP_PPG_10s 폴더 파일 로드  
    파일 shape: (6, 5, 1280)  
    선택: 첫 번째 채널은 항상 사용 (원본), 
           두번째와 세번째 채널 중 ch_choice_pair1 (예: 1 또는 2),
           네번째와 다섯번째 채널 중 ch_choice_pair2 (예: 3 또는 4)
    최종 shape: (6, 1280, 3)
    그리고 로드된 데이터에 가우시안 노이즈를 추가합니다.
    """
    subject_data = {}
    file_paths = glob.glob(os.path.join(data_dir, "*.npy"))
    logging.info(f"총 {len(file_paths)}개의 파일을 찾았습니다. (1s)")
    for file_path in file_paths:
        base = os.path.basename(file_path)
        try:
            subject_id = base.split('_')[0]
            label = int(base.split('_')[-1].split('.')[0])
        except Exception as e:
            logging.error(f"라벨/서브젝트 추출 오류: {file_path}, {e}")
            continue
        data = np.load(file_path)  # (6, 5, 1280)
        # 입력 데이터에 가우시안 노이즈 추가
        # data = add_diffusion_noise_numpy(data, stddev=noise_std)
        # 채널 선택: 항상 0번 채널, 그리고 ch_choice_pair1, ch_choice_pair2 선택
        data = data[:, [0, ch_choice_pair1, ch_choice_pair2], :]  # (6, 3, 1280)
        data = np.transpose(data, (0, 2, 1))  # (6, 1280, 3)
        if subject_id not in subject_data:
            subject_data[subject_id] = {'X': [], 'y': []}
        subject_data[subject_id]['X'].append(data)
        subject_data[subject_id]['y'].append(np.full((data.shape[0],), label, dtype=np.int32))
    for subject in subject_data:
        subject_data[subject]['X'] = np.concatenate(subject_data[subject]['X'], axis=0)
        subject_data[subject]['y'] = np.concatenate(subject_data[subject]['y'], axis=0)
        logging.info(f"{subject} - X shape: {subject_data[subject]['X'].shape}, y shape: {subject_data[subject]['y'].shape}")
    return subject_data

def load_data_10s_1soverlap(data_dir, ch_choice_pair1=1, ch_choice_pair2=3, noise_std=0.05):
    """
    DEAP_PPG_10s_1soverlap_4label 폴더 파일 로드  
    파일 shape: (51, 5, 1280)  
    1. np.load 후 transpose: (51, 1280, 5)  
    2. 선택: 채널 0, 그리고 ch_choice_pair1, ch_choice_pair2 선택  
    최종 shape: (51, 1280, 3)
    그리고 로드된 데이터에 가우시안 노이즈를 추가합니다.
    """
    subject_data = {}
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    logging.info(f"총 {len(file_list)}개의 파일을 찾았습니다. (10s)")
    for f in file_list:
        file_path = os.path.join(data_dir, f)
        try:
            subject_id = f.split('_')[0]
            label = int(f.split('_')[-1].split('.')[0])
        except Exception as e:
            logging.error(f"라벨/서브젝트 추출 오류: {file_path}, {e}")
            continue
        data = np.load(file_path)            # shape (60, 5, 128) → NumPy array
        data = data[:, [0, ch_choice_pair1, ch_choice_pair2], :]  # advanced indexing works
        data = np.transpose(data, (0, 2, 1))                      # still NumPy
        # data = add_diffusion_noise_numpy(data, stddev=noise_std)   # add noise as NumPy
        if subject_id not in subject_data:
            subject_data[subject_id] = {'X': [], 'y': []}
        subject_data[subject_id]['X'].append(data)
        subject_data[subject_id]['y'].append(np.full((data.shape[0],), label, dtype=np.int32))
    for subject in subject_data:
        subject_data[subject]['X'] = np.concatenate(subject_data[subject]['X'], axis=0)
        subject_data[subject]['y'] = np.concatenate(subject_data[subject]['y'], axis=0)
        logging.info(f"{subject} - X shape: {subject_data[subject]['X'].shape}, y shape: {subject_data[subject]['y'].shape}")
    return subject_data

# -----------------------------
# 4. "Inter-subject" Cross Validation
# -----------------------------
def train_model_inter_subject(
    subject_data,
    num_classes=4,
    epochs=300,
    batch_size=128,
    result_dir_base="results_inter_subject",
    start_subject="s01"
):
    """
    - For each subject S, use that subject's data as the test set,
      and the combined data from all other subjects as the train set.
    - Internally do a train/val split on the combined set before training.
    - Evaluate on S.
    """
    os.makedirs(result_dir_base, exist_ok=True)
    all_subjects = sorted(subject_data.keys(), key=lambda x: int(x[1:]))  # e.g. ['s01', 's02', ...]

    # start_subject부터 시작하도록 필터링
    start_index = all_subjects.index(start_subject)
    subjects_to_run = all_subjects[start_index:]
    
    for left_out_sub in subjects_to_run:
        
        # Prepare test set = left_out_sub
        X_test = subject_data[left_out_sub]['X']
        y_test = subject_data[left_out_sub]['y']

        # Prepare train set = all others
        X_train_all, y_train_all = [], []
        for s in all_subjects:
            if s == left_out_sub:
                continue
            
            
            X_train_all.append(subject_data[s]['X'])
            y_train_all.append(subject_data[s]['y'])

        print(f"=======Test subject = {left_out_sub}=======")
        
        X_train_all = np.concatenate(X_train_all, axis=0)
        y_train_all = np.concatenate(y_train_all, axis=0)

        
        # Train/Val Split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_all,
            y_train_all,
            test_size=0.15,  # e.g. 85/15
            stratify=y_train_all,
            random_state=42
        )

        # Build & Compile model
        input_length = X_train.shape[1]  # e.g. 128 or 1280
        model = EMCNN(num_classes=num_classes,
                      smoothing_kernel_size=2,
                      downsample_factor=2,
                      input_length=input_length)
        model.compile(optimizer=tf.keras.optimizers.Adam(
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.99),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

        # Train
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val, y_val), verbose=1)

        # Save model & logs
        sub_result_dir = os.path.join(result_dir_base, f"{left_out_sub}")
        os.makedirs(sub_result_dir, exist_ok=True)
        model.save(os.path.join(sub_result_dir, f"{left_out_sub}_model.keras"))
        np.save(os.path.join(sub_result_dir, f"{left_out_sub}_history.npy"), history.history)

        # Evaluate on test set
        y_pred = np.argmax(model.predict(X_test), axis=1)
        eval_path = os.path.join(sub_result_dir, f"{left_out_sub}_evaluation.txt")
        with open(eval_path, 'w') as f:
            f.write(f"Inter-subject (Left-out subject = {left_out_sub})\n")
            f.write(classification_report(y_test, y_pred))
            f.write("\nConfusion Matrix:\n")
            f.write(str(confusion_matrix(y_test, y_pred)))

        # Save confusion matrix + training curves
        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
        plt.title(f"{left_out_sub} Confusion Matrix (Inter-subject)")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.savefig(os.path.join(sub_result_dir, f"{left_out_sub}_confusion_matrix.png"))
        plt.close()

        plt.figure()
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.legend()
        plt.title(f"{left_out_sub} - Loss")
        plt.savefig(os.path.join(sub_result_dir, f"{left_out_sub}_loss.png"))
        plt.close()

        plt.figure()
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.legend()
        plt.title(f"{left_out_sub} - Accuracy")
        plt.savefig(os.path.join(sub_result_dir, f"{left_out_sub}_accuracy.png"))
        plt.close()

# -----------------------------
# 5. 메인: 모드 선택 후 전체 피실험자에 대해 intra-subject cross validation 실행
# -----------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mode = '1s'  # '1s' 또는 '10s' 또는 '10s_1soverlap'
    if mode == '1s':
        data_dir = '/home/bcml1/2025_EMOTION/DEAP_PPG_1s'
        subject_data = load_data_1s(data_dir, ch_choice_pair1=1, ch_choice_pair2=3)
        result_dir_base = '/home/bcml1/sigenv/_4월/_ppg/diff_EMCNN_noise_1s_inter'
    elif mode == '10s':
        data_dir = '/home/bcml1/2025_EMOTION/DEAP_PPG_10s_(6,5,1280)'
        subject_data = load_data_10s(data_dir, ch_choice_pair1=1, ch_choice_pair2=3)
        result_dir_base = '/home/bcml1/sigenv/_4월/_ppg/diff_EMCNN_noise_10s_inter'
    elif mode == '10s_1soverlap':
        data_dir = '/home/bcml1/2025_EMOTION/DEAP_PPG_10s_1soverlap_4label'
        subject_data = load_data_10s_1soverlap(data_dir, ch_choice_pair1=1, ch_choice_pair2=3)
        result_dir_base = '/home/bcml1/sigenv/_4월/_ppg/diff_EMCNN_noise_10s_1soverlap_inter'
    else:
        raise ValueError("mode는 '1s' 또는 '10s'여야 합니다.")

    
        
    train_model_inter_subject(subject_data,
                              num_classes=4,
                              epochs=200,
                              batch_size=64,
                              result_dir_base=result_dir_base,
                              start_subject="s15")
