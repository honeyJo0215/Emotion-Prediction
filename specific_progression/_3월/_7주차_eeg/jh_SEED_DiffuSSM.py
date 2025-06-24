import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, Lambda, Flatten)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import LayerNormalization

###############################################################################
# 1. GPU 메모리 제한
###############################################################################
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

###############################################################################
# 2. 데이터 처리 함수
###############################################################################
def split_trial_to_time_slices(trial):
    """
    trial: (channels, T, bands)
    반환값: (T, channels, 1, bands)
    """
    channels, T, bands = trial.shape
    slices = []
    for t in range(T):
        # shape: (channels, 1, bands)
        slice_t = trial[:, t:t+1, :]
        slices.append(slice_t)
    return np.array(slices)  # (T, channels, 1, bands)

def replicate_labels_for_trials(trials_list, labels):
    """
    trial.shape[1] == T (시간길이)에 맞춰서 레이블을 복제
    """
    replicated = []
    for trial, label in zip(trials_list, labels):
        # trial.shape = (channels, T, bands)
        T = trial.shape[1]
        # label.shape = (4,) 같은 one-hot 벡터라고 가정
        repeated = np.repeat(label[np.newaxis, :], T, axis=0)  # (T, num_classes)
        replicated.append(repeated)
    return np.concatenate(replicated, axis=0) if replicated else None

def load_seediv_data(base_dirs, de_keys=["de_movingAve"], psd_keys=["psd_movingAve"], debug=False):
    data_de = {}
    data_psd = {}
    file_counter = 0
    for base_dir in base_dirs:
        file_list = glob.glob(os.path.join(base_dir, "*.npy"))
        for file in file_list:
            filename = os.path.basename(file)
            parts = filename.replace('.npy','').split('_')
            if len(parts) < 8:
                if debug:
                    print(f"DEBUG: Skipping file {filename} because parts < 8")
                continue
            subject, trial = parts[0], parts[3]
            key_name = parts[4] + "_" + parts[5]
            try:
                label_val = int(parts[7])
            except Exception as e:
                if debug:
                    print(f"DEBUG: Could not extract label from {filename}: {e}")
                continue
            arr = np.load(file)[..., 1:]  # shape: (channels, T, bands)
            if key_name in de_keys:
                data_de[(subject, trial)] = (arr, label_val)
            elif key_name in psd_keys:
                data_psd[(subject, trial)] = (arr, label_val)
            file_counter += 1
    if debug:
        print(f"DEBUG: Total files processed: {file_counter}")
    
    common_ids = set(data_de.keys()).intersection(set(data_psd.keys()))
    if debug:
        print(f"DEBUG: Found {len(common_ids)} common IDs between DE and PSD")
    de_list, psd_list, label_list, subject_list = [], [], [], []
    for sid in sorted(common_ids):
        subj, trial = sid
        arr_de, label_de = data_de[sid]
        arr_psd, label_psd = data_psd[sid]
        if label_de != label_psd:
            if debug:
                print(f"DEBUG: Mismatch in labels for {sid}: DE label {label_de} vs PSD label {label_psd}")
            continue
        de_list.append(arr_de)
        psd_list.append(arr_psd)
        label_list.append(label_de)
        subject_list.append(subj)
    
    return de_list, psd_list, label_list, subject_list

###############################################################################
# 3. 채널 위치 매핑 (9x9 grid)
###############################################################################
channel_positions_9x9 = {
    0: (0,3),   1: (0,4),   2: (0,5),
    3: (1,3),   4: (1,5),
    5: (2,0),   6: (2,1),   7: (2,2),   8: (2,3),   9: (2,4),
    10: (2,5),  11: (2,6),  12: (2,7),  13: (2,8),
    14: (3,0),  15: (3,1),  16: (3,2),  17: (3,3),  18: (3,4),
    19: (3,5),  20: (3,6),  21: (3,7),  22: (3,8),
    23: (4,0),  24: (4,1),  25: (4,2),  26: (4,3),  27: (4,4),
    28: (4,5),  29: (4,6),  30: (4,7),  31: (4,8),
    32: (5,0),  33: (5,1),  34: (5,2),  35: (5,3),  36: (5,4),
    37: (5,5),  38: (5,6),  39: (5,7),  40: (5,8),
    41: (6,0),  42: (6,1),  43: (6,2),  44: (6,3),  45: (6,4),
    46: (6,5),  47: (6,6),  48: (6,7),  49: (6,8),
    50: (7,1),  51: (7,2),  52: (7,3),  53: (7,4),  54: (7,5),
    55: (7,6),  56: (7,7),
    57: (8,2),  58: (8,3),  59: (8,4),  60: (8,5),  61: (8,6)
}

def map_channels_to_2d_9x9(segment, channel_positions):
    """
    segment: (62, bands)
    mapped: (9,9,bands)
    """
    num_channels, num_bands = segment.shape
    mapped = np.zeros((9, 9, num_bands), dtype=np.float32)
    for ch_idx in range(num_channels):
        if ch_idx not in channel_positions:
            continue
        row, col = channel_positions[ch_idx]
        mapped[row, col, :] = segment[ch_idx, :]
    return mapped

def process_slice_9x9(slice_, channel_positions):
    """
    slice_: (62,1,4) → squeeze → (62,4) → map → (9,9,4) → transpose → (4,9,9,1)
    """
    slice_2d = np.squeeze(slice_, axis=1)  # (62,4)
    mapped_9x9 = map_channels_to_2d_9x9(slice_2d, channel_positions)  # (9,9,4)
    # (bands, row, col)
    img = np.transpose(mapped_9x9, (2, 0, 1))  # (4,9,9)
    img = np.expand_dims(img, axis=-1)        # (4,9,9,1)
    return img

def process_trial_9x9(de_trial, psd_trial, channel_positions):
    """
    de_trial, psd_trial: (channels, T, bands) => split => (T, channels, 1, bands)
    최종 shape: (T, 2, 4, 9, 9, 1)
    """
    slices_de = split_trial_to_time_slices(de_trial)   # (T, 62,1,4)
    slices_psd = split_trial_to_time_slices(psd_trial) # (T, 62,1,4)
    T = slices_de.shape[0]
    sample_list = []
    for t in range(T):
        de_img = process_slice_9x9(slices_de[t], channel_positions)   # (4,9,9,1)
        psd_img = process_slice_9x9(slices_psd[t], channel_positions) # (4,9,9,1)
        # (2,4,9,9,1)
        sample = np.stack([de_img, psd_img], axis=0)
        sample_list.append(sample)
    return np.array(sample_list)

###############################################################################
# 4. 모델 관련 레이어들
###############################################################################
class DynamicGraphConvLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, debug=False, **kwargs):
        super(DynamicGraphConvLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.debug = debug
        self.dense_proj = tf.keras.layers.Dense(output_dim, activation='relu',
                                                kernel_regularizer=tf.keras.regularizers.l2(1e-6))
        self.fc1 = tf.keras.layers.Dense(output_dim, activation='elu',
                                         kernel_regularizer=tf.keras.regularizers.l2(1e-6))
        self.fc2 = tf.keras.layers.Dense(output_dim, activation='tanh',
                                         kernel_regularizer=tf.keras.regularizers.l2(1e-6))
    def call(self, inputs):
        proj = self.dense_proj(inputs)  # shape: (batch, N, output_dim)
        A = tf.matmul(proj, proj, transpose_b=True)  # (batch, N, N)
        A = tf.nn.softmax(A, axis=-1)  # 소프트맥스로 인접행렬
        if self.debug:
            tf.print("A softmax mean =", tf.reduce_mean(A), "std =", tf.math.reduce_std(A))
        H = tf.matmul(A, inputs)       # (batch, N, input_dim)
        H = self.fc1(H)
        H = self.fc2(H)
        return H

class AdapterModule(tf.keras.layers.Layer):
    def __init__(self, d_model, bottleneck_dim=16, **kwargs):
        super(AdapterModule, self).__init__(**kwargs)
        self.down = tf.keras.layers.Dense(bottleneck_dim, activation='elu',
                                          kernel_regularizer=tf.keras.regularizers.l2(1e-6))
        self.up = tf.keras.layers.Dense(d_model, activation=None,
                                        kernel_regularizer=tf.keras.regularizers.l2(1e-6))
    def call(self, x):
        shortcut = x
        x = self.down(x)
        x = self.up(x)
        return x + shortcut

class DiffuSSMLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, dropout_rate=0.1, bottleneck_dim=16, **kwargs):
        super(DiffuSSMLayer, self).__init__(**kwargs)
        self.hourglass_down = tf.keras.layers.Dense(d_model // 2, activation=tf.nn.gelu,
                                                    kernel_regularizer=tf.keras.regularizers.l2(1e-6))
        self.hourglass_up = tf.keras.layers.Dense(d_model, activation=None,
                                                  kernel_regularizer=tf.keras.regularizers.l2(1e-6))
        self.bi_gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(d_model, return_sequences=True), merge_mode='sum'
        )
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation=tf.nn.gelu,
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-6)),
            tf.keras.layers.Dense(d_model,
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-6))
        ])
        self.dropout_ffn = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm_ffn = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.adapter = AdapterModule(d_model, bottleneck_dim)
        self.alpha = self.add_weight(shape=(1,), initializer=tf.zeros_initializer(), trainable=True, name="ssm_alpha")
    
    def call(self, inputs, training=False):
        x_hour = self.hourglass_down(inputs)
        x_hour = self.hourglass_up(x_hour)
        x_hour = self.dropout(x_hour, training=training)
        ssm_out = self.bi_gru(inputs, training=training)
        ssm_out = self.dropout(ssm_out, training=training)
        ssm_out = self.alpha * ssm_out
        x = inputs + x_hour + ssm_out
        x = self.layernorm(x)
        ffn_out = self.ffn(x)
        ffn_out = self.dropout_ffn(ffn_out, training=training)
        x = self.layernorm_ffn(x + ffn_out)
        x = self.adapter(x)
        return x

###############################################################################
# 5. 단일 시점 모델 (2,4,9,9,1) 입력 → 특징 추출
###############################################################################
def build_single_segment_model(d_model=64, num_nodes=62, debug=False):
    """
    입력 shape: (2,4,9,9,1)
    """
    input_seg = Input(shape=(2,4,9,9,1))
    de_input = Lambda(lambda x: x[:, 0, ...])(input_seg)   # (batch, 4,9,9,1)
    psd_input = Lambda(lambda x: x[:, 1, ...])(input_seg)  # (batch, 4,9,9,1)
    flat_de = Flatten()(de_input)   # (batch, 4*9*9*1)
    flat_psd = Flatten()(psd_input) # (batch, 4*9*9*1)
    
    x_de = Dense(num_nodes * d_model, activation="relu", kernel_regularizer=l2(1e-6))(flat_de)
    x_de = Lambda(lambda x: tf.reshape(x, (-1, num_nodes, d_model)))(x_de)  # (batch, 62, d_model)
    
    x_psd = Dense(num_nodes * d_model, activation="relu", kernel_regularizer=l2(1e-6))(flat_psd)
    x_psd = Lambda(lambda x: tf.reshape(x, (-1, num_nodes, d_model)))(x_psd) # (batch, 62, d_model)
    
    H_de = DynamicGraphConvLayer(d_model, debug=debug)(x_de)    # (batch, 62, d_model)
    H_psd = DynamicGraphConvLayer(d_model, debug=debug)(x_psd)  # (batch, 62, d_model)
    H_fused = tf.keras.layers.Concatenate(axis=-1)([H_de, H_psd])  # (batch, 62, 2*d_model)
    
    fused = Dense(d_model, activation="relu", kernel_regularizer=l2(1e-6))(H_fused)  # (batch, 62, d_model)
    out_feature = Lambda(lambda x: tf.reduce_mean(x, axis=1))(fused)  # (batch, d_model)
    model = Model(inputs=input_seg, outputs=out_feature)
    return model

###############################################################################
# 6. 최종 모델 구성 (Diffusion + 분류)
###############################################################################
def build_emotion_diffusion_model(input_shape=(2,4,9,9,1), 
                                  n_layers=2, d_ff=512, p_drop=0.1,
                                  d_model=64, noise_std=0.02, num_classes=4,
                                  debug=False):
    """
    - single_seg_model: (2,4,9,9,1) → (d_model,)
    - 가우시안 노이즈 추가
    - n_layers번 Dense 통과
    - 최종 Softmax
    """
    inputs = Input(shape=input_shape)
    single_seg_model = build_single_segment_model(d_model=d_model, debug=debug)
    features = single_seg_model(inputs)  # (batch, d_model)
    
    def add_diffusion_noise(x):
        noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=noise_std)
        return x + noise
    
    noisy_features = Lambda(add_diffusion_noise, name="add_diffusion_noise")(features)
    x = noisy_features
    for _ in range(n_layers):
        x = Dense(d_model, activation='relu', kernel_regularizer=l2(1e-6))(x)
    
    outputs = Dense(num_classes, activation="softmax", kernel_regularizer=l2(1e-6))(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

###############################################################################
# 7. 유틸리티 콜백 및 함수
###############################################################################
class GradualUnfreeze(Callback):
    def __init__(self, unfreeze_schedule, unfreeze_lr=3e-4):
        super().__init__()
        self.unfreeze_schedule = unfreeze_schedule
        self.unfreeze_lr = unfreeze_lr
    def on_epoch_begin(self, epoch, logs=None):
        if epoch in self.unfreeze_schedule:
            print(f"\nUnfreezing layers at epoch {epoch}...")
            for layer_name in self.unfreeze_schedule[epoch]:
                for layer in self.model.layers:
                    if layer.name == layer_name:
                        layer.trainable = True
                        print(f"Layer {layer.name} unfreezed.")
            self.model.optimizer.learning_rate.assign(self.unfreeze_lr)
            print(f"Learning rate set to {self.unfreeze_lr} after unfreezing.")

def check_data_label_alignment(X, Y, data_name="Data"):
    """
    X.shape = (total_slices, 2,4,9,9,1)
    Y.shape = (total_slices, num_classes)
    """
    print(f"{data_name} X shape: {X.shape}, Y shape: {Y.shape}")
    if X.shape[0] != Y.shape[0]:
        print(f"[WARNING] {data_name} X and Y have different 1st dimension!")
    else:
        print(f"[OK] {data_name} X and Y match in first dimension.")

def compute_channel_importance(model, X_test_final, y_test, baseline_acc, channel_positions, batch_size=16):
    """
    X_test_final shape: (batch, 2,4,9,9,1)
    => 인덱싱: X_test_mod[:, :, :, row, col, :]
    """
    num_channels = 62
    channel_importance = np.zeros(num_channels, dtype=np.float32)
    for ch in range(num_channels):
        X_test_mod = np.copy(X_test_final)
        if ch in channel_positions:
            row, col = channel_positions[ch]
            # 기존: X_test_mod[:, :, :, :, row, col, :] = 0.0 (7차원 인덱싱) -> 에러
            # 수정: (batch, 2, 4, 9, 9, 1)에서 row, col은 3,4번째 축에 해당
            X_test_mod[:, :, :, row, col, :] = 0.0
        test_dataset_mod = tf.data.Dataset.from_tensor_slices(
            (X_test_mod, to_categorical(y_test, num_classes=4))
        ).batch(batch_size)
        _, masked_acc = model.evaluate(test_dataset_mod, verbose=0)
        channel_importance[ch] = baseline_acc - masked_acc
    
    importance_map = np.zeros((9, 9), dtype=np.float32)
    for ch in range(num_channels):
        if ch in channel_positions:
            r, c = channel_positions[ch]
            importance_map[r, c] = channel_importance[ch]
    return importance_map, channel_importance

def plot_channel_importance(importance_map, save_path="channel_importance.png"):
    plt.figure(figsize=(6,5))
    sns.heatmap(importance_map, annot=True, fmt=".3f", cmap="Reds")
    plt.title("Channel Importance (Permutation-based)")
    plt.xlabel("X-axis (columns)")
    plt.ylabel("Y-axis (rows)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Channel importance heatmap saved to {save_path}")

###############################################################################
# 8. 메인 학습 함수
###############################################################################
def inter_subject_cv_training_9x9(base_dirs, result_dir,
                                  epochs=150, batch_size=16,
                                  target_subjects=None,
                                  pretrain_lr=1e-3, fine_tune_lr=1e-3, unfreeze_lr=1e-5,
                                  toy_test=False,
                                  debug=False):
    """
    toy_test=True: 작은 규모 데이터(예: 첫 몇 개 trial만)로 모델이 100% 가까이 학습 가능한지 확인.
    debug=True: DynamicGraphConvLayer에서 인접행렬 분포 등 출력.
    """
    os.makedirs(result_dir, exist_ok=True)
    overall_folder = os.path.join(result_dir, "overall")
    os.makedirs(overall_folder, exist_ok=True)
    
    # 1) 데이터 로드
    de_trials, psd_trials, label_list, subject_list = load_seediv_data(base_dirs, debug=debug)
    subject_data = {}
    for de, psd, label, subj in zip(de_trials, psd_trials, label_list, subject_list):
        if subj not in subject_data:
            subject_data[subj] = {"de": [], "psd": [], "labels": []}
        subject_data[subj]["de"].append(de)
        subject_data[subj]["psd"].append(psd)
        subject_data[subj]["labels"].append(label)
    subjects = sorted(subject_data.keys(), key=lambda x: int(x) if x.isdigit() else x)
    if target_subjects is not None:
        subjects = [subj for subj in subjects if subj in target_subjects]
    
    overall_acc = {}
    overall_reports = {}
    
    # 2) 소규모 toy 테스트 (선택 사항)
    if toy_test:
        # 예시: 첫 subject의 첫 2개 trial만 사용
        test_subj = subjects[0]
        print(f"[TOY TEST] Using subject={test_subj}, first 2 trials only.")
        X_de_small = subject_data[test_subj]["de"][:2]
        X_psd_small = subject_data[test_subj]["psd"][:2]
        y_small = subject_data[test_subj]["labels"][:2]
        
        # one-hot
        y_small_oh = to_categorical(y_small, num_classes=4)
        
        # process
        X_small_slices = [process_trial_9x9(de, psd, channel_positions_9x9) for de, psd in zip(X_de_small, X_psd_small)]
        y_small_rep = replicate_labels_for_trials(X_de_small, y_small_oh)
        X_small_final = np.concatenate(X_small_slices, axis=0)
        
        print("[TOY TEST] X_small_final shape:", X_small_final.shape)
        print("[TOY TEST] y_small_rep shape:", y_small_rep.shape)
        
        # 모델 빌드
        model_toy = build_emotion_diffusion_model(debug=debug)
        model_toy.compile(optimizer=Adam(learning_rate=1e-3),
                          loss="categorical_crossentropy", metrics=["accuracy"])
        
        # 과연 이 작은 데이터에 과적합(=정확도 100%) 가까이 되는지 확인
        hist_toy = model_toy.fit(X_small_final, y_small_rep,
                                 batch_size=batch_size, epochs=50, verbose=1)
        return  # toy_test 모드에서는 여기서 종료

    ############################################################################
    # 3) 본 학습 (LOSO)
    ############################################################################
    for test_subj in subjects:
        print(f"\n========== LOSO: Test subject: {test_subj} ==========")
        X_de_test_trials = subject_data[test_subj]["de"]
        X_psd_test_trials = subject_data[test_subj]["psd"]
        y_test_trials = np.array(subject_data[test_subj]["labels"])
        
        test_indices = np.arange(len(X_de_test_trials))
        # 20%를 fine-tune 용으로 떼고, 나머지를 final test
        ft_idx, final_idx = train_test_split(test_indices, train_size=0.2,
                                             random_state=42, stratify=y_test_trials)
        X_de_test_ft = [X_de_test_trials[i] for i in ft_idx]
        X_psd_test_ft = [X_psd_test_trials[i] for i in ft_idx]
        y_test_ft_cat = to_categorical([y_test_trials[i] for i in ft_idx], num_classes=4)
        
        X_de_test_final = [X_de_test_trials[i] for i in final_idx]
        X_psd_test_final = [X_psd_test_trials[i] for i in final_idx]
        y_test_final_cat = to_categorical([y_test_trials[i] for i in final_idx], num_classes=4)
        
        # 학습 데이터(테스트 subject 제외)
        X_de_train_trials = []
        X_psd_train_trials = []
        y_train_list = []
        for subj in subjects:
            if subj == test_subj:
                continue
            X_de_train_trials.extend(subject_data[subj]["de"])
            X_psd_train_trials.extend(subject_data[subj]["psd"])
            y_train_list.extend(subject_data[subj]["labels"])
        y_train_list = np.array(y_train_list)
        y_cat_train = to_categorical(y_train_list, num_classes=4)
        
        # train/val split
        num_train_trials = len(X_de_train_trials)
        indices = np.arange(num_train_trials)
        train_idx, val_idx = train_test_split(indices, test_size=0.2,
                                              random_state=42, stratify=y_train_list)
        
        X_de_train_split = [X_de_train_trials[i] for i in train_idx]
        X_psd_train_split = [X_psd_train_trials[i] for i in train_idx]
        X_de_val_split = [X_de_train_trials[i] for i in val_idx]
        X_psd_val_split = [X_psd_train_trials[i] for i in val_idx]
        
        y_train_split = y_cat_train[train_idx]
        y_val_split = y_cat_train[val_idx]
        
        # 4) time-slice & 레이블 복제
        X_train_slices = [process_trial_9x9(de, psd, channel_positions_9x9)
                          for de, psd in zip(X_de_train_split, X_psd_train_split)]
        y_train = replicate_labels_for_trials(X_de_train_split, y_train_split)
        X_train = np.concatenate(X_train_slices, axis=0)
        
        X_val_slices = [process_trial_9x9(de, psd, channel_positions_9x9)
                        for de, psd in zip(X_de_val_split, X_psd_val_split)]
        y_val = replicate_labels_for_trials(X_de_val_split, y_val_split)
        X_val = np.concatenate(X_val_slices, axis=0)
        
        X_test_ft_slices = [process_trial_9x9(de, psd, channel_positions_9x9)
                            for de, psd in zip(X_de_test_ft, X_psd_test_ft)]
        y_test_ft = replicate_labels_for_trials(X_de_test_ft, y_test_ft_cat)
        X_test_ft = np.concatenate(X_test_ft_slices, axis=0)
        
        X_test_final_slices = [process_trial_9x9(de, psd, channel_positions_9x9)
                               for de, psd in zip(X_de_test_final, X_psd_test_final)]
        y_test_final = replicate_labels_for_trials(X_de_test_final, y_test_final_cat)
        X_test_final = np.concatenate(X_test_final_slices, axis=0)
        
        # 데이터·레이블 정렬 확인
        check_data_label_alignment(X_train, y_train, "Train")
        check_data_label_alignment(X_val, y_val, "Val")
        check_data_label_alignment(X_test_ft, y_test_ft, "Test_FT")
        check_data_label_alignment(X_test_final, y_test_final, "Test_Final")
        
        # 5) Dataset 구성
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
        ft_dataset = tf.data.Dataset.from_tensor_slices((X_test_ft, y_test_ft)).shuffle(1000).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test_final, y_test_final)).batch(batch_size)
        
        # 6) 모델 빌드 & Pre-training
        print(f"\n[Test Subject: {test_subj}] Starting pre-training for {epochs} epochs...")
        model = build_emotion_diffusion_model(input_shape=(2,4,9,9,1),
                                              n_layers=2, d_ff=512, p_drop=0.1,
                                              d_model=64, noise_std=0.02,
                                              num_classes=4,
                                              debug=debug)
        model.compile(optimizer=Adam(learning_rate=pretrain_lr),
                      loss="categorical_crossentropy", metrics=["accuracy"])
        model.summary()
        
        pretrain_callbacks = [EarlyStopping(monitor='val_accuracy', patience=50,
                                            min_delta=0.001, restore_best_weights=True)]
        history_pretrain = model.fit(train_dataset, epochs=epochs,
                                     validation_data=val_dataset,
                                     callbacks=pretrain_callbacks, verbose=1)
        
        # 학습 곡선 저장
        subj_folder = os.path.join(result_dir, f"s{test_subj.zfill(2)}")
        os.makedirs(subj_folder, exist_ok=True)
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(history_pretrain.history['loss'], label='Pre-train Train Loss')
        plt.plot(history_pretrain.history['val_loss'], label='Pre-train Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Pre-training Loss Curve')
        
        plt.subplot(1,2,2)
        plt.plot(history_pretrain.history['accuracy'], label='Pre-train Train Acc')
        plt.plot(history_pretrain.history['val_accuracy'], label='Pre-train Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Pre-training Accuracy Curve')
        
        pretrain_curve_path = os.path.join(subj_folder, "pretrain_training_curves.png")
        plt.savefig(pretrain_curve_path)
        plt.close()
        print(f"[Test Subject {test_subj}] Pre-training curves saved to {pretrain_curve_path}")
        
        # 7) Fine-tuning (20 epochs)
        print(f"[Test Subject {test_subj}] Fine tuning with {X_test_ft.shape[0]} time-slices.")
        model.fit(ft_dataset, epochs=20, verbose=1)
        
        # 8) 최종 테스트
        test_loss, test_acc = model.evaluate(test_dataset, verbose=0)
        print(f"[Test Subject {test_subj}] Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        overall_acc[test_subj] = test_acc
        
        # 9) 채널 중요도 (Permutation) 분석
        y_test_int = np.argmax(y_test_final, axis=-1)
        importance_map, channel_imp = compute_channel_importance(
            model=model,
            X_test_final=X_test_final,
            y_test=y_test_int,
            baseline_acc=test_acc,
            channel_positions=channel_positions_9x9,
            batch_size=batch_size
        )
        imp_fig_path = os.path.join(subj_folder, "channel_importance.png")
        plot_channel_importance(importance_map, save_path=imp_fig_path)
        np.save(os.path.join(subj_folder, "channel_importance.npy"), channel_imp)
        
        # 10) 분류 결과 (Confusion Matrix, Classification Report)
        y_pred_prob = model.predict(test_dataset)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(np.concatenate([y for _, y in test_dataset], axis=0), axis=1)
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, digits=2)
        overall_reports[test_subj] = report
        
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        cm_path = os.path.join(subj_folder, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        print(f"[Test Subject {test_subj}] Confusion matrix saved to {cm_path}")
        
        report_path = os.path.join(subj_folder, "classification.txt")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"[Test Subject {test_subj}] Classification report saved to {report_path}")
        
        # 모델 저장
        model_save_path = os.path.join(subj_folder, "model_eeg_9x9.keras")
        model.save(model_save_path)
        print(f"[Test Subject {test_subj}] Model saved to {model_save_path}")
    
    # 11) 전체 결과 요약
    overall_avg_acc = np.mean(list(overall_acc.values()))
    overall_report_path = os.path.join(overall_folder, "overall_classification.txt")
    with open(overall_report_path, "w") as f:
        f.write("Overall LOSO Test Accuracy: {:.4f}\n\n".format(overall_avg_acc))
        for subj in sorted(overall_reports.keys(), key=lambda x: int(x) if x.isdigit() else x):
            f.write(f"Test Subject {subj}:\n")
            f.write(overall_reports[subj])
            f.write("\n\n")
    print(f"[DONE] Overall results saved to {overall_report_path}")


###############################################################################
# 9. 실제 실행 부분
###############################################################################
if __name__ == "__main__":
    base_dirs = [
        "/home/bcml1/2025_EMOTION/SEED_IV/eeg_feature_smooth/1_npy_sample",
        "/home/bcml1/2025_EMOTION/SEED_IV/eeg_feature_smooth/2_npy_sample",
        "/home/bcml1/2025_EMOTION/SEED_IV/eeg_feature_smooth/3_npy_sample"
    ]
    RESULT_DIR = "/home/bcml1/sigenv/_7주차_eeg/jh_Diffu_1"
    
    inter_subject_cv_training_9x9(
        base_dirs=base_dirs,
        result_dir=RESULT_DIR,
        epochs=80,
        batch_size=16,
        target_subjects=[str(i) for i in range(1, 16)],
        pretrain_lr=1e-3,
        fine_tune_lr=1e-3,
        unfreeze_lr=1e-5,
        toy_test=False,   # True로 두면 아주 작은 데이터만으로 과적합 테스트를 합니다.
        debug=False       # True로 두면 DynamicGraphConvLayer에서 A 분포를 출력
    )
