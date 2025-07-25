import os
import re
import gc
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import sys  # 파일 상단에 추가

# 1) GPU 메모리 제한
def limit_gpu_memory(memory_limit_mib=5000):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit_mib)]
            )
        except RuntimeError as e:
            print(e)
limit_gpu_memory()


# # → 결과 저장 폴더 생성
# save_dir = "/home/bcml1/sigenv/_4월/_data_cross/m3_pre_DEAP_ft_SEED"
# os.makedirs(save_dir, exist_ok=True)

# 2) 모델 정의 (dtype=tf.float32 방어)
class DiffuSSMLayer(layers.Layer):
    def __init__(self, hidden_dim=64, output_units=None, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = layers.Dense(hidden_dim, activation='relu')
        self.norm1  = layers.LayerNormalization()
        self.dense2 = layers.Dense(hidden_dim, activation='relu')
        self.norm2  = layers.LayerNormalization()
        final_units = output_units if output_units else hidden_dim
        self.out_dense = layers.Dense(final_units)
        self.norm_out  = layers.LayerNormalization()

    def call(self, x):
        h = self.norm1(self.dense1(x))
        h = self.norm2(self.dense2(h))
        return self.norm_out(self.out_dense(h))

def add_diffusion_noise(x, stddev=0.05):
    return x + tf.random.normal(tf.shape(x), stddev=stddev)

def build_freq_branch_with_diffusion(input_shape=(200,8,4), noise_std=0.05):
    inp = layers.Input(shape=input_shape, dtype=tf.float32)
    x   = layers.Lambda(lambda x: add_diffusion_noise(x, stddev=noise_std))(inp)
    x   = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x   = layers.MaxPooling2D((2,2))(x)
    x   = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x   = layers.GlobalAveragePooling2D()(x)
    noisy = layers.Lambda(lambda x: add_diffusion_noise(x, stddev=noise_std))(x)
    diff  = DiffuSSMLayer()(noisy)
    out   = layers.Add()([x, diff])
    return models.Model(inp, out)

def build_chan_branch_with_diffusion(input_shape=(200,4,8), noise_std=0.05):
    inp = layers.Input(shape=input_shape, dtype=tf.float32)
    x   = layers.Lambda(lambda x: add_diffusion_noise(x, stddev=noise_std))(inp)
    x   = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x   = layers.MaxPooling2D((2,2))(x)
    x   = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x   = layers.GlobalAveragePooling2D()(x)
    noisy = layers.Lambda(lambda x: add_diffusion_noise(x, stddev=noise_std))(x)
    diff  = DiffuSSMLayer()(noisy)
    out   = layers.Add()([x, diff])
    return models.Model(inp, out)

def build_separated_model_with_diffusion(num_classes=4, noise_std=0.05):
    inp         = layers.Input(shape=(4,200,8), dtype=tf.float32, name='CSP_Input')
    f           = layers.Permute((2,3,1))(inp)
    freq_branch = build_freq_branch_with_diffusion(noise_std=noise_std)
    f_feat      = freq_branch(f)
    c           = layers.Permute((2,1,3))(inp)
    chan_branch = build_chan_branch_with_diffusion(noise_std=noise_std)
    c_feat      = chan_branch(c)
    x = layers.Concatenate()([f_feat, c_feat])
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs=inp, outputs=out, name='Separated_CNN_Model_with_Diffusion')


# 3) 데이터 로딩 (np.stack + dtype 보장)
def load_deap_csp_features(csp_feature_dir="/home/bcml1/2025_EMOTION/DEAP_eeg_new_label_CSP"):
    """
    지정된 디렉토리 내의 모든 npy 파일을 로드합니다.
    파일명 형식: folder{folder_num}_subject{subject}_sample{sample}_label{label}.npy
    각 파일은 (4, T, 8) 형태이며, 1초(200 샘플) 단위로 분할하여 (4,200,8) 샘플 생성.
    반환: X, Y, subjects, file_ids
    """
    X_list, Y_list, subjects_list, files_list = [], [], [], []
    for file_name in os.listdir(csp_feature_dir):
        if not file_name.endswith(".npy"):
            continue
        file_path = os.path.join(csp_feature_dir, file_name)
        try:
            data = np.load(file_path)  # 예상 shape: (4, T, 8)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
        if data.ndim != 3 or data.shape[0] != 4 or data.shape[2] != 8:
            print(f"Unexpected shape in {file_path}: {data.shape}")
            continue

        # label 추출: "label" 뒤의 숫자
        label_match = re.search(r'label(\d+)', file_name, re.IGNORECASE)
        if label_match is None:
            print(f"No label found in {file_name}, skipping file.")
            continue
        label = int(label_match.group(1))
        if label not in [0, 1, 2, 3]:
            print(f"Label {label} in file {file_name} is not in [0,1,2,3], skipping file.")
            continue

        # subject 추출: 파일명 내 "subject" 뒤의 숫자 (예: folder3_subject11_sample23_label1.npy)
        subject_match = re.search(r'subject(\d+)', file_name, re.IGNORECASE)
        # subject = subject_match.group(1) if subject_match else 'unknown'
        subject = int(subject_match.group(1)) if subject_match else 'unknown'
        
        T = data.shape[1]
        n_windows = T // 200  # 1초 = 200 샘플
        if n_windows < 1:
            continue
        for i in range(n_windows):
            window = data[:, i*200:(i+1)*200, :]  # (4,200,8)
            X_list.append(window)
            Y_list.append(label)
            files_list.append(f"{file_name}_win{i}")
            subjects_list.append(subject)
    X = np.array(X_list)
    Y = np.array(Y_list)
    subjects = np.array(subjects_list)
    file_ids = np.array(files_list)
    
    valid_mask = Y >= 0
    X = X[valid_mask]
    Y = Y[valid_mask]
    subjects = subjects[valid_mask]
    file_ids = file_ids[valid_mask]
    
    print(f"Loaded {X.shape[0]} samples, each sample shape: {X.shape[1:]}")
    print(f"Unique labels found: {np.unique(Y)}")
    print(f"Unique subjects found: {np.unique(subjects)}")
    return X, Y, subjects
    # return X, Y, subjects, file_ids

def load_seed_csp_features(csp_feature_dir="/home/bcml1/2025_EMOTION/SEED_IV/eeg_EM-CSP"):
    """
    지정된 디렉토리 내의 모든 npy 파일을 로드합니다.
    파일명 형식: folder{folder_num}_subject{subject}_sample{sample}_label{label}.npy
    각 파일은 (4, T, 8) 형태이며, 1초(200 샘플) 단위로 분할하여 (4,200,8) 샘플 생성.
    반환: X, Y, subjects, file_ids
    """
    """
    지정된 디렉토리 내의 모든 npy 파일을 로드합니다.
    파일명 형식: folder{folder_num}_subject{subject}_sample{sample}_label{label}.npy
    각 파일은 (4, T, 8) 형태이며, 1초(200 샘플) 단위로 분할하여 (4,200,8) 샘플 생성.
    반환: X, Y, subjects, file_ids
    """
    X_list, Y_list, subjects_list, files_list = [], [], [], []
    for file_name in os.listdir(csp_feature_dir):
        if not file_name.endswith(".npy"):
            continue
        file_path = os.path.join(csp_feature_dir, file_name)
        try:
            data = np.load(file_path)  # 예상 shape: (4, T, 8)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
        if data.ndim != 3 or data.shape[0] != 4 or data.shape[2] != 8:
            print(f"Unexpected shape in {file_path}: {data.shape}")
            continue
        T = data.shape[1]
        n_windows = T // 200  # 1초 = 200 샘플
        if n_windows < 1:
            continue
        for i in range(n_windows):
            window = data[:, i*200:(i+1)*200, :]  # (4,200,8)
            X_list.append(window)
            # subject 추출: 파일명 내 "subject" 뒤의 숫자 (예: folder3_subject11_sample23_label1.npy)
            subject_match = re.search(r'subject(\d+)', file_name, re.IGNORECASE)
            subject = int(subject_match.group(1)) if subject_match else 'unknown'
            # label 추출: "label" 뒤의 숫자
            label_match = re.search(r'label(\d+)', file_name, re.IGNORECASE)
            label = int(label_match.group(1)) if label_match else -1
            Y_list.append(label)
            files_list.append(f"{file_name}_win{i}")
            subjects_list.append(subject)
    X = np.array(X_list)
    Y = np.array(Y_list)
    subjects = np.array(subjects_list)
    file_ids = np.array(files_list)
    
    valid_mask = Y >= 0
    X = X[valid_mask]
    Y = Y[valid_mask]
    subjects = subjects[valid_mask]
    file_ids = file_ids[valid_mask]
    
    print(f"Loaded {X.shape[0]} samples, each sample shape: {X.shape[1:]}")
    print(f"Unique labels found: {np.unique(Y)}")
    print(f"Unique subjects found: {np.unique(subjects)}")
    return X, Y, subjects
    # return X, Y, subjects, file_ids

def plot_and_save_history(hist, save_path_prefix, test_metrics=None):
    """hist: Keras History, test_metrics: dict e.g. {'loss':…, 'accuracy':…}"""
    epochs = range(1, len(hist.history['loss'])+1)
    # Loss
    plt.figure()
    plt.plot(epochs, hist.history['loss'], label='train_loss')
    plt.plot(epochs, hist.history['val_loss'], label='val_loss')
    if test_metrics is not None:
        plt.hlines(test_metrics['loss'], 1, epochs[-1], colors='k', linestyles='--', label='test_loss')
    plt.title('Loss curve')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{save_path_prefix}_loss.png")
    plt.clf()

    # Accuracy
    plt.figure()
    plt.plot(epochs, hist.history['accuracy'], label='train_acc')
    plt.plot(epochs, hist.history['val_accuracy'], label='val_acc')
    if test_metrics is not None:
        plt.hlines(test_metrics['accuracy'], 1, epochs[-1], colors='k', linestyles='--', label='test_acc')
    plt.title('Accuracy curve')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"{save_path_prefix}_acc.png")
    plt.clf()
    
experiments = [
    ('DEAP', 'SEED', range(1,16), load_deap_csp_features, load_seed_csp_features),
    ('SEED', 'DEAP', range(1,33), load_seed_csp_features, load_deap_csp_features)
]

for src, tgt, subs, loader_src, loader_tgt in experiments:
    # → 실험별 결과 폴더 동적 생성
    save_dir = f"/home/bcml1/sigenv/_4월/_data_cross/m3_pre_{src}_ft_{tgt}"
    os.makedirs(save_dir, exist_ok=True)
    
    # --- 1) 사전학습 데이터 준비 ---
    src_dir = (f"/home/bcml1/2025_EMOTION/{src}_eeg_new_label_CSP"
               if src=='DEAP'
               else f"/home/bcml1/2025_EMOTION/SEED_IV/eeg_EM-CSP")
    Xs, ys, _ = loader_src(src_dir)
    X_train, X_val, y_train, y_val = train_test_split(
        Xs, ys, test_size=0.2, stratify=ys, random_state=42
    )

    # --- 2) tf.data.Dataset 생성 (tf.constant + map(cast)) ---
    batch_pre = 32
    train_ds_pre = (
        tf.data.Dataset
          .from_tensor_slices((
             tf.constant(X_train, dtype=tf.float32),
             tf.constant(y_train, dtype=tf.int32)
           ))
          .map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.int32)))
          .shuffle(len(X_train))
          .batch(batch_pre)
          .prefetch(tf.data.AUTOTUNE)
    )
    val_ds_pre = (
        tf.data.Dataset
          .from_tensor_slices((
             tf.constant(X_val, dtype=tf.float32),
             tf.constant(y_val, dtype=tf.int32)
           ))
          .map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.int32)))
          .batch(batch_pre)
          .prefetch(tf.data.AUTOTUNE)
    )

    # --- 1차 진단: element_spec & batch dtype ---
    print("train_ds_pre spec:", train_ds_pre.element_spec)
    for xb, yb in train_ds_pre.take(1):
        print(" train batch dtype:", xb.dtype, yb.dtype)
        break

    # --- 3) 사전학습 ---
    model = build_separated_model_with_diffusion()
    model.compile(
      optimizer='adam',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy']
    )
    hist_pre = model.fit(
      train_ds_pre,
      validation_data=val_ds_pre,
      epochs=5
    )
    w_pre = model.get_weights()

    # --- 4) 파인튜닝 ---
    tgt_dir = (f"/home/bcml1/2025_EMOTION/{tgt}_eeg_new_label_CSP"
               if tgt=='DEAP'
               else f"/home/bcml1/2025_EMOTION/SEED_IV/eeg_EM-CSP")
    
    Xt, yt, sub_ids = loader_tgt(tgt_dir)
    for sub in subs:
        tf.keras.backend.clear_session()
        gc.collect()

        mask = (sub_ids == sub)
        if not mask.any():
            print(f"Subject {sub}에 해당하는 데이터가 없으므로 건너뜁니다.")
            continue

        Xsub, ysub = Xt[mask], yt[mask]
        # 6:1:3 비율로 split
        X_tv, X_te, y_tv, y_te = train_test_split(Xsub, ysub, test_size=0.3, stratify=ysub, random_state=42)
        X_tr, X_va, y_tr, y_va = train_test_split(X_tv, y_tv, test_size=1/7, stratify=y_tv, random_state=42)

        batch_ft = 16
        train_ds_ft = (
            tf.data.Dataset
              .from_tensor_slices((
                  tf.constant(X_tr, dtype=tf.float32),
                  tf.constant(y_tr, dtype=tf.int32)
              ))
              .map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.int32)))
              .shuffle(len(X_tr))
              .batch(batch_ft)
              .prefetch(tf.data.AUTOTUNE)
        )
        val_ds_ft = (
            tf.data.Dataset
              .from_tensor_slices((
                  tf.constant(X_va, dtype=tf.float32),
                  tf.constant(y_va, dtype=tf.int32)
              ))
              .map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.int32)))
              .batch(batch_ft)
              .prefetch(tf.data.AUTOTUNE)
        )
        test_ds = (
            tf.data.Dataset
              .from_tensor_slices(tf.constant(X_te, dtype=tf.float32))
              .map(lambda x: tf.cast(x, tf.float32))
              .batch(batch_ft)
        )

        ft_model = build_separated_model_with_diffusion()
        ft_model.set_weights(w_pre)
        ft_model.compile(
          optimizer='adam',
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy']
        )
        hist_pre = ft_model.fit(
          train_ds_ft,
          validation_data=val_ds_ft,
          epochs=5,
          verbose=1
        )

        preds = ft_model.predict(test_ds).argmax(axis=1)
        # --- subject별 결과 저장 ---
        report_sub = classification_report(y_te, preds)
        rpt_path = os.path.join(save_dir, f"{src}_to_{tgt}_sub{sub}_report.txt")
        with open(rpt_path, "w") as f:
            f.write(report_sub)
        print(f"Report saved to {rpt_path}")

        cm = confusion_matrix(y_te, preds)
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title(f"{src}_to_{tgt} Sub{sub} Confusion Matrix")
        cm_path = os.path.join(save_dir, f"{src}_to_{tgt}_sub{sub}_cm.png")
        plt.savefig(cm_path)
        plt.clf()
        print(f"Confusion matrix saved to {cm_path}")

    print(f"All subjects done for {src}_to_{tgt}")