#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diffusion 노이즈 + SVM 통합 예제
- EEG CSP 입력 → 주파수/채널 브랜치별로 노이즈 추가 후 Flatten → SVM 분류
- 실험: src 데이터(예: SEED)로 SVM 학습 → tgt 데이터(예: DEAP) 각 피실험자별 평가
"""
import os, re, gc
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
# GPU 메모리 제한 (필요시 사용)
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

tf.config.optimizer.set_jit(False)

# =============================================================================
# 1) BidirectionalSSMLayer (변경 없음)
class BidirectionalSSMLayer(layers.Layer):
    def __init__(self, units, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_size = kernel_size
        self.fwd_conv = layers.Conv1D(filters=1,
                                       kernel_size=kernel_size,
                                       padding='causal',
                                       activation=None,
                                       name="ssm_fwd_conv")
        self.bwd_conv = layers.Conv1D(filters=1,
                                       kernel_size=kernel_size,
                                       padding='causal',
                                       activation=None,
                                       name="ssm_bwd_conv")

    def call(self, x, training=False):
        seq = tf.expand_dims(x, axis=-1)           # (batch, units, 1)
        fwd = self.fwd_conv(seq)                   # (batch, units, 1)
        rev_seq = tf.reverse(seq, axis=[1])
        bwd = self.bwd_conv(rev_seq)
        bwd = tf.reverse(bwd, axis=[1])            # (batch, units, 1)
        out = fwd + bwd
        return tf.squeeze(out, axis=-1)            # (batch, units)

# =============================================================================
# 2) DiffuSSMLayer (변경 없음)
class DiffuSSMLayer(layers.Layer):
    def __init__(self, model_dim, hidden_dim, cond_dim=None, alpha_init=0.0, **kwargs):
        super().__init__(**kwargs)
        self.norm = layers.LayerNormalization()
        self.cond_mlp = layers.Dense(2 * model_dim, name="cond_mlp") if cond_dim else None
        self.hg_down = layers.Dense(hidden_dim, activation='swish', name="hourglass_down")
        self.ssm     = BidirectionalSSMLayer(hidden_dim, kernel_size=3)
        self.hg_up   = layers.Dense(model_dim, activation=None, name="hourglass_up")
        self.alpha   = self.add_weight(name="fusion_scale_alpha",
                                       shape=(),
                                       initializer=tf.keras.initializers.Constant(alpha_init),
                                       trainable=True)

    def call(self, x, cond=None, training=False):
        x_ln = self.norm(x)
        if self.cond_mlp is not None and cond is not None:
            gamma, beta = tf.split(self.cond_mlp(cond), 2, axis=-1)
            x_ln = gamma * x_ln + beta
        h = self.hg_down(x_ln)
        h = self.ssm(h)
        h = self.hg_up(h)
        return x + self.alpha * h

# =============================================================================
# 3) 가우시안 노이즈 추가 함수
def add_diffusion_noise(x, stddev=0.05):
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=stddev)
    return x + noise

def build_diffussm_emotion_model(
    input_shape=(4, 200, 8),
    noise_std=0.05,
    hidden_dim=64,
    num_classes=4
):
    inp = Input(shape=input_shape, name="EEG_Input")
    
    # 1) 노이즈 추가 (확산 노이즈)
    x = layers.Lambda(lambda x: x + tf.random.normal(tf.shape(x), stddev=noise_std),
                      name="DiffusionNoise")(inp)
    
    # 2) Flatten → DiffuSSM
    flat = layers.Flatten(name="Flatten")(x)  # (batch, 6400)
    dssm = DiffuSSMLayer(model_dim=np.prod(input_shape),
                         hidden_dim=hidden_dim,
                         name="DiffuSSM")(flat)     # (batch, 6400)
    
    # 3) 분류 헤드
    fc = layers.Dense(128, activation='relu', name="FC1")(dssm)
    dp = layers.Dropout(0.3, name="Dropout1")(fc)
    out = layers.Dense(num_classes, activation='softmax', name="Output")(dp)
    
    return models.Model(inputs=inp, outputs=out, name="DiffuSSM_EmotionModel")

# 3) 로더 함수 (unchanged from before)… 데이터 로딩 (np.stack + dtype 보장)
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


def plot_and_save_history(hist, save_path_prefix, test_metrics=None):
    epochs = range(1, len(hist.history['loss'])+1)
    plt.figure()
    plt.plot(epochs, hist.history['loss'], label='train_loss')
    plt.plot(epochs, hist.history['val_loss'], label='val_loss')
    if test_metrics:
        plt.hlines(test_metrics['loss'], 1, epochs[-1], linestyles='--', label='test_loss')
    plt.title('Loss curve'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.savefig(f"{save_path_prefix}_loss.png"); plt.clf()

    plt.figure()
    plt.plot(epochs, hist.history['accuracy'], label='train_acc')
    plt.plot(epochs, hist.history['val_accuracy'], label='val_acc')
    if test_metrics:
        plt.hlines(test_metrics['accuracy'], 1, epochs[-1], linestyles='--', label='test_acc')
    plt.title('Accuracy curve'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.savefig(f"{save_path_prefix}_acc.png"); plt.clf()

experiments = [
    ('DEAP', 'SEED', range(1,16), load_deap_csp_features, load_seed_csp_features),
    ('SEED', 'DEAP', range(1,33), load_seed_csp_features, load_deap_csp_features)
]

# =============================================================================
#  실행부: build_diffussm_emotion_model 로 연결되도록 수정한 부분
# =============================================================================
def plot_and_save_history(hist, save_path_prefix, test_metrics=None):
    epochs = range(1, len(hist.history['loss'])+1)
    plt.figure()
    plt.plot(epochs, hist.history['loss'], label='train_loss')
    plt.plot(epochs, hist.history['val_loss'], label='val_loss')
    if test_metrics:
        plt.hlines(test_metrics['loss'], 1, epochs[-1], colors='k', linestyles='--', label='test_loss')
    plt.title('Loss curve'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.savefig(f"{save_path_prefix}_loss.png"); plt.clf()
    plt.figure()
    plt.plot(epochs, hist.history['accuracy'], label='train_acc')
    plt.plot(epochs, hist.history['val_accuracy'], label='val_acc')
    if test_metrics:
        plt.hlines(test_metrics['accuracy'], 1, epochs[-1], colors='k', linestyles='--', label='test_acc')
    plt.title('Accuracy curve'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.savefig(f"{save_path_prefix}_acc.png"); plt.clf()

save_dir = f"/home/bcml1/sigenv/_5월/svm_data_cross/SVM_datacross/pre_{src}_ft_{tgt}"

# experiment 설정
experiments = [
    ('DEAP', 'SEED', range(1,16), load_deap_csp_features, load_seed_csp_features),
    ('SEED', 'DEAP', range(1,33), load_seed_csp_features, load_deap_csp_features)
]

# 실행부: inter-subject 평가 방식
for src, tgt, subs, loader_src, loader_tgt in datasets:
    save_dir = f"/home/bcml1/sigenv/_5월/svm_data_cross/SVM_datacross/pre_{src}_ft_{tgt}"
    os.makedirs(save_dir, exist_ok=True)

    # 1) Src 데이터로 사전학습
    Xs, ys, _ = loader_src(dirs[src])
    X_tr, X_va, y_tr, y_va = train_test_split(Xs, ys, test_size=0.2, stratify=ys, random_state=42)
    num_classes = len(np.unique(ys))
    model_pre = build_diffussm_emotion_model(num_classes=num_classes)
    model_pre.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])
    hist_pre = model_pre.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                             epochs=300, batch_size=32,
                             callbacks=[callbacks.EarlyStopping('val_loss', patience=200, restore_best_weights=True)])
    plot_and_save_history(hist_pre, os.path.join(save_dir, f"{src}_pretrain"))
    w_pre = model_pre.get_weights()

    # 2) Inter-subject 평가 (tgt 데이터)
    Xt, yt, sub_ids = loader_tgt(dirs[tgt])
    overall_true, overall_pred = [], []

    for sub in subs:
        tf.keras.backend.clear_session(); gc.collect()
        mask_test = (sub_ids == sub)
        if not mask_test.any(): continue
        X_test, y_test = Xt[mask_test], yt[mask_test]
        X_train_all, y_train_all = Xt[~mask_test], yt[~mask_test]

        X_tr2, X_va2, y_tr2, y_va2 = train_test_split(
            X_train_all, y_train_all, test_size=0.2, stratify=y_train_all, random_state=42)

        model_ft = build_diffussm_emotion_model(num_classes=num_classes)
        model_ft.set_weights(w_pre)
        model_ft.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])
        hist_ft = model_ft.fit(X_tr2, y_tr2, validation_data=(X_va2, y_va2),
                               epochs=500, batch_size=16,
                               callbacks=[callbacks.EarlyStopping('val_loss', patience=200, restore_best_weights=True)])

        loss, acc = model_ft.evaluate(X_test, y_test, verbose=0)
        prefix = os.path.join(save_dir, f"{src}_to_{tgt}_sub{sub}")
        plot_and_save_history(hist_ft, prefix, {'loss': loss, 'accuracy': acc})

        preds = model_ft.predict(X_test).argmax(axis=1)
        with open(f"{prefix}_report.txt", "w") as f:
            f.write(classification_report(y_test, preds))
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title(f"{src}→{tgt} Sub{sub} CM (Inter)")
        plt.savefig(f"{prefix}_cm.png"); plt.clf()

        overall_true.extend(y_test)
        overall_pred.extend(preds)

    # 전체 결과 저장
    with open(os.path.join(save_dir, "overall_report_inter.txt"), "w") as f:
        f.write(classification_report(overall_true, overall_pred))
    cm = confusion_matrix(overall_true, overall_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"{src}→{tgt} Overall CM (Inter)")
    plt.savefig(os.path.join(save_dir, "overall_cm_inter.png")); plt.clf()
    print(f"Done inter-subject {src}→{tgt}. Results saved in {save_dir}")
