import os
import re
import gc
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten,
    Conv2D, MaxPooling2D, Dense, Dropout,
    concatenate
)
import glob
import scipy.signal
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model

# -------------------------------------------------------------------------
# 설정: 결과 저장 경로
# -------------------------------------------------------------------------
RESULT_DIR_BASE = '/home/bcml1/sigenv/_4월/_PPG_5ch/EMCNN+diffuSSM'

# -------------------------------------------------------------------------
# GPU 메모리 제한 (필요시)
# -------------------------------------------------------------------------
def limit_gpu_memory(memory_limit_mib=8000):
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

limit_gpu_memory(5000)

# -------------------------------------------------------------------------
# Feature computation
# -------------------------------------------------------------------------
def compute_hrv(ppg: np.ndarray, fs: int = 128) -> np.ndarray:
    """
    PPG 1차원 시계열(ppg, 길이 N)에서 피크를 검출하고,
    RR 간격 → 순간 심박수(단위: bpm) → 원본 길이 N으로 보간한 벡터 반환.
    """
    # 1) 피크 검출 (최소 0.4초 간격 유지)
    peaks, _ = scipy.signal.find_peaks(ppg, distance=int(fs*0.4))
    if len(peaks) < 2:
        # 피크가 충분치 않으면 0으로 채움
        return np.zeros_like(ppg, dtype=np.float32)

    # 2) RR 간격(초) → 순간 심박수(bpm)
    rr_intervals = np.diff(peaks) / fs
    hr_inst = 60.0 / rr_intervals  # (len(peaks)-1,)

    # 3) 보간: 피크 위치(peaks[1:])에 대응하는 hr_inst를
    #    전체 시점(np.arange(N))으로 선형 보간
    times = peaks[1:]
    hr_series = np.interp(
        np.arange(len(ppg)),  # 보간할 시점
        times,                # x-coordinates of data points
        hr_inst,              # y-coordinates (심박수)
        left=hr_inst[0],
        right=hr_inst[-1]
    )
    return hr_series.astype(np.float32)


def compute_cwt(ppg: np.ndarray, cwt_shape=(32,4)) -> np.ndarray:
    """
    PPG 1차원 시계열(ppg, 길이 N)에 대해
    Ricker(wavelet) CWT를 수행하고 (32, N) 매트릭스 획득 → 
    시간축을 4개 구간으로 나누어 평균 → (32,4) → 채널축 추가 → (32,4,1)
    """
    n_scales, n_bins = cwt_shape
    # 1) 스케일 리스트: 1부터 n_scales까지
    widths = np.arange(1, n_scales+1)
    # 2) CWT 수행 → shape (n_scales, N)
    cwt_mat = scipy.signal.cwt(ppg, scipy.signal.ricker, widths)
    # 3) 시간축(길이 N)을 n_bins 구간으로 나누어 평균
    N = cwt_mat.shape[1]
    assert N % n_bins == 0, "N이 n_bins로 나누어 떨어져야 합니다."
    segment_len = N // n_bins
    # reshape → (n_scales, n_bins, segment_len)
    cwt_reshaped = cwt_mat.reshape(n_scales, n_bins, segment_len)
    # 구간별 평균 → (n_scales, n_bins)
    cwt_down = cwt_reshaped.mean(axis=2)
    # 채널축 추가 → (n_scales, n_bins, 1)
    return cwt_down[..., np.newaxis].astype(np.float32)

# -------------------------------------------------------------------------
# 1) 데이터 로딩 함수
# -------------------------------------------------------------------------
def load_all_data(data_path: str,
                  smooth_idx: int = 1,
                  down_idx: int = 3,
                  cwt_shape=(32,4),
                  fs: int = 128):
    """
    - data_path 내 모든 .npy 파일 로드
    - 각 세그먼트당 5개 채널(orig, smooth, down, hrv, cwt) 생성
    - HRV 채널만 StandardScaler로 표준화
    - 반환: X = [orig, smo, dwn, hrv, cwt], y, subjects
    """
    files = glob.glob(os.path.join(data_path, '*.npy'))
    print(f"Found {len(files)} files in {data_path}")

    Xo, Xs, Xd, Xh, Xc = [], [], [], [], []
    y_list, subj_list = [], []
    pattern = re.compile(r's(\d{2})_trial_(\d{2})_label_(\d)\.npy')

    for fp in files:
        fn = os.path.basename(fp)
        m = pattern.match(fn)
        if not m:
            print(f"  skip (pattern mismatch): {fn}")
            continue

        subj  = int(m.group(1))
        label = int(m.group(3))
        data  = np.load(fp)  # (n_seg, 5, 128)

        for seg in data:
            raw = seg[0]            # original PPG (128,)
            smo = seg[smooth_idx]   # smoothing PPG
            dwn = seg[down_idx]     # downsampling PPG

            # 1) PPG 채널 reshape
            Xo.append(raw.reshape(128,1))
            Xs.append(smo.reshape(128,1))
            Xd.append(dwn.reshape(128,1))

            # 2) 원본으로부터 HRV/CWT 계산
            hrv = compute_hrv(raw, fs=fs)                     # (128,)
            cwt = compute_cwt(raw, cwt_shape=cwt_shape)        # (32,4,1)

            Xh.append(hrv)
            Xc.append(cwt)

            y_list.append(label)
            subj_list.append(subj)

    # numpy 변환 & dtype 통일
    Xo = np.stack(Xo, axis=0).astype(np.float32)  # (샘플,128,1)
    Xs = np.stack(Xs, axis=0).astype(np.float32)
    Xd = np.stack(Xd, axis=0).astype(np.float32)
    Xh = np.stack(Xh, axis=0).astype(np.float32)  # (샘플,128)
    Xc = np.stack(Xc, axis=0).astype(np.float32)  # (샘플,32,4,1)

    y        = np.array(y_list,    dtype=np.int32)
    subjects = np.array(subj_list, dtype=np.int32)

    # HRV만 추가 표준화
    scaler = StandardScaler()
    Xh = scaler.fit_transform(Xh).astype(np.float32)

    X = [Xo, Xs, Xd, Xh, Xc]
    print("Shapes:", [arr.shape for arr in X], y.shape, subjects.shape)
    return X, y, subjects

# 2) 모델 정의 (변경 없음)
class BidirectionalSSMLayer(layers.Layer):
    def __init__(self, units, kernel_size=3, **kwargs):
        """
        units: 입력/출력 차원 (== 피처 개수)
        kernel_size: causal conv의 커널 크기
        """
        super().__init__(**kwargs)
        self.units = units
        self.kernel_size = kernel_size
        # 1) 순방향 필터
        self.fwd_conv = layers.Conv1D(
            filters=1,
            kernel_size=kernel_size,
            padding='causal',
            activation=None,
            name="ssm_fwd_conv"
        )
        # 2) 역방향 필터
        self.bwd_conv = layers.Conv1D(
            filters=1,
            kernel_size=kernel_size,
            padding='causal',
            activation=None,
            name="ssm_bwd_conv"
        )

    def call(self, x, training=False):
        # x: (batch, units)
        # 1) 시퀀스 차원 추가 → (batch, units, 1)
        seq = tf.expand_dims(x, axis=-1)

        # 2) 순방향 causal convolution
        fwd = self.fwd_conv(seq)  # (batch, units, 1)

        # 3) 역방향 causal convolution
        rev_seq = tf.reverse(seq, axis=[1])
        bwd = self.bwd_conv(rev_seq)
        bwd = tf.reverse(bwd, axis=[1])  # (batch, units, 1)

        # 4) 합치고 마지막 차원 제거 → (batch, units)
        out = fwd + bwd
        return tf.squeeze(out, axis=-1)

# =============================================================================
# DiffuSSMLayer
class DiffuSSMLayer(layers.Layer):
    def __init__(self,
                 model_dim,           # 입력/출력 차원
                 hidden_dim,          # bottleneck 차원
                 cond_dim=None,       # conditioning 입력 차원 (optional)
                 alpha_init=0.0,      # fusion 스케일 초기값
                 **kwargs):
        super().__init__(**kwargs)
        # 1) Layer Norm + Conditioning MLP → γ, β 생성
        self.norm = layers.LayerNormalization()
        if cond_dim is not None:
            self.cond_mlp = layers.Dense(2 * model_dim, name="cond_mlp")
        else:
            self.cond_mlp = None

        # 2) Hourglass Dense (차원 축소 → 확장)
        self.hg_down = layers.Dense(hidden_dim, activation='swish', name="hourglass_down")
        # 3) Bidirectional SSM (여기에 실제 SSM 블록을 넣으세요)
        self.ssm = BidirectionalSSMLayer(hidden_dim, kernel_size=3)    # → (batch, hidden_dim)
        # 4) Hourglass 복원
        self.hg_up = layers.Dense(model_dim, activation=None, name="hourglass_up")

        # 5) fusion 스케일 α
        self.alpha = self.add_weight(
            name="fusion_scale_alpha",
            shape=(),
            initializer=tf.keras.initializers.Constant(alpha_init),
            trainable=True)

    def call(self, x, cond=None, training=False):
        # --- 1) Layer norm + (optional) conditioning scale&shift ---
        x_ln = self.norm(x)
        if self.cond_mlp is not None and cond is not None:
            # cond → [γ, β]
            gamma, beta = tf.split(self.cond_mlp(cond), num_or_size_splits=2, axis=-1)
            x_ln = gamma * x_ln + beta

        # --- 2) Hourglass down ---
        h = self.hg_down(x_ln)

        # --- 3) Bidirectional SSM 처리 ---
        h = self.ssm(h)  

        # --- 4) Hourglass up ---
        h = self.hg_up(h)

        # --- 5) Residual with learnable scale α ---
        return x + self.alpha * h

# =============================================================================
# 가우시안 노이즈 추가 함수
def add_diffusion_noise(x, stddev=0.05):
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=stddev)
    return x + noise

# =============================================================================
# Frequency branch CNN block with pre- and post-diffusion
# 입력 shape: (200,8,4)
# =============================================================================
# Frequency branch CNN block with pre- and post-diffusion
def build_freq_branch_with_diffusion(input_shape=(200,8,4), noise_std=0.05):
    inp = layers.Input(shape=input_shape)

    # ----- Pre-CNN block -----
    pre_processed = layers.Lambda(lambda x: add_diffusion_noise(x, stddev=noise_std))(inp)

    # ----- CNN block -----
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(pre_processed)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)  # -> (64,)
    freq_feat = x

    # ----- Post-diffusion block -----
    noisy_freq_feat = layers.Lambda(lambda x: add_diffusion_noise(x, stddev=noise_std))(freq_feat)

    # 여기서 model_dim=64, hidden_dim=64 을 명시적으로 전달
    diffu_freq = DiffuSSMLayer(model_dim=64, hidden_dim=64)(noisy_freq_feat)

    freq_res = layers.Add()([freq_feat, diffu_freq])

    return models.Model(inputs=inp, outputs=freq_res, name="FreqBranchDiff")


# =============================================================================
# Channel branch CNN block with pre- and post-diffusion
def build_chan_branch_with_diffusion(input_shape=(200,4,8), noise_std=0.05):
    inp = layers.Input(shape=input_shape)

    # ----- Pre-CNN block -----
    pre_processed = layers.Lambda(lambda x: add_diffusion_noise(x, stddev=noise_std))(inp)

    # ----- CNN block -----
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(pre_processed)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)  # -> (64,)
    chan_feat = x

    # ----- Post-diffusion block -----
    noisy_chan_feat = layers.Lambda(lambda x: add_diffusion_noise(x, stddev=noise_std))(chan_feat)

    # model_dim=64, hidden_dim=64
    diffu_chan = DiffuSSMLayer(model_dim=64, hidden_dim=64)(noisy_chan_feat)

    chan_res = layers.Add()([chan_feat, diffu_chan])

    return models.Model(inputs=inp, outputs=chan_res, name="ChanBranchDiff")

def build_emcnn_with_diffusion(
    seq_len=128,
    hrv_len=128,
    cwt_shape=(32,4,1),
    num_classes=4,
    diffu_hidden=64
):
    """
    EMCNN + DiffuSSM 모델 빌더
    - 1D-CNN 브랜치(orig, smooth, down) + HRV, CWT 브랜치
    - 각 브랜치 끝에 DiffuSSMLayer 적용
    """
    # 입력 정의
    inp_o = Input((seq_len,1), name="orig")
    inp_s = Input((seq_len,1), name="smooth")
    inp_d = Input((seq_len,1), name="down")
    inp_h = Input((hrv_len,),  name="hrv")
    inp_c = Input(cwt_shape,   name="cwt")

    # 1D-CNN + DiffuSSM 브랜치 함수
    def ppg_branch(inp, prefix):
        x = Conv1D(16,3,activation='relu',padding='same', name=f"{prefix}_conv1")(inp)
        x = MaxPooling1D(2, name=f"{prefix}_pool1")(x)
        x = Conv1D(32,3,activation='relu',padding='same', name=f"{prefix}_conv2")(x)
        x = MaxPooling1D(2, name=f"{prefix}_pool2")(x)
        flat = Flatten(name=f"{prefix}_flat")(x)
        diff = DiffuSSMLayer(
            model_dim=flat.shape[-1],
            hidden_dim=diffu_hidden,
            name=f"{prefix}_diffu"
        )(flat)
        return diff

    o_feat = ppg_branch(inp_o, "orig")
    s_feat = ppg_branch(inp_s, "smooth")
    d_feat = ppg_branch(inp_d, "down")

    # HRV 브랜치: Dense → DiffuSSM
    h = Dense(64, activation='relu', name="hrv_dense")(inp_h)
    h = Dropout(0.3, name="hrv_drop")(h)
    h_feat = DiffuSSMLayer(
        model_dim=h.shape[-1],
        hidden_dim=diffu_hidden,
        name="hrv_diffu"
    )(h)

    # CWT 브랜치: 2D-CNN → Flatten → DiffuSSM
    z = Conv2D(8,(3,3),activation='relu',padding='same', name="cwt_conv1")(inp_c)
    z = MaxPooling2D((2,2), name="cwt_pool1")(z)
    z = Conv2D(16,(3,3),activation='relu',padding='same', name="cwt_conv2")(z)
    z = MaxPooling2D((2,2), name="cwt_pool2")(z)
    z_flat = Flatten(name="cwt_flat")(z)
    c_feat = DiffuSSMLayer(
        model_dim=z_flat.shape[-1],
        hidden_dim=diffu_hidden,
        name="cwt_diffu"
    )(z_flat)

    # 특징 결합 및 분류기
    merged = concatenate(
        [o_feat, s_feat, d_feat, h_feat, c_feat],
        name="merged_feats"
    )
    x = Dense(64, activation='relu', name="fc1")(merged)
    x = Dropout(0.5, name="drop1")(x)
    x = Dense(32, activation='relu', name="fc2")(x)
    x = Dropout(0.3, name="drop2")(x)
    out = Dense(num_classes, activation='softmax', name="emotion")(x)

    return Model(
        inputs=[inp_o, inp_s, inp_d, inp_h, inp_c],
        outputs=out,
        name="EMCNN_with_DiffuSSM"
    )
    
# -------------------------------------------------------------------------
# 3) Intra-subject CV
# -------------------------------------------------------------------------
def intra_subject_cv(X, y, subjects):
    """
    Intra-subject CV with explicit validation split.
    For each subject s:
      1) 15% test split
      2) 남은 85% 중에서 15%를 val split
      3) train_ds / val_ds 생성
      4) model.fit(..., validation_data=val_ds)
      5) test set으로 최종 평가 및 결과 저장
    """
    results = {}
    for s in np.unique(subjects):
        # 1) 해당 subject 데이터 분리
        mask = (subjects == s)
        Xo_sub, Xs_sub, Xd_sub, Xh_sub, Xc_sub = [arr[mask] for arr in X]
        y_sub = y[mask]

        # 2) 15%를 test로 분리
        Xo_trval, Xo_te, Xs_trval, Xs_te, Xd_trval, Xd_te, \
        Xh_trval, Xh_te, Xc_trval, Xc_te, y_trval, y_te = train_test_split(
            Xo_sub, Xs_sub, Xd_sub, Xh_sub, Xc_sub, y_sub,
            test_size=0.15, stratify=y_sub, random_state=42
        )

        # 3) 나머지 85% 중에서 15/85 ≈ 0.1765를 val로 분리
        val_frac = 0.1765
        Xo_tr, Xo_val, Xs_tr, Xs_val, Xd_tr, Xd_val, \
        Xh_tr, Xh_val, Xc_tr, Xc_val, y_tr, y_val = train_test_split(
            Xo_trval, Xs_trval, Xd_trval, Xh_trval, Xc_trval, y_trval,
            test_size=val_frac, stratify=y_trval, random_state=42
        )

        # 4) tf.data.Dataset 생성
        train_ds = tf.data.Dataset.from_tensor_slices((
            (Xo_tr, Xs_tr, Xd_tr, Xh_tr, Xc_tr),
            y_tr
        )).shuffle(1000).batch(64).prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((
            (Xo_val, Xs_val, Xd_val, Xh_val, Xc_val),
            y_val
        )).batch(64).prefetch(tf.data.AUTOTUNE)

        # 5) 모델 생성·컴파일
        model = build_emcnn_with_diffusion(
            seq_len=Xo_tr.shape[1],
            hrv_len=Xh_tr.shape[1],
            cwt_shape=Xc_tr.shape[1:]
        )
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # 6) 학습 (validation_data 지정)
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=2000,
            verbose=1
        )

        # 7) 테스트셋으로 최종 평가
        loss, acc = model.evaluate(
            [Xo_te, Xs_te, Xd_te, Xh_te, Xc_te],
            y_te,
            verbose=0
        )
        results[s] = (loss, acc)
        print(f"[Intra w/ Val] Subject {s:02d} Test Acc={acc:.4f}")

        # 8) 예측 및 리포트
        y_pred = np.argmax(
            model.predict([Xo_te, Xs_te, Xd_te, Xh_te, Xc_te], batch_size=64),
            axis=1
        )
        report = classification_report(y_te, y_pred)

        # 9) 결과 저장 디렉터리
        subj_dir = os.path.join(RESULT_DIR_BASE, 'intra', f"subject_{s:02d}")
        os.makedirs(subj_dir, exist_ok=True)

        # metrics.txt
        with open(os.path.join(subj_dir, 'metrics.txt'), 'w') as f:
            f.write(f"loss: {loss:.4f}, accuracy: {acc:.4f}\n")

        # classification_report.txt
        with open(os.path.join(subj_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)

        # confusion_matrix.png
        cm = confusion_matrix(y_te, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Subject {s:02d} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(subj_dir, 'confusion_matrix.png'))
        plt.close()

        # loss_curve.png
        plt.figure()
        plt.plot(history.history['loss'],    label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(subj_dir, 'loss_curve.png'))
        plt.close()

        # accuracy_curve.png
        plt.figure()
        plt.plot(history.history['accuracy'],    label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.title("Accuracy Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(subj_dir, 'accuracy_curve.png'))
        plt.close()

    return results

# -------------------------------------------------------------------------
# 4) Inter-subject CV (LOSO)
# -------------------------------------------------------------------------
def inter_subject_cv(X, y, subjects):
    """
    LOSO inter-subject CV with explicit validation split.
    For each held-out subject s:
      1) subjects != s → training pool
      2) subjects == s → final test set
      3) training pool → split 15% val
      4) train_ds / val_ds 생성, model.fit(validation_data=val_ds)
      5) test set으로 최종 평가 및 리포트 저장
    """
    results = {}
    for s in np.unique(subjects):
        # 1) train/test mask
        train_m = (subjects != s)
        test_m  = (subjects == s)

        # 2) pool split
        Xo_pool, Xs_pool, Xd_pool, Xh_pool, Xc_pool = [arr[train_m] for arr in X]
        y_pool = y[train_m]
        Xo_te, Xs_te, Xd_te, Xh_te, Xc_te = [arr[test_m] for arr in X]
        y_te = y[test_m]

        # 3) validation 분리 (pool의 15%)
        Xo_tr, Xo_val, Xs_tr, Xs_val, Xd_tr, Xd_val, \
        Xh_tr, Xh_val, Xc_tr, Xc_val, y_tr, y_val = train_test_split(
            Xo_pool, Xs_pool, Xd_pool, Xh_pool, Xc_pool, y_pool,
            test_size=0.15, stratify=y_pool, random_state=42
        )

        # 4) tf.data.Dataset 생성
        train_ds = tf.data.Dataset.from_tensor_slices((
            (Xo_tr, Xs_tr, Xd_tr, Xh_tr, Xc_tr),
            y_tr
        )).shuffle(1000).batch(64).prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((
            (Xo_val, Xs_val, Xd_val, Xh_val, Xc_val),
            y_val
        )).batch(64).prefetch(tf.data.AUTOTUNE)

        # 5) 모델 생성·컴파일
        model = build_emcnn_with_diffusion(
            seq_len=Xo_tr.shape[1],
            hrv_len=Xh_tr.shape[1],
            cwt_shape=Xc_tr.shape[1:]
        )
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # 6) 학습 (validation_data 지정)
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=200,
            verbose=1
        )

        # 7) 테스트셋 평가
        loss, acc = model.evaluate(
            [Xo_te, Xs_te, Xd_te, Xh_te, Xc_te],
            y_te,
            verbose=0
        )
        results[s] = (loss, acc)
        print(f"[Inter w/ Val] Held-out {s:02d} Test Acc={acc:.4f}")

        # 8) 예측 및 classification report
        y_pred = np.argmax(
            model.predict([Xo_te, Xs_te, Xd_te, Xh_te, Xc_te], batch_size=64),
            axis=1
        )
        report = classification_report(y_te, y_pred)

        # 9) 결과 저장 디렉터리
        subj_dir = os.path.join(RESULT_DIR_BASE, 'inter', f"subject_{s:02d}_LOSO")
        os.makedirs(subj_dir, exist_ok=True)

        # metrics.txt
        with open(os.path.join(subj_dir, 'metrics.txt'), 'w') as f:
            f.write(f"loss: {loss:.4f}, accuracy: {acc:.4f}\n")

        # classification_report.txt
        with open(os.path.join(subj_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)

        # confusion_matrix.png
        cm = confusion_matrix(y_te, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Subject {s:02d} LOSO Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(subj_dir, 'confusion_matrix.png'))
        plt.close()

        # loss_curve.png
        plt.figure()
        plt.plot(history.history['loss'],    label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(subj_dir, 'loss_curve.png'))
        plt.close()

        # accuracy_curve.png
        plt.figure()
        plt.plot(history.history['accuracy'],    label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.title("Accuracy Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(subj_dir, 'accuracy_curve.png'))
        plt.close()

    return results

# -------------------------------------------------------------------------
# 5) 실행
# -------------------------------------------------------------------------
if __name__ == '__main__':
    DATA_DIR = '/home/bcml1/2025_EMOTION/DEAP_PPG_1s'
    X, y, subjects = load_all_data(DATA_DIR)
    print("\n--- Intra-Subject CV ---")
    intra_subject_cv(X, y, subjects)
    print("\n--- Inter-Subject LOSO CV ---")
    inter_subject_cv(X, y, subjects)
