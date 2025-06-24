# original, smoothing, downsampling, hrv, cwt 가 각각의 채널로 들어가서 총 5개의 채널을 다룸
import os
import glob
import re
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import scipy.signal
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten,
    Conv2D, MaxPooling2D, Dense, Dropout,
    concatenate
)
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------------
# 설정: 결과 저장 경로
# -------------------------------------------------------------------------
RESULT_DIR_BASE = '/home/bcml1/sigenv/_4월/_PPG_5ch/basicCNN'
# intra/inter 최상위 폴더 미리 생성
os.makedirs(os.path.join(RESULT_DIR_BASE, 'intra'), exist_ok=True)
os.makedirs(os.path.join(RESULT_DIR_BASE, 'inter'), exist_ok=True)

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

# -------------------------------------------------------------------------
# 2) 모델 정의 함수 (5-branch multi-input)
# -------------------------------------------------------------------------
def build_multibranch_model(seq_len=128, hrv_len=128, cwt_shape=(32,4,1)):
    inp_o = Input((seq_len,1), name='orig')
    inp_s = Input((seq_len,1), name='smooth')
    inp_d = Input((seq_len,1), name='down')
    inp_h = Input((hrv_len,), name='hrv')
    inp_c = Input(cwt_shape, name='cwt')

    def branch1(inp):
        x = Conv1D(16,3,activation='relu',padding='same')(inp)
        x = MaxPooling1D(2)(x)
        x = Conv1D(32,3,activation='relu',padding='same')(x)
        x = MaxPooling1D(2)(x);
        return Flatten()(x)

    o1 = branch1(inp_o)
    o2 = branch1(inp_s)
    o3 = branch1(inp_d)
    h = Dense(64,activation='relu')(inp_h)
    h = Dropout(0.3)(h)
    o4 = Dense(32,activation='relu')(h)
    z = Conv2D(8,(3,3),activation='relu',padding='same')(inp_c)
    z = MaxPooling2D((2,2))(z)
    z = Conv2D(16,(3,3),activation='relu',padding='same')(z)
    z = MaxPooling2D((2,2))(z)
    o5 = Flatten()(z)
    m = concatenate([o1,o2,o3,o4,o5])
    fc = Dense(64,activation='relu')(m)
    fc = Dropout(0.5)(fc)
    out = Dense(4, activation='softmax', name='emotion')(fc)
    return Model([inp_o,inp_s,inp_d,inp_h,inp_c], out)

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
        model = build_multibranch_model(
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
        model = build_multibranch_model(
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

    # #체크 start
    # for name, arr in zip(['orig','smooth','down','hrv','cwt'], X):
    #     print(name, arr.shape, arr.dtype)
    # print('y', y.shape, y.dtype)
    # #체크 end

    print("\n--- Intra-Subject CV ---")
    intra_subject_cv(X, y, subjects)

    print("\n--- Inter-Subject LOSO CV ---")
    inter_subject_cv(X, y, subjects)
