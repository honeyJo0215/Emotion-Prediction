import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Multiply, Flatten, Lambda, MultiHeadAttention
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.losses import MeanSquaredError

from tensorflow.keras.utils import register_keras_serializable
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import re
from collections import defaultdict

from sklearn.metrics import accuracy_score
import copy
import os
import numpy as np
from EMCSP_1D_CNN_diff import EEGDataLoader  # 위에 주신 CSP 코드가 이 모듈에 정의되어 있다고 가정
from sklearn.model_selection import train_test_split
#__raw EEG 부분__________________________________
# 설정
RAW_FOLDER   = "/home/bcml1/2025_EMOTION/DEAP_eeg_npy_files"
NUM_SUBJECTS = 20
WINDOW_LEN   = 128  # 예: 600
MODE         = 'inter'      # 혹은 'inter'
LEAVE_OUT    = 11         # inter 모드일 때 제외할 subject 번호
TEST_SIZE    = 0.2
RANDOM_SEED  = 42


# ───────────────────────────────────────────────────────────────
# 설정: 경로 및 하이퍼파라미터
# ───────────────────────────────────────────────────────────────
BASE_DIR       = "/home/bcml1/sigenv/_7월/multimodal_livetime"
EYE_DIR        = os.path.join(BASE_DIR, "features_imp")
FACE_DIR       = os.path.join(BASE_DIR, "face_features")
MOUTH_DIR      = "/home/bcml1/yigeon06/Mouth/feature_1s_segment"
RPPG_DIR       = os.path.join(BASE_DIR, "rppg_features")
APEX_DIR = "/home/bcml1/sigenv/_7월/face_micro/restart1_apex8/apex_scores"  
APEX_SCORE_DIR = "/home/bcml1/sigenv/_7월/face_micro/restart1_apex8/apex_scores"
APEX_WEIGHT    = 0.4   # 최종 trial-level에서 apex 비중

# EEG_DIR        = "/home/bcml1/2025_EMOTION/DEAP_eeg_new_label_CSP"
LABEL_ROOT     = "/home/bcml1/2025_EMOTION/DEAP_5labels"
RESULT_DIR     = "/home/bcml1/sigenv/_7월/multimodal_livetime/multi_attention_eeg_cnn"

SEG_LEN_EYE    = 1
SEG_LEN_FACE   = 1
SEG_LEN_MOUTH  = 1
SEG_LEN_RPPG   = 50
WINDOW_LEN_EEG = 128
TEST_RATIO     = 0.2
BATCH_SIZE     = 16
EPOCHS         = 100
NUM_SUBJ       = 20
NUM_TRIAL      = 40

os.makedirs(RESULT_DIR, exist_ok=True)

@register_keras_serializable(package='custom_layers')
class SplitModality(tf.keras.layers.Layer):
    def __init__(self, idx, **kwargs):
        super().__init__(**kwargs)
        self.idx = idx
    def call(self, inputs):
        return tf.expand_dims(inputs[:, self.idx], -1)
    def get_config(self):
        config = super().get_config()
        config.update({'idx': self.idx})
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class EEGDiffusionDenoiser(tf.keras.layers.Layer):
    """
    간단한 1D diffusion-style 복원 레이어.
    - 학습 중에만: x -> x에 노이즈를 섞어서(noisy_x) 복원하도록 MSE loss를 추가
    - 추론 시에는: 그냥 입력을 깨끗하게 통과시킨 값만 사용
    """
    def __init__(self, noise_max=0.5, **kwargs):
        super().__init__(**kwargs)
        self.noise_max = noise_max
        self.mse = MeanSquaredError()

    def build(self, input_shape):
        feat_dim = input_shape[-1]
        # 아주 가벼운 encoder-decoder
        self.enc1 = Conv1D(64, 3, padding="same", activation="relu")
        self.enc2 = Conv1D(64, 3, padding="same", activation="relu")
        self.dec1 = Conv1D(64, 3, padding="same", activation="relu")
        # 원래 채널 수로 복원
        self.out_conv = Conv1D(feat_dim, 1, padding="same", activation=None)
        super().build(input_shape)

    def call(self, x, training=None):
        if training:
            # 0~1 사이 랜덤 timestep 비슷하게 사용
            b = tf.shape(x)[0]
            t = tf.random.uniform((b, 1, 1), 0.0, 1.0)
            # timestep에 비례해서 노이즈량 늘리기
            noise_scale = t * self.noise_max
            noise = tf.random.normal(tf.shape(x)) * noise_scale
            noisy_x = x + noise
        else:
            # 평가/추론 시에는 그냥 원본을 넣어서 통과
            noisy_x = x

        h = self.enc1(noisy_x)
        h = self.enc2(h)
        h = self.dec1(h)
        clean_hat = self.out_conv(h)

        if training:
            # x를 target으로 복원하도록 loss 추가
            self.add_loss(self.mse(x, clean_hat))

        return clean_hat

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"noise_max": self.noise_max})
        return cfg

# ───────────────────────────────────────────────────────────────
# EEG 세그먼트 로드: (channels, timestamps, filters) -> 윈도우별 (WINDOW_LEN_EEG, channels*filters)
# ───────────────────────────────────────────────────────────────
# def prepare_eeg_segments(eeg_dir, label_root, window_len):
#     import os, re, numpy as np
#     from EMCSP_1D_CNN_diff import EEGDataLoader, EMCSP_EEG_1DCNN_Encoder

#     # 1) raw signals 불러와서 세션 리스트 구성
#     sessions = []
#     for subj in range(1, NUM_SUBJ+1):
#         sid = f"s{subj:02d}"
#         raw_path = os.path.join(eeg_dir, f"{sid}_signals.npy")
#         label_path = os.path.join(label_root, f"subject{subj:02d}.npy")
#         if not os.path.isfile(raw_path) or not os.path.isfile(label_path):
#             continue

#         data = np.load(raw_path)           # shape: (40 trials, 32 channels, T timestamps)
#         labels = np.load(label_path)       # shape: (40,)
#         for ti in range(data.shape[0]):
#             lbl = int(labels[ti])
#             if lbl == 4:                   # Neutral 제외
#                 continue
#             sessions.append({
#                 'subject': sid,
#                 'session': ti+1,
#                 'data':    data[ti],       # (32, T)
#                 'label':   lbl
#             })

#     # 2) 세션 → 윈도우 분할 (직접 구현해서 segment index 확보)
#     raw_trials = []
#     for sess in sessions:
#         subj, trial, X_full, lbl = sess['subject'], sess['session'], sess['data'], sess['label']
#         C, T = X_full.shape
#         for seg_idx, start in enumerate(range(0, T - window_len + 1, window_len)):
#             window = X_full[:, start:start+window_len]
#             raw_trials.append({
#                 'subject': subj,
#                 'session': trial,
#                 'data':    window,
#                 'label':   lbl,
#                 'segment': seg_idx      # ← 여기에 segment index 저장
#             })
#     if not raw_trials:
#         return np.empty((0,window_len,0)), np.array([],dtype=int), []

#     # 3) CSP 계산 & 추출
#     n_ch = raw_trials[0]['data'].shape[0]
#     csp_enc = EMCSP_EEG_1DCNN_Encoder(fs=200,
#                                      window_len=window_len,
#                                      n_channels=n_ch)
#     csp_enc.compute_filters_from_trials(raw_trials)
#     X_raw, y = csp_enc.extract_features_from_trials(raw_trials)

#     # 4) reshape
#     N, nb, wl, nc = X_raw.shape
#     X = X_raw.transpose(0,2,1,3).reshape(N, wl, nb*nc)

#     # 5) 메타 정보 (이제 올바릅니다)
#     meta = [(t['subject'], t['session'], t['segment']) for t in raw_trials]
 
#     return X, np.array(y, dtype=int), meta
def prepare_apex_scores(apex_dir, label_root):
    X, y, meta = [], [], []
    for subj in range(1, NUM_SUBJ+1):
        sid = f"s{subj:02d}"
        label_file = os.path.join(label_root, f"subject{subj:02d}.npy")
        if not os.path.isfile(label_file): 
            continue
        labels = np.load(label_file)
        # 예시: apex_dir/s01_trial01_apex.npy 이런 식으로 있다고 가정
        for trial in range(1, 41):
            label = int(labels[trial-1])
            if label == 4:
                continue
            # 네가 만든 이름 규칙에 맞춰서 바꿔
            fpath = os.path.join(apex_dir, f"{sid}_trial{trial:02d}_apex.npy")
            if not os.path.isfile(fpath):
                continue
            arr = np.load(fpath)  # (1,) 또는 (4,)
            # 1초 세그먼트가 여러 개면 거기에 그대로 복제해서 쓴다
            # 여기서는 일단 1개 세그먼트만 있다고 보고 seg_idx=0 으로
            X.append(arr.reshape(1, -1) if arr.ndim == 1 else arr)  # (1,1) or (1,4)
            y.append(label)
            meta.append((sid, trial, 0))
    if not X:
        return np.empty((0,1)), np.array([],dtype=int), []
    return np.concatenate(X, axis=0), np.array(y,dtype=int), meta

def load_apex_score_dict(apex_dir):
    """
    apex_dir 안에 sXX_trialYY_apex.npy 파일들을 읽어서
    {('s01', 1): (4,), ...} 이런 dict로 돌려줌
    """
    d = {}
    for fname in os.listdir(apex_dir):
        if not fname.endswith("_apex.npy"):
            continue
        m = re.match(r"s(\d{2})_trial(\d{2})_apex\.npy", fname)
        if not m:
            continue
        sid = f"s{int(m.group(1)):02d}"
        trial = int(m.group(2))
        arr = np.load(os.path.join(apex_dir, fname))
        d[(sid, trial)] = arr.astype(np.float32)
    return d

def prepare_eeg_segments(eeg_dir, label_root, window_len):
    """
    RAW_FOLDER 내 sXX_signals.npy(40,32,T)와
    LABEL_ROOT 내 subjectXX.npy(40,)를 읽어
    label!=4 인 trial들만 1초 단위(128샘플)로 분할
    → X: (N,128,32) 배열,
      y: (N,) 레이블,
      meta: [(sid,trial_idx,segment_idx),...]
    """
    X, y, meta = [], [], []
    seg_counts = defaultdict(int)

    # 1) raw signals → sessions
    sessions = []
    for subj in range(1, NUM_SUBJ+1):
        sid = f"s{subj:02d}"
        raw_path   = os.path.join(eeg_dir, f"{sid}_signals.npy")
        label_path = os.path.join(label_root, f"subject{subj:02d}.npy")
        if not os.path.isfile(raw_path) or not os.path.isfile(label_path):
            continue

        data   = np.load(raw_path)    # (40 trials,32 ch,T)
        labels = np.load(label_path)  # (40,)
        for ti in range(data.shape[0]):
            lbl = int(labels[ti])
            if lbl == 4:              # Neutral 제외
                continue
            sessions.append({
                'subject': sid,
                'session': ti+1,
                'data':    data[ti],
                'label':   lbl
            })

    # 2) 세션 → 윈도우 분할(앞 3초 스킵)
    raw_trials = []
    for sess in sessions:
        subj, trial, X_full, lbl = sess['subject'], sess['session'], sess['data'], sess['label']
        C, T = X_full.shape
        for seg_idx, start in enumerate(range(3*window_len, T - window_len + 1, window_len)):
            window = X_full[:, start:start+window_len]
            raw_trials.append({
                'subject': subj,
                'session': trial,
                'data':    window,
                'label':   lbl,
                'segment': seg_idx
            })
            seg_counts[lbl] += 1

    if not raw_trials:
        return np.empty((0,window_len,0)), np.array([],dtype=int), []

    # 3) CSP 계산 & 추출
    n_ch = raw_trials[0]['data'].shape[0]
    from EMCSP_1D_CNN_diff import EMCSP_EEG_1DCNN_Encoder
    csp_enc = EMCSP_EEG_1DCNN_Encoder(fs=200,
                                     window_len=window_len,
                                     n_channels=n_ch)
    csp_enc.compute_filters_from_trials(raw_trials)
    X_raw, y = csp_enc.extract_features_from_trials(raw_trials)

    # 4) reshape
    N, nb, wl, nc = X_raw.shape
    X = X_raw.transpose(0,2,1,3).reshape(N, wl, nb*nc)

    # 5) meta
    meta = [(t['subject'], t['session'], t['segment'])
            for t in raw_trials]

    # ── 디버그: 0~3 레이블별 세그먼트 개수 출력 ──
    print(">> [EEG] loaded segments per label:")
    for lbl in range(4):
        print(f"   label {lbl}: {seg_counts.get(lbl, 0)}")
    print("──────────────────────────────────────")

    return X, np.array(y, dtype=int), meta
# ───────────────────────────────────────────────────────────────
# 기존 모달리티 로드 (변경 없음)
# ───────────────────────────────────────────────────────────────
def prepare_segments(base_dir, label_root, seg_len, is_rppg=False):
    X, y, meta = [], [], []
    for subj in range(1, NUM_SUBJ+1):
        sid = f"s{subj:02d}"
        label_file = os.path.join(label_root, f"subject{subj:02d}.npy")
        if not os.path.isfile(label_file): continue
        labels = np.load(label_file)
        subj_dir = os.path.join(base_dir, sid)
        if not os.path.isdir(subj_dir):
            if 'Mouth' in base_dir:
                subj_dir = base_dir
            else:
                continue
        for fname in sorted(os.listdir(subj_dir)):
            if not fname.endswith('.npy'): continue
            if is_rppg:
                m = re.match(r"trial(\d{2})_csp_rppg_multiclass\.npy", fname)
            elif 'Mouth' in base_dir and sid in fname:
                m = re.match(rf"{sid}_trial(\d{{2}})_features\.npy", fname)
            else:
                m = re.match(rf"{sid}_trial(\d{{2}})\.npy", fname)
            if not m: continue
            trial_idx = int(m.group(1))
            label = int(labels[trial_idx-1])
            if label == 4: continue
            data = np.load(os.path.join(subj_dir, fname))
            length = data.shape[1]
            for i in range(0, length, seg_len):
                j = i + seg_len
                if j > length: break
                seg = data[:, i:j]
                if seg.shape[1] != seg_len: continue
                X.append(seg.T)
                y.append(label)
                meta.append((sid, trial_idx, i//seg_len))
    if not X:
        return np.empty((0, seg_len, 0)), np.array([], dtype=int), []
    return np.stack(X, axis=0), np.array(y, dtype=int), meta

if __name__ == '__main__':
    # 1) Load segments
    X_eye,  y_eye,  meta_eye  = prepare_segments(EYE_DIR,  LABEL_ROOT, SEG_LEN_EYE, False)
    X_face, y_face, meta_face = prepare_segments(FACE_DIR, LABEL_ROOT, SEG_LEN_FACE, False)
    X_mouth,y_mouth,meta_mouth= prepare_segments(MOUTH_DIR,LABEL_ROOT,SEG_LEN_MOUTH, False)
    X_rppg, y_rppg, meta_rppg = prepare_segments(RPPG_DIR, LABEL_ROOT, SEG_LEN_RPPG, True)
    X_eeg,  y_eeg,  meta_eeg  = prepare_eeg_segments(RAW_FOLDER, LABEL_ROOT, WINDOW_LEN_EEG)
    #X_apex, y_apex, meta_apex = prepare_apex_scores(APEX_DIR, LABEL_ROOT)
    # --- Debug: count segments per subject for each modality ---
    def print_subject_counts(name, meta):
        counts = defaultdict(int)
        for subj, trial, start in meta:
            counts[subj] += 1
        print(f">>> {name} segments per subject:")
        for subj in sorted(counts):
            print(f"    {subj}: {counts[subj]}")
        print()

    print_subject_counts("Eye",   meta_eye)
    print_subject_counts("Face",  meta_face)
    print_subject_counts("Mouth", meta_mouth)
    print_subject_counts("rPPG",  meta_rppg)
    print_subject_counts("EEG",   meta_eeg)
    # 1-1) apex dict 읽기
    apex_dict = load_apex_score_dict(APEX_SCORE_DIR)

    # 2) Validate existence
    for name, arr in [('Eye',X_eye),('Face',X_face),('Mouth',X_mouth),('rPPG',X_rppg),('EEG',X_eeg)]:
        if arr.size == 0:
            raise FileNotFoundError(f"{name} features empty")

    # 3) Align common meta
    common = set(meta_eye) & set(meta_face) & set(meta_mouth) & set(meta_rppg) & set(meta_eeg)
    ordered = [m for m in meta_eye if m in common]

    def index_list(meta_list, ordered):
        idx_map = {m:i for i,m in enumerate(meta_list)}
        return [idx_map[m] for m in ordered]

    ix_e  = index_list(meta_eye,   ordered)
    ix_f  = index_list(meta_face,  ordered)
    ix_m  = index_list(meta_mouth, ordered)
    ix_r  = index_list(meta_rppg,  ordered)
    ix_q  = index_list(meta_eeg,   ordered)

    X_eye, X_face, X_mouth, X_rppg, X_eeg, y = (
        X_eye[ix_e], X_face[ix_f], X_mouth[ix_m], X_rppg[ix_r], X_eeg[ix_q], y_eye[ix_e]
    )

    # 3-1) apex를 ordered 길이에 맞게 브로드캐스트
    # ordered: [(sid, trial, seg_1s), ...]
    X_apex = []
    for (sid, trial, seg_1s) in ordered:
        score = apex_dict.get((sid, trial), None)
        if score is None:
            # apex 없으면 균등분포로 넣어줌
            score = np.ones(4, dtype=np.float32) / 4.0
        X_apex.append(score)
    X_apex = np.stack(X_apex, axis=0)  # (N,4)

    # meta도 같이 보존해서 나중에 trial별 집계에 씀
    meta_all = np.array(ordered, dtype=object)
    # ── 디버그 #1: alignment 후 라벨 분포 확인 ──
    unique, counts = np.unique(y, return_counts=True)
    print("Labels after alignment:", dict(zip(unique, counts)))

    # 4) Remap labels
    classes = np.unique(y)
    print("Unique classes:", classes)
    cmap = {c: i for i, c in enumerate(classes)}
    y = np.array([cmap[v] for v in y], dtype=int)
    num_classes = len(classes)

    # 5) Normalize
    def normalize(X):
        flat = X.reshape(-1, X.shape[-1])
        mean = np.nanmean(flat, axis=0)
        std = np.nanstd(flat, axis=0) + 1e-6
        return (X - mean) / std

    X_eye   = normalize(X_eye)
    X_face  = normalize(X_face)
    X_mouth = normalize(X_mouth)
    X_rppg  = normalize(X_rppg)
    X_eeg   = normalize(X_eeg)

    # 6) Replace NaN/Inf in EEG (and other modalities) with zero, no segment drop
    X_eeg   = np.nan_to_num(X_eeg,   nan=0.0, posinf=0.0, neginf=0.0)
    X_eye   = np.nan_to_num(X_eye,   nan=0.0, posinf=0.0, neginf=0.0)
    X_face  = np.nan_to_num(X_face,  nan=0.0, posinf=0.0, neginf=0.0)
    X_mouth = np.nan_to_num(X_mouth, nan=0.0, posinf=0.0, neginf=0.0)
    X_rppg  = np.nan_to_num(X_rppg,  nan=0.0, posinf=0.0, neginf=0.0)
      # y는 변함 없음
    
    # 7) Train/test split
    splits = train_test_split(
        X_eye, X_face, X_mouth, X_rppg, X_eeg, X_apex, meta_all, y,
        test_size=TEST_RATIO, random_state=42, stratify=y
    )
    (Xe_tr, Xe_te,
    Xf_tr, Xf_te,
    Xm_tr, Xm_te,
    Xr_tr, Xr_te,
    Xq_tr, Xq_te,
    Xa_tr, Xa_te,
    meta_tr, meta_te,
    y_tr, y_te) = splits


    # ── 디버그 #2: train/test 분포 확인 ──
    ut, ct = np.unique(y_tr, return_counts=True)
    print("Train labels:", dict(zip(ut, ct)))
    ut, ct = np.unique(y_te, return_counts=True)
    print(" Test labels:", dict(zip(ut, ct)))
    single_accs = {}
    from tensorflow.keras.layers import Input, Flatten, Dense
    from tensorflow.keras.models import Model

    def train_single_modal(X_tr, X_te, name):
        inp = Input(shape=X_tr.shape[1:], name=f"{name}_in")          
        x   = Flatten()(inp)
        x   = Dense(128, activation='relu')(x)
        out = Dense(num_classes, activation='softmax')(x)
        m   = Model(inp, out, name=f"{name}_single")
        m.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
        # 10 epochs 정도면 충분합니다
        m.fit(X_tr, y_tr,
            validation_data=(X_te, y_te),
            epochs=10,
            batch_size=BATCH_SIZE,
            verbose=2)
        _, acc = m.evaluate(X_te, y_te, verbose=0)
        print(f"{name} single accuracy = {acc:.4f}")
        return acc

    # ─── 여기서 각 모달리티별 단일-모달 정확도 계산 ───
    single_accs['eye']   = train_single_modal(Xe_tr, Xe_te, 'eye')
    single_accs['face']  = train_single_modal(Xf_tr, Xf_te, 'face')
    single_accs['mouth'] = train_single_modal(Xm_tr, Xm_te, 'mouth')
    single_accs['rppg']  = train_single_modal(Xr_tr, Xr_te, 'rppg')
    single_accs['eeg']   = train_single_modal(Xq_tr, Xq_te, 'eeg')

    # ─── 정확도를 합 1로 정규화해 가중치로 변환 ───
    import tensorflow.keras.backend as K
    weights = np.array([single_accs[m] for m in ['eye','face','mouth','rppg','eeg']])
    weights = weights / weights.sum()           # shape (5,)
    print("Gating weights:", weights)
    # 배치×모달×1 형태 텐서로 만들어 둡니다
    weight_tensor = K.constant(weights.reshape(1,5,1), dtype=tf.float32)

    # 8) Load extractors
    from Eye_model   import build_eye_model
    from Face_model  import build_face_model
    from Mouth_model import build_mouth_model
    # [EEG CNN extractor]
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, BatchNormalization, Flatten
    def build_eeg_cnn(seq_len, feat_dim, embed_dim):
        inp = Input((seq_len, feat_dim), name='eeg_cnn_in')
        x = Conv1D(64, 7, activation='relu', padding='same')(inp)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(embed_dim, activation='relu')(x)
        return Model(inp, x, name='eeg_extractor')

    eye_base   = build_eye_model(num_classes, SEG_LEN_EYE, Xe_tr.shape[2])
    eye_w_path = os.path.join(BASE_DIR,'eye_model','best_eye.weights.h5')
    if os.path.exists(eye_w_path):
        # weights-only 파일이니까 by_name 쓰지 말고 그냥 로드
        eye_base.load_weights(eye_w_path)
        print("[load] eye weights loaded from", eye_w_path)
    else:
        print("[warn] eye weights not found:", eye_w_path)

    face_base  = build_face_model(num_classes, SEG_LEN_FACE, Xf_tr.shape[2])
    face_w_path = os.path.join(BASE_DIR,'face_model','best_face.weights.h5')
    if os.path.exists(face_w_path):
        face_base.load_weights(face_w_path)
        print("[load] face weights loaded from", face_w_path)
    else:
        print("[warn] face weights not found:", face_w_path)

    mouth_base = build_mouth_model(num_classes, SEG_LEN_MOUTH, Xm_tr.shape[2])

    from tensorflow.keras import backend as K
    from tensorflow.keras.models import Model

    # eye / face / mouth base 모델을 이미 build 하고,
    # 가중치도 load 한 '바로 다음'에 넣으세요.
    # -------------------------------------------------
    # 1) eye 모델에서 임베딩 차원 얻기
    eye_cls_layer = eye_base.layers[-1]             # 마지막 Dense(num_classes)
    eye_cls_in_shape = K.int_shape(eye_cls_layer.input)  # 예: (None, 128)
    embed_dim = eye_cls_in_shape[-1]
    print("[info] embed_dim from eye_base =", embed_dim)

    # 2) 분류기 '직전'까지 뽑는 공통 함수
    def make_feature_extractor(base_model):
        cls_layer = base_model.layers[-1]
        feat_tensor = cls_layer.input      # Dense 들어가기 직전 텐서
        return Model(base_model.input, feat_tensor)

    eye_ext   = make_feature_extractor(eye_base)
    face_ext  = make_feature_extractor(face_base)
    mouth_ext = make_feature_extractor(mouth_base)
    # -------------------------------------------------

    eeg_base  = build_eeg_cnn(WINDOW_LEN_EEG, Xq_tr.shape[2], embed_dim)
    eeg_ext   = Model(eeg_base.input,   eeg_base.output)
    for ext in (eye_ext, face_ext, mouth_ext, eeg_ext): ext.trainable=True
    
    from tensorflow.keras.layers import MultiHeadAttention, Lambda
    # --- 하이퍼파라미터 ---
    pdim          = embed_dim   # 기존에 정의하신 embedding 차원
    num_classes   = num_classes
    SEG_LEN_EEG   = WINDOW_LEN_EEG
    EEG_FEAT_DIM  = Xq_tr.shape[2]  # 예: 4 bands × 8 components = 32

    # --- rPPG 브랜치 (변경 없음) ---
    in_r = Input((SEG_LEN_RPPG, Xr_tr.shape[2]), name='rppg_input')
    pr   = Flatten(name='rppg_flat')(in_r)
    feat_r = Dense(pdim, activation='relu', name='rppg_proj')(pr)

    # === 15segment 브랜치 추가===
    # --- apex 입력 (4차원 softmax) ---
    in_apex = Input((4,), name='apex_input')
    # --- 기존 모달리티들 (eye_ext, face_ext, mouth_ext 는 freeze된 extractor) ---
    in_e = Input((SEG_LEN_EYE, Xe_tr.shape[2]),  name='eye_input')
    in_f = Input((SEG_LEN_FACE, Xf_tr.shape[2]), name='face_input')
    in_m = Input((SEG_LEN_MOUTH, Xm_tr.shape[2]),name='mouth_input')
    fe  = eye_ext(in_e)    # (batch, pdim)
    ff  = face_ext(in_f)   # (batch, pdim)
    fm  = mouth_ext(in_m)  # (batch, pdim)

    # --- EEG 브랜치 (raw→CSP→flatten→Dense) ---
    # in_q = Input((SEG_LEN_EEG, EEG_FEAT_DIM), name='eeg_input')
    # qf   = Flatten(name='eeg_flat')(in_q)
    # feat_q = Dense(pdim, activation='relu', name='eeg_proj')(qf)  # (batch, pdim)
    # --- EEG 브랜치 (raw→diffusion denoise→flatten→Dense) ---
    in_q = Input((SEG_LEN_EEG, EEG_FEAT_DIM), name='eeg_input')

    # 1) diffusion-style denoise
    clean_q = EEGDiffusionDenoiser(name="eeg_diff_denoise")(in_q)   # (batch, len, feat)

    # 2) 이후는 원래와 동일
    qf   = Flatten(name='eeg_flat')(clean_q)
    feat_q = Dense(pdim, activation='relu', name='eeg_proj')(qf)    # (batch, pdim)

    # --- 모달리티 스택 & 셀프-어텐션 ---
    modal_stack = Lambda(lambda tensors: tf.stack(tensors, axis=1),
                        name='modal_stack')([fe, ff, fm, feat_r, feat_q])
    
    from tensorflow.keras.layers import Dense, Activation, Multiply, LayerNormalization, GaussianNoise, Dropout
    from tensorflow.keras.layers import MultiHeadAttention, Flatten, Lambda
    # attn_output = MultiHeadAttention(
    #     num_heads=1, key_dim=pdim, name='modal_self_attention'
    # )(
    #     query=modal_stack, value=modal_stack, key=modal_stack
    # )  # (batch, 5, pdim)
    # ─── (B) 단일-모달 정확도 기반 게이팅 적용 ───
    # 1) 정규화 + 노이즈
    x = LayerNormalization(name='modal_norm')(modal_stack)
    x = GaussianNoise(0.1, name='modal_noise')(x)

    # 2) 학습 가능한 게이팅
    g = Flatten(name='gating_flat')(x)                     # (batch, 5*pdim)
    g = Dense(5, name='gating_logits')(g)                  # (batch,5)
    g = Activation('softmax', name='gating_weights')(g)    # (batch,5)

    # ── 여기가 추가되는 부분 ─────────────────────────
    # apex_score: (batch,1) → (batch,1) 에서 0.5~1.5 사이로 늘려서 “세게/약하게”
    def _scale_with_apex_prob(args):
        g, apex_prob = args   # g:(B,5), apex_prob:(B,4)
        # neutral을 마지막 인덱스라고 가정 (네 4클래스랑 맞춰서)
        neutral_p = apex_prob[:, -1:]            # (B,1)
        energy = 1.0 - neutral_p                 # (B,1) → 표정변화 크기
        scale = 0.5 + energy * 1.0               # 0.5 ~ 1.5
        g = g * scale
        g = g / tf.reduce_sum(g, axis=1, keepdims=True)
        return g

    g = Lambda(_scale_with_apex_prob, name="gating_apex_scale")([g, in_apex])
    # ────────────────────────────────────────────────

    g = Lambda(lambda w: tf.expand_dims(w, -1), name='gating_expand')(g)  # (batch,5,1)
    gated = Multiply(name='gated_stack')([x, g])           # (batch,5,pdim)
            # (batch,5,pdim)

    # 3) Modality Dropout (batch × 5 × pdim) → 확률적으로 모달 하나씩 끄기
    
    def mod_dropout(x, p=0.3):
        shape = tf.shape(x)
        batch_size = shape[0]
        num_modal  = shape[1]
        # (batch, num_modal, 1) 으로 mask 생성
        mask = tf.cast(
            tf.random.uniform((batch_size, num_modal, 1)) > p,
            x.dtype
        )
        return x * mask

    # Lambda 부분도 아래처럼 교체
    gated = Lambda(lambda t: mod_dropout(t, p=0.3),
                name='modality_dropout')(gated)
    # 4) Multi-Head Attention (2 heads, key_dim = pdim//2)
    attn_out = MultiHeadAttention(
        num_heads=2,
        key_dim=pdim//2,
        name='modal_self_attention'
    )(
        query=gated, value=gated, key=gated
    )  # (batch,5,pdim)

    # 5) 풀링 → 분류기 헤드 (이전과 동일)
    fused = Lambda(lambda x: tf.reduce_mean(x, axis=1), name='fused')(attn_out)
    x = Dense(128, activation='relu', name='fusion_dense')(fused)
    out = Dense(num_classes, activation='softmax', name='out')(x)
    # --- 모델 컴파일 ---
    model = Model(inputs=[in_e, in_f, in_m, in_r, in_q, in_apex], outputs=out,
                name='attn_multimodal_eeg_rawcsp')
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    model.summary()
    # 9) Train
    cp = ModelCheckpoint(
        os.path.join(RESULT_DIR, 'best_model.weights.h5'),
        save_best_only=True,
        monitor='val_accuracy',
        save_weights_only=True   # ← 전체 모델이 아니라 가중치만 저장
    )

    hist = model.fit(
        [Xe_tr, Xf_tr, Xm_tr, Xr_tr, Xq_tr, Xa_tr],
        y_tr,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[cp],
        verbose=2
    )

    model.save(os.path.join(RESULT_DIR, 'final_model.keras'))


    # 10) Evaluate & Save
    # 1) 세그먼트 별 확률
    seg_probs = model.predict(
        [Xe_te, Xf_te, Xm_te, Xr_te, Xq_te, Xa_te],
        verbose=0
    )   # (Nseg, 4)

    # 2) trial 별로 묶기
    from collections import defaultdict
    trial_to_indices = defaultdict(list)
    for i, m in enumerate(meta_te):
        sid, trial, seg = m
        trial_to_indices[(sid, int(trial))].append(i)

    rows = []
    for (sid, trial), idxs in trial_to_indices.items():
        # 1초 세그먼트 평균
        seg_mean = seg_probs[idxs].mean(axis=0)     # (4,)

        # apex 원본 점수
        apex_prob = apex_dict.get((sid, trial), np.ones(4, dtype=np.float32)/4.0)
        apex_prob = apex_prob / apex_prob.sum()

        # 최종 결합
        final_prob = (1 - APEX_WEIGHT) * seg_mean + APEX_WEIGHT * apex_prob
        final_label = int(final_prob.argmax())
        apex_label  = int(apex_prob.argmax())
        mm_label    = int(seg_mean.argmax())

        rows.append({
            "subject": sid,
            "trial": trial,
            "final_label": final_label,
            "final_prob": final_prob.tolist(),
            "apex_label": apex_label,
            "apex_prob": apex_prob.tolist(),
            "mm_label": mm_label,
            "mm_prob": seg_mean.tolist(),
            "num_segments": len(idxs),
        })

    # 3) 저장
    import json, csv
    save_json = os.path.join(RESULT_DIR, "trial_level_predictions.json")
    with open(save_json, "w") as f:
        json.dump(rows, f, indent=2)

    save_csv = os.path.join(RESULT_DIR, "trial_level_predictions.csv")
    with open(save_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "subject","trial","final_label","final_prob",
            "apex_label","apex_prob","mm_label","mm_prob","num_segments"
        ])
        for r in rows:
            writer.writerow([
                r["subject"], r["trial"], r["final_label"],
                r["final_prob"],
                r["apex_label"], r["apex_prob"],
                r["mm_label"], r["mm_prob"],
                r["num_segments"]
            ])
    print("[save] trial-level predictions ->", save_json)

    #loss, acc = model.evaluate([Xe_te, Xf_te, Xm_te, Xr_te, Xq_te], y_te, verbose=2)
    loss, acc = model.evaluate([Xe_te, Xf_te, Xm_te, Xr_te, Xq_te, Xa_te], y_te, verbose=2)
    print(f"Test loss={loss:.4f}, acc={acc:.4f}")
    
    #pred = model.predict([Xe_te, Xf_te, Xm_te, Xr_te, Xq_te]).argmax(1)
    pred = model.predict([Xe_te, Xf_te, Xm_te, Xr_te, Xq_te, Xa_te]).argmax(1)
    rep  = classification_report(y_te, pred)
    cm   = confusion_matrix(y_te, pred)
    with open(os.path.join(RESULT_DIR,'report.txt'),'w') as f: f.write(rep)
    np.savetxt(os.path.join(RESULT_DIR,'cm.txt'), cm, fmt='%d')
    plt.figure();plt.plot(hist.history['loss'],label='loss');plt.plot(hist.history['val_loss'],label='val_loss');plt.legend();plt.savefig(os.path.join(RESULT_DIR,'loss.png'))
    plt.figure();plt.plot(hist.history['accuracy'],label='acc');plt.plot(hist.history['val_accuracy'],label='val_acc');plt.legend();plt.savefig(os.path.join(RESULT_DIR,'acc.png'))

        
    # 1) 원본 정확도 계산
    y_pred_orig = model.predict([Xe_te, Xf_te, Xm_te, Xr_te, Xq_te, Xa_te], verbose=0).argmax(axis=1)
    base_acc = accuracy_score(y_te, y_pred_orig)

    print(f"Base accuracy: {base_acc:.4f}")
    
    # 2) 퍼뮤테이션 중요도 함수 정의
    def permutation_importance(model, X_list, X_apex_fixed, y_true, idx):
        X_list_perm = [x.copy() for x in X_list]
        # idx번째 모달리티만 배치 축 셔플
        np.random.shuffle(X_list_perm[idx])
        # 모델은 6개 입력을 받으니까 마지막에 apex 고정으로 붙여줌
        y_pred = model.predict(X_list_perm + [X_apex_fixed], verbose=0).argmax(axis=1)
        return base_acc - accuracy_score(y_true, y_pred)

    modal_names = ['eye', 'face', 'mouth', 'rppg', 'eeg']
    X_list = [Xe_te, Xf_te, Xm_te, Xr_te, Xq_te]

    for i, name in enumerate(modal_names):
        delta = permutation_importance(model, X_list, Xa_te, y_te, i)
        print(f"{name:6} importance drop: {delta:.4f}")

