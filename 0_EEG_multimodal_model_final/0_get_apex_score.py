# 0_get_apex_score.py

import os, glob, re, cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv3D, MaxPool3D, Dropout,
    Dense, BatchNormalization
)
from tensorflow.keras.models import Model

# ───────────────────── 경로 ─────────────────────
APEX_SEG_DIR   = "/home/bcml1/sigenv/_7월/face_micro/restart1_apex8/apex_segments_30frame"
LABEL_DIR      = "/home/bcml1/2025_EMOTION/DEAP_5labels"
APEX_SCORE_DIR = "/home/bcml1/sigenv/_7월/face_micro/restart1_apex8/apex_scores"
os.makedirs(APEX_SCORE_DIR, exist_ok=True)

BEST_WEIGHTS   = "/home/bcml1/sigenv/_7월/face_micro/restart1_apex8/3_feature_3DCNN_test1/best_microexp3dstcnn.h5"

# ───────────────────── 1) 30프레임용 3D CNN ─────────────────────
def build_microexp3dstcnn_30(
    input_shape=(30, 64, 64, 1),
    num_classes=4,
    base_filters=16,
    dropout_rate=0.3
):
    inp = Input(input_shape)

    x = Conv3D(base_filters, (3,3,3), padding='same', activation='relu')(inp)
    x = BatchNormalization()(x)
    x = MaxPool3D((1,2,2), padding='same')(x)   # 시간 그대로, 공간만 1/2
    x = Dropout(dropout_rate)(x)

    x = Conv3D(base_filters*2, (3,3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool3D((2,2,2), padding='same')(x)   # 여기서부터 시간도 1/2
    x = Dropout(dropout_rate)(x)

    x = Conv3D(base_filters*4, (3,3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool3D((2,2,2), padding='same')(x)
    x = Dropout(dropout_rate)(x)

    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 모델 만들고 가중치 최대한 불러오기
apex_model = build_microexp3dstcnn_30()
# 150프레임짜리에서 가져오는 거라 이름 맞는 것만 불러옴
apex_model.load_weights(BEST_WEIGHTS, by_name=True, skip_mismatch=True)
print("[apex] weights loaded (by_name, skip_mismatch) from:", BEST_WEIGHTS)

# ───────────────────── 2) 유틸 ─────────────────────
def get_label_sid_tid(fpath):
    base = os.path.basename(fpath)
    m = re.match(r"s(\d{1,2})_trial(\d{1,2})\.npz", base)
    if not m:
        raise ValueError(f"filename not matched: {base}")
    sid = int(m.group(1))
    tid = int(m.group(2))
    label_arr = np.load(os.path.join(LABEL_DIR, f"subject{sid:02d}.npy"))
    return int(label_arr[tid-1]), f"s{sid:02d}", tid

def frames30_to_gray6464(frames30):
    """
    frames30: (30, 112, 112, 3) or (30, 112, 112) → (30, 64, 64, 1)
    """
    frames30 = np.asarray(frames30)
    if frames30.ndim == 4 and frames30.shape[-1] == 3:
        # RGB -> gray
        frames30 = np.dot(frames30[...,:3], [0.299, 0.587, 0.114])  # (30,H,W)
    elif frames30.ndim == 4 and frames30.shape[-1] == 1:
        frames30 = frames30[..., 0]  # (30,H,W)
    elif frames30.ndim == 3:
        pass  # already (30,H,W)
    else:
        raise ValueError(f"frames30_to_gray6464: unexpected shape {frames30.shape}")

    T, H, W = frames30.shape
    if T != 30:
        raise ValueError(f"expected 30 frames, got {T}")
    out = np.zeros((30, 64, 64, 1), np.float32)
    for t in range(30):
        resized = cv2.resize(frames30[t], (64, 64))
        out[t] = resized[:, :, None]
    return out

# ───────────────────── 3) 파일 찾기 ─────────────────────
print(f"[apex] listing files in {APEX_SEG_DIR}:")
for n in os.listdir(APEX_SEG_DIR):
    print("   -", n)

all_files = sorted(glob.glob(os.path.join(APEX_SEG_DIR, "**", "*.npz"), recursive=True))
print(f"[apex] found {len(all_files)} .npz files (recursive).")
if not all_files:
    print("[apex][ERROR] no .npz found")
    raise SystemExit(1)

# ───────────────────── 4) 메인 루프 ─────────────────────
saved_cnt = 0
for fpath in all_files:
    try:
        label, sid, tid = get_label_sid_tid(fpath)
    except Exception as e:
        print("[apex][skip:badname]", fpath, e)
        continue

    # neutral(4) 은 저장 안 함
    if label == 4:
        continue

    z = np.load(fpath)
    if "multi_clip" not in z:
        print("[apex][skip:nokey]", fpath, "no 'multi_clip' key")
        continue

    mc = z["multi_clip"]
    mc = np.asarray(mc)

    # 여기서 진짜 형식 처리
    # 기대: (8, 3, 30, 112, 112, 3)
    # 허용: (3, 30, 112, 112, 3)  ← 8 없이 들어온 경우
    seg_probs = []

    if mc.ndim == 6:
        # (8, 3, 30, 112, 112, 3)
        num_seg = mc.shape[0]
        for seg_i in range(num_seg):
            three_sec = mc[seg_i]          # (3, 30, 112, 112, 3)
            sec_probs = []
            for sec_i in range(three_sec.shape[0]):
                frames30 = three_sec[sec_i]    # (30,112,112,3)
                try:
                    clip_30 = frames30_to_gray6464(frames30)  # (30,64,64,1)
                except Exception as e:
                    print("[apex][warn] sec skipped in", fpath, "->", e)
                    continue
                clip_30 = np.expand_dims(clip_30, 0)          # (1,30,64,64,1)
                p = apex_model.predict(clip_30, verbose=0)[0] # (4,)
                sec_probs.append(p)
            if not sec_probs:
                continue
            seg_prob = np.stack(sec_probs, axis=0).mean(axis=0)  # 3초 평균
            seg_probs.append(seg_prob)
    elif mc.ndim == 5:
        # (3, 30, 112, 112, 3)  ← 세그먼트 1개만 있는 경우
        sec_probs = []
        for sec_i in range(mc.shape[0]):
            frames30 = mc[sec_i]
            try:
                clip_30 = frames30_to_gray6464(frames30)
            except Exception as e:
                print("[apex][warn] sec skipped in", fpath, "->", e)
                continue
            clip_30 = np.expand_dims(clip_30, 0)
            p = apex_model.predict(clip_30, verbose=0)[0]
            sec_probs.append(p)
        if sec_probs:
            seg_probs.append(np.stack(sec_probs, axis=0).mean(axis=0))
    else:
        print("[apex][warn]", fpath, "-> unexpected ndim", mc.ndim, "shape", mc.shape)
        continue

    if not seg_probs:
        print("[apex][warn] no valid segment for", fpath)
        continue

    # 8개(또는 그 이하) 세그먼트 평균 → trial 하나의 apex score
    trial_prob = np.stack(seg_probs, axis=0).mean(axis=0)
    trial_prob = trial_prob / trial_prob.sum()

    save_path = os.path.join(APEX_SCORE_DIR, f"{sid}_trial{tid:02d}_apex.npy")
    np.save(save_path, trial_prob.astype(np.float32))
    saved_cnt += 1
    print(f"[apex][saved {saved_cnt}] {save_path} -> {trial_prob}")

print(f"[apex] done. saved {saved_cnt} score files to {APEX_SCORE_DIR}")
