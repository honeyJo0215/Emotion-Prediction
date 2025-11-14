import os
import cv2
import numpy as np
import face_alignment
from face_alignment import LandmarksType
from scipy.signal import detrend, butter, filtfilt
from mne.decoding import CSP
import time

# ───────────────────────────────────────────────────────────────
# 설정
# ───────────────────────────────────────────────────────────────
VIDEO_ROOT   = "/home/bcml1/2025_EMOTION/face_video"
LABEL_ROOT   = "/home/bcml1/2025_EMOTION/DEAP_5labels"
OUTPUT_ROOT  = "/home/bcml1/sigenv/_7월/_rppg/minipatch_rppg_csp"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

N_PATCHES      = 100    # 미니패치 개수
CSP_COMPONENTS = 5      # 클래스당 CSP 컴포넌트 개수
TARGET_LEN     = 3000   # 고정 저장 프레임 길이
EXPECTED_CLASSES = [0, 1, 2, 3]  # 기대하는 감정 클래스 인덱스

# GPU 사용 face_alignment 초기화
fa = face_alignment.FaceAlignment(
    LandmarksType.TWO_D,
    device='cuda',
    flip_input=False
)

# 랜덤 선택용 시드
rng = np.random.RandomState(42)

def get_landmark_coords(frame):
    pts = fa.get_landmarks(frame)
    return None if pts is None else [(int(x), int(y)) for x, y in pts[0]]

def bandpass_filter(data, fs, lowcut=0.65, highcut=4.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, data, axis=-1)

def chrom_method(rgb):
    R, G, B = rgb
    X = 3*R - 2*G
    Y = 1.5*R + G - 1.5*B
    alpha = np.std(Y) / (np.std(X)+1e-8)
    return detrend(Y - alpha*X)

def process_video_patches(video_path, patch_size=20, n_patches=N_PATCHES):
    cap = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    patch_traces = [[] for _ in range(n_patches)]
    landmark_idxs = None
    for _ in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        coords = get_landmark_coords(frame)
        if coords is None:
            for buf in patch_traces:
                buf.append([np.nan]*3)
            continue
        if landmark_idxs is None:
            landmark_idxs = np.linspace(0, len(coords)-1, n_patches, dtype=int)
        for i, idx in enumerate(landmark_idxs):
            x, y = coords[idx]
            x1, y1 = max(0, x-patch_size//2), max(0, y-patch_size//2)
            x2, y2 = min(frame.shape[1], x1+patch_size), min(frame.shape[0], y1+patch_size)
            patch = frame[y1:y2, x1:x2]
            if patch.size==0:
                patch_traces[i].append([np.nan]*3)
            else:
                patch_traces[i].append(patch.reshape(-1,3).mean(axis=0).tolist())
    cap.release()

    patch_traces = np.array(patch_traces, float)
    signals = []
    for trace in patch_traces:
        for c in range(3):
            ch = trace[:,c]
            nans = np.isnan(ch)
            if nans.any():
                ch[nans] = np.interp(np.flatnonzero(nans),
                                     np.flatnonzero(~nans),
                                     ch[~nans])
                trace[:,c] = ch
        norm = trace / (trace.mean(axis=0)+1e-8)
        filt = bandpass_filter(norm.T, fs=fps).T
        signals.append(chrom_method(filt.T))
    return np.array(signals), fps

def load_labels(subj):
    fn = os.path.join(LABEL_ROOT, subj.replace("s","subject") + ".npy")
    try:
        return np.load(fn).astype(int)
    except FileNotFoundError:
        return None

if __name__ == "__main__":
    for subj in [f"s{idx:02d}" for idx in range(10,23)]:
        print(f"\n[INFO] Subject {subj}")
        raw_labels = load_labels(subj)
        if raw_labels is None:
            print("  [WARN] No label file, skipping.")
            continue

        vids = sorted(f for f in os.listdir(os.path.join(VIDEO_ROOT, subj)) if f.endswith('.avi'))
        signals_list, labels_list, trials_list = [], [], []
        for ti, fn in enumerate(vids):
            if ti >= len(raw_labels):
                continue
            raw = raw_labels[ti]
            if raw == 4:
                print(f"  [SKIP] {fn} (label=4)")
                continue
            cls = {0:0, 1:1, 2:2, 3:3}[raw]
            path = os.path.join(VIDEO_ROOT, subj, fn)
            sigs, _ = process_video_patches(path)
            signals_list.append(sigs)
            labels_list.append(cls)
            trials_list.append(fn)
            print(f"  [DEBUG] Collected {fn}: signal shape={sigs.shape}, label={cls}")


        if not signals_list:
            print("  [WARN] No valid trials, skipping.")
            continue

        # CSP 학습: 클래스별 one-vs-rest + 부족 클래스 보완
        X_sub = np.stack(signals_list)  # (n_trials, n_patches, frames)
        y_sub = np.array(labels_list)
        filters_per_class = {}
        for c in EXPECTED_CLASSES:
            if c in np.unique(y_sub):
                # 실제 데이터 기반 CSP
                y_bin = (y_sub == c).astype(int)
                csp = CSP(n_components=CSP_COMPONENTS, reg=None, log=False, norm_trace=False)
                csp.fit(X_sub, y_bin)
                W = csp.filters_[:CSP_COMPONENTS]
            else:
                # 부족 클래스: 랜덤 채널 추출 필터 생성
                idxs = rng.choice(N_PATCHES, CSP_COMPONENTS, replace=False)
                W = np.eye(N_PATCHES)[idxs]
            filters_per_class[c] = W
            print(f"  Class {c} filters shape: {W.shape}")

        out_dir = os.path.join(OUTPUT_ROOT, subj)
        os.makedirs(out_dir, exist_ok=True)
        for fn, sigs, cls in zip(trials_list, signals_list, labels_list):
            trial_id = fn.split('_')[1].split('.')[0]
            comps_all = []
            for c in EXPECTED_CLASSES:
                W = filters_per_class[c]
                comps = W.dot(sigs)  # (CSP_COMPONENTS, frames)
                n = comps.shape[1]
                if n >= TARGET_LEN:
                    comps = comps[:, :TARGET_LEN]
                else:
                    comps = np.pad(comps, ((0,0),(0, TARGET_LEN-n)), mode='edge')
                comps_all.append(comps)
            multi_comps = np.vstack(comps_all)
            save_path = os.path.join(out_dir, f"{trial_id}_csp_rppg_multiclass.npy")
            np.save(save_path, multi_comps)
            print(f"  [SAVED] {trial_id}, shape={multi_comps.shape}")
