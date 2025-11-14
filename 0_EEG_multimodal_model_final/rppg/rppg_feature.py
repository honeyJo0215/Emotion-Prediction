import os
import cv2
import numpy as np
import face_alignment
from face_alignment import LandmarksType
from scipy.signal import detrend, butter, filtfilt, welch
from mne.decoding import CSP
import time

# ───────────────────────────────────────────────────────────────
# 설정
# ───────────────────────────────────────────────────────────────
VIDEO_ROOT   = "/home/bcml1/2025_EMOTION/face_video"
LABEL_ROOT   = "/home/bcml1/2025_EMOTION/DEAP_5labels"
OUTPUT_ROOT  = "/home/bcml1/sigenv/_7월/_rppg/minipatch_rppg_csp"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

N_PATCHES      = 100
TOP_CHANNELS   = 10
CSP_COMPONENTS = 10

# GPU 사용 face_alignment 초기화
fa = face_alignment.FaceAlignment(
    LandmarksType.TWO_D,  # 2D 랜드마크
    device='cuda',        # GPU 사용
    flip_input=False
)

def get_landmark_coords(frame):
    pts = fa.get_landmarks(frame)
    if pts is None:
        return None
    return [(int(x), int(y)) for x, y in pts[0]]

def bandpass_filter(data, fs, lowcut=0.65, highcut=4.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, data, axis=-1)

def chrom_method(rgb):
    R, G, B = rgb
    X = 3*R - 2*G
    Y = 1.5*R + G - 1.5*B
    alpha = np.std(Y) / (np.std(X) + 1e-8)
    s = Y - alpha*X
    return detrend(s)

def process_video_patches(video_path, patch_size=20, n_patches=N_PATCHES):
    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[DEBUG] Processing {os.path.basename(video_path)}: fps={fps:.1f}, frames={total}")

    patch_traces = [[] for _ in range(n_patches)]
    landmark_idxs = None

    for frame_idx in range(total):
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
            if patch.size == 0:
                patch_traces[i].append([np.nan]*3)
            else:
                patch_traces[i].append(patch.reshape(-1,3).mean(axis=0).tolist())
    cap.release()

    patch_traces = np.array(patch_traces, dtype=float)
    signals = []
    for trace in patch_traces:
        # NaN 보간
        for c in range(3):
            ch = trace[:,c]
            nans = np.isnan(ch)
            if nans.any():
                ch[nans] = np.interp(np.flatnonzero(nans),
                                     np.flatnonzero(~nans),
                                     ch[~nans])
                trace[:,c] = ch
        # 정규화 → 밴드패스 → CHROM
        trace = trace / (trace.mean(axis=0) + 1e-8)
        trace = bandpass_filter(trace.T, fs=fps).T
        sig   = chrom_method(trace.T)
        signals.append(sig)

    elapsed = time.time() - start_time
    signals = np.array(signals)
    print(f"[DEBUG] → Completed: signals.shape={signals.shape}, elapsed={elapsed:.2f}s")
    return signals, fps

def load_labels(subj):
    subj_num = subj.replace("s", "subject")
    fn = os.path.join(LABEL_ROOT, f"{subj_num}.npy")
    return np.load(fn).astype(int)

# ───────────────────────────────────────────────────────────────
# 1) rPPG + label 수집 (raw_label==4 스킵)
# ───────────────────────────────────────────────────────────────
entries = []
subjects = [f"s{idx:02d}" for idx in range(1,23)]

for subj in subjects:
    print(f"[INFO] --- Subject {subj} ---")
    labels_raw = load_labels(subj)
    in_dir     = os.path.join(VIDEO_ROOT, subj)
    trials     = sorted([f for f in os.listdir(in_dir) if f.endswith('.avi')])
    for ti, fn in enumerate(trials):
        raw_label = labels_raw[ti]
        if raw_label == 4:
            print(f"  [SKIP] {fn} (label=4)")
            continue

        print(f"  [Trial {ti+1}/{len(trials)}] {fn} (label={raw_label})")
        path = os.path.join(in_dir, fn)
        sigs, fps = process_video_patches(path)
        quality = [np.max(welch(sig, fs=fps, nperseg=min(256,len(sig)))[1]) for sig in sigs]
        idx_top  = np.argsort(quality)[-TOP_CHANNELS:]
        selected = sigs[idx_top]
        print(f"    [DEBUG] top_channels={idx_top.tolist()}, selected.shape={selected.shape}")

        label_bin = 1 if raw_label >= 3 else 0
        entries.append({
            'subj':     subj,
            'trial_fn': fn,
            'signal':   selected,
            'label':    label_bin
        })

all_signals = np.stack([e['signal'] for e in entries])
all_labels  = np.array([e['label']  for e in entries])

# ───────────────────────────────────────────────────────────────
# 2) CSP 학습 & W 필터 얻기
# ───────────────────────────────────────────────────────────────
print("[INFO] Fitting CSP ...")
csp = CSP(n_components=CSP_COMPONENTS, reg=None, log=False, norm_trace=False)
csp.fit(all_signals, all_labels)
W = csp.filters_
print("[INFO] CSP fitted. filters_.shape=", W.shape)

# ───────────────────────────────────────────────────────────────
# 3) CSP 시계열 & log-var 저장
# ───────────────────────────────────────────────────────────────
for e in entries:
    subj     = e['subj']
    fn       = e['trial_fn']
    trial_id = fn.split('_')[1].split('.')[0]
    sig      = e['signal']
    comps    = W.dot(sig)
    logvar   = np.log(np.var(comps, axis=1))

    out_dir = os.path.join(OUTPUT_ROOT, subj)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{trial_id}_csp_ts.npy"),     comps)
    np.save(os.path.join(out_dir, f"{trial_id}_csp_logvar.npy"), logvar)
    print(f"[INFO] Saved CSP for {subj} {trial_id}")
