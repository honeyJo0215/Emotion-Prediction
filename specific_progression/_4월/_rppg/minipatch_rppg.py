import os
import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import detrend, butter, filtfilt, welch
from sklearn.cluster import KMeans

# 경로
VIDEO_ROOT = "/home/bcml1/2025_EMOTION/face_video"
OUTPUT_ROOT = "/home/bcml1/sigenv/_4월/_rppg/minipatch_rppg"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Butterworth bandpass filter (0.65–4 Hz)
def bandpass_filter(data, fs, lowcut=0.65, highcut=4.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)

# CHROM 알고리즘: RGB → 1D rPPG
def chrom_method(rgb):
    R, G, B = rgb
    X = 3 * R - 2 * G
    Y = 1.5 * R + G - 1.5 * B
    alpha = np.std(Y) / (np.std(X) + 1e-8)
    s = Y - alpha * X
    return detrend(s)

# MediaPipe FaceMesh 초기화
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1)

def get_landmark_coords(frame):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0]
    return [(int(p.x * w), int(p.y * h)) for p in lm.landmark]

def process_video(video_path, patch_size=20, n_patches=100):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 패치별 RGB 트레이스 저장 버퍼
    patch_traces = [[] for _ in range(n_patches)]
    landmark_idxs = None

    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        coords = get_landmark_coords(frame)
        if coords is None:
            # 얼굴 인식 실패 시 NaN 채우기
            for buf in patch_traces:
                buf.append([np.nan, np.nan, np.nan])
            continue

        if landmark_idxs is None:
            # 균등 샘플링으로 패치 위치 선택 (초기 1회만)
            landmark_idxs = np.linspace(0, len(coords) - 1, n_patches, dtype=int)

        # 각 패치 중심 RGB 평균값 추출
        for i, idx in enumerate(landmark_idxs):
            x, y = coords[idx]
            x1, y1 = max(0, x - patch_size//2), max(0, y - patch_size//2)
            x2, y2 = min(frame.shape[1], x1 + patch_size), min(frame.shape[0], y1 + patch_size)
            patch = frame[y1:y2, x1:x2]
            if patch.size == 0:
                patch_traces[i].append([np.nan, np.nan, np.nan])
            else:
                patch_traces[i].append(patch.reshape(-1,3).mean(axis=0).tolist())

    cap.release()

    # numpy 배열로 변환: (n_patches, frames, 3)
    patch_traces = np.array(patch_traces, dtype=float)

    # 전처리 및 CHROM → 1D rPPG
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
        # 정규화, 밴드패스, CHROM
        trace = trace / (trace.mean(axis=0) + 1e-8)
        trace = bandpass_filter(trace.T, fs=fps).T
        sig = chrom_method(trace.T)
        signals.append(sig)
    signals = np.array(signals)

    # 패치별 PSD 피크 주파수 추출
    peaks = []
    for sig in signals:
        f, Pxx = welch(sig, fs=fps, nperseg=min(256,len(sig)))
        mask = (f>=0.65)&(f<=4.0)
        if not mask.any():
            peaks.append(0)
        else:
            peaks.append(f[mask][np.argmax(Pxx[mask])])
    peaks = np.array(peaks).reshape(-1,1)

    # KMeans(2)로 최적의(good) 패치 군집 선택
    km = KMeans(n_clusters=2, random_state=0).fit(peaks)
    labels = km.labels_
    stds = [peaks[labels==k].std() for k in [0,1]]
    good_label = int(np.argmin(stds))

    # 최종 rPPG = good 패치들 시그널 평균
    final_sig = signals[labels==good_label].mean(axis=0)

    # 최종 BPM 추정
    f, Pxx = welch(final_sig, fs=fps, nperseg=min(256,len(final_sig)))
    mask = (f>=0.65)&(f<=4.0)
    hr_hz = f[mask][np.argmax(Pxx[mask])] if mask.any() else np.nan
    return final_sig, hr_hz * 60

if __name__ == "__main__":
    for subj in [f"s{idx:02d}" for idx in range(1,23)]:
        out_dir = os.path.join(OUTPUT_ROOT, subj)
        os.makedirs(out_dir, exist_ok=True)
        in_dir = os.path.join(VIDEO_ROOT, subj)
        for trial in range(1,41):
            fn = f"{subj}_trial{trial:02d}.avi"
            path = os.path.join(in_dir, fn)
            if not os.path.isfile(path):
                continue
            rppg, hr = process_video(path)
            # 결과 저장
            np.save(os.path.join(out_dir, fn.replace('.avi','_rppg.npy')), rppg)
            with open(os.path.join(out_dir, fn.replace('.avi','_hr.txt')), 'w') as f:
                f.write(f"{hr:.2f}\n")
