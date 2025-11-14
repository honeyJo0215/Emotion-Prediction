#!/usr/bin/env python3
"""
rPPG Multi-Channel ROI+MiniPatch CSP Pipeline + Emotion Analysis
1) Video → 3 ROI-green + N_PATCHES mini-patch chrom-method rPPG → Butterworth BP → 10s windows → intermediate save (NPZ)
2) Load NPZ → NaN/Inf clean → CSP → SVM & RF evaluation → save reports, confusion matrices, charts, CSVs
3) Emotion-based feature visualization
   - ROI 시계열 비교
   - Patch 스펙트럼 바 차트
   - 대표 프레임 ROI 마스크 오버레이

* Time optimization: skip extraction if INTERMEDIATE exists
"""
import os, glob, re, math, numpy as np, pandas as pd
import matplotlib.pyplot as plt, cv2
import mediapipe as mp
from scipy.signal import butter, filtfilt, detrend
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from mne.decoding import CSP
import scipy.fftpack as fft

# ─── Configuration ─────────────────────────────────────────────────────
VIDEO_DIR       = "/home/bcml1/2025_EMOTION/face_video"
LABEL_DIR       = "/home/bcml1/2025_EMOTION/DEAP_5labels"
INTERMEDIATE    = "windows_labels.npz"
OUTPUT_DIR      = "results_0723_rppg"
ANALYSIS_DIR    = "emotion_feature_analysis"
FS              = 30.0
LOWCUT, HIGHCUT = 0.65, 4.0
WINDOW_SEC      = 10
WINDOW_LEN      = int(FS * WINDOW_SEC)
STEP            = WINDOW_LEN
N_PATCHES       = 100
CSP_COMPONENTS  = 4
ROI_NAMES       = ['forehead','left_cheek','right_cheek']
LABEL_MAP       = {0:'neutral',1:'sad',2:'fear',3:'happy'}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# ─── Mediapipe FaceMesh init ───────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                              max_num_faces=1,
                                              min_detection_confidence=0.5)

# 3 ROI landmark indices (Facemesh indices)
ROI = [
    [67,69,66,107,9,336,296,299,297,338,10,109],
    [34,234,227,137,147,187,207,216,186,92,165,98,64,48,49,209,198,236,174,188,128,232,231,230,229,228,31,35,143,34],
    [264,372,265,261,448,449,450,451,452,453,412,399,456,420,429,279,278,327,391,322,410,436,427,411,376,323,454,264]
]

# ─── Utility ────────────────────────────────────────────────────────────
def butter_bandpass_filter(data, low, high, fs, order=4):
    nyq = fs * 0.5
    b,a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data, axis=-1)

def get_landmarks(frame):
    # return list of (x,y) for all Facemesh landmarks or None
    h,w = frame.shape[:2]
    res = mp_face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    return [(int(p.x*w), int(p.y*h)) for p in lm]

def chrom_method(rgb_trace):  # rgb_trace: (3,n)
    R,G,B = rgb_trace
    X = 3*R - 2*G
    Y = 1.5*R + G - 1.5*B
    alpha = np.std(Y)/(np.std(X)+1e-8)
    return detrend(Y - alpha*X)

def extract_patches_trace(path, n_patches=N_PATCHES, patch_size=20):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or FS
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    patch_bufs = [[] for _ in range(n_patches)]
    lm_idxs = None
    for _ in range(total):
        ret, frame = cap.read()
        if not ret: break
        pts = get_landmarks(frame)
        if pts is None:
            for buf in patch_bufs: buf.append([np.nan]*3)
            continue
        if lm_idxs is None:
            lm_idxs = np.linspace(0, len(pts)-1, n_patches, dtype=int)
        for i, idx in enumerate(lm_idxs):
            x,y = pts[idx]; h,w = frame.shape[:2]
            x1,y1 = max(0,x-patch_size//2), max(0,y-patch_size//2)
            patch = frame[y1:y1+patch_size, x1:x1+patch_size]
            if patch.size == 0:
                patch_bufs[i].append([np.nan]*3)
            else:
                patch_bufs[i].append(patch.reshape(-1,3).mean(axis=0).tolist())
    cap.release()
    traces = []
    for buf in patch_bufs:
        arr = np.array(buf, float).T  # (3,frames)
        for c in range(3):
            ch = arr[c]; nan = np.isnan(ch)
            if nan.any(): ch[nan] = np.interp(np.flatnonzero(nan), np.flatnonzero(~nan), ch[~nan])
            arr[c] = ch / (ch.mean()+1e-8)
        filt = butter_bandpass_filter(arr, LOWCUT, HIGHCUT, fps)
        traces.append(chrom_method(filt))
    return np.array(traces), fps

# ─── Step1: extract/save windows ─────────────────────────────────────────
def extract_and_save_windows():
    subject_labels = {int(re.search(r"s(\d+)", p).group(1)): np.load(p)
                      for p in glob.glob(os.path.join(LABEL_DIR, "s*_emotion_labels.npy"))}
    Xs, ys, gs = [], [], []
    for subj, labels in subject_labels.items():
        vid_dir = os.path.join(VIDEO_DIR, f"s{subj:02d}")
        for vid in sorted(os.listdir(vid_dir)):
            if not vid.endswith('.avi'): continue
            trial = int(re.search(r"trial(\d+)", vid).group(1))
            lab = labels[trial-1]
            video_path = os.path.join(vid_dir, vid)
            # patches
            patch_signals, fps = extract_patches_trace(video_path)
            # ROI greens
            cap = cv2.VideoCapture(video_path)
            n_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            raw_roi = np.zeros((3, n_fr))
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                pts = get_landmarks(frame)
                for i, roi in enumerate(ROI):
                    if pts is None:
                        raw_roi[i, idx] = np.nan
                    else:
                        poly = np.array([pts[j] for j in roi])
                        mask = np.zeros(frame.shape[:2], np.uint8)
                        cv2.fillPoly(mask, [poly], 255)
                        _, g, _, _ = cv2.mean(frame, mask=mask)
                        raw_roi[i, idx] = g
                idx += 1
            cap.release()
            combined = np.vstack([patch_signals, raw_roi[:, :patch_signals.shape[1]]])
            clean = np.nan_to_num(combined, nan=0, posinf=0)
            filtered = butter_bandpass_filter(clean, LOWCUT, HIGHCUT, FS)
            wins = np.stack([filtered[:, i:i+WINDOW_LEN].T
                             for i in range(0, filtered.shape[1]-WINDOW_LEN+1, STEP)], axis=0)
            Xs.append(wins); ys.append(np.full(wins.shape[0], lab)); gs.append(np.full(wins.shape[0], subj))
            print(f"[s{subj:02d} t{trial:02d}] windows={wins.shape[0]}")
    X = np.vstack(Xs); y = np.concatenate(ys); g = np.concatenate(gs)
    np.savez(INTERMEDIATE, X=X, y=y, groups=g)
    print("▶ Saved intermediate to", INTERMEDIATE)

# ─── Step2: CSP & classification ─────────────────────────────────────────
def run_csp_pipeline():
    data = np.load(INTERMEDIATE); X, y, g = data['X'], data['y'], data['groups']
    mask = np.any(np.isnan(X) | np.isinf(X), axis=(1,2))
    X, y, g = X[~mask], y[~mask], g[~mask]
    Xc = X.transpose(0,2,1).astype(float)
    logo = LeaveOneGroupOut()
    y_true, y_svm, y_rf = [], [], []
    acc_svm, acc_rf = {}, {}
    for tr, te in logo.split(Xc, y, g):
        sid = g[te][0]
        csp = CSP(n_components=CSP_COMPONENTS, log=True, norm_trace=False)
        ftr = csp.fit_transform(Xc[tr], y[tr]); fte = csp.transform(Xc[te])
        ps = SVC().fit(ftr, y[tr]).predict(fte)
        pr = RandomForestClassifier().fit(ftr, y[tr]).predict(fte)
        y_true.extend(y[te]); y_svm.extend(ps); y_rf.extend(pr)
        acc_svm[sid] = accuracy_score(y[te], ps); acc_rf[sid] = accuracy_score(y[te], pr)
        with open(os.path.join(OUTPUT_DIR, f"report_s{sid:02d}_svm.txt"),'w') as f: f.write(classification_report(y[te], ps))
        with open(os.path.join(OUTPUT_DIR, f"report_s{sid:02d}_rf.txt"),'w') as f: f.write(classification_report(y[te], pr))
    with open(os.path.join(OUTPUT_DIR,'report_all_svm.txt'),'w') as f: f.write(classification_report(y_true, y_svm))
    with open(os.path.join(OUTPUT_DIR,'report_all_rf.txt'),'w') as f: f.write(classification_report(y_true, y_rf))
    classes = np.unique(y)
    for name, preds in [('svm', y_svm), ('rf', y_rf)]:
        cm = confusion_matrix(y_true, preds, labels=classes)
        plt.figure(figsize=(6,6)); plt.imshow(cm, cmap='Blues')
        plt.title(f'Confusion Matrix {name.upper()}'); plt.colorbar()
        plt.savefig(os.path.join(OUTPUT_DIR, f'cm_all_{name}.png')); plt.close()
    df_acc = pd.DataFrame({'Subject': list(acc_svm.keys()), 'SVM_Acc': list(acc_svm.values()), 'RF_Acc': list(acc_rf.values())}).sort_values('Subject')
    df_acc.to_csv(os.path.join(OUTPUT_DIR,'subject_accuracy.csv'), index=False)
    stats = df_acc[['SVM_Acc','RF_Acc']].agg(['mean','std'])
    stats.to_csv(os.path.join(OUTPUT_DIR,'accuracy_summary.csv'))
    plt.figure(figsize=(10,5)); plt.bar(df_acc['Subject']-0.2, df_acc['SVM_Acc'], 0.4, label='SVM'); plt.bar(df_acc['Subject']+0.2, df_acc['RF_Acc'], 0.4, label='RF'); plt.xlabel('Subject'); plt.ylabel('Accuracy'); plt.title('Inter-Subject Accuracy'); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR,'accuracy_bar.png')); plt.close()

# ─── Step3: Emotion visualization ────────────────────────────────────────
def emotion_analysis():
    data = np.load(INTERMEDIATE); X, y, _ = data['X'], data['y'], data['groups']
    time = np.arange(WINDOW_LEN)/FS
    for i,name in enumerate(ROI_NAMES):
        plt.figure(figsize=(8,5))
        for lbl,lbl_name in LABEL_MAP.items():
            sel = (y==lbl)
            plt.plot(time, X[sel,:,i].mean(axis=0), label=lbl_name)
        plt.title(f'ROI {name} – Emotion TS'); plt.xlabel('Time(s)'); plt.ylabel('Amp'); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(ANALYSIS_DIR,f'roi_{name}_ts.png')); plt.close()
    freqs = fft.fftfreq(WINDOW_LEN,1/FS); mask=(freqs>=0.5)&(freqs<=5)
    for lbl,lbl_name in LABEL_MAP.items():
        sel=(y==lbl)
        patch=X[sel,:,3:3+N_PATCHES][0]
        amp=np.abs(fft.fft(patch,axis=0)); spec=amp[mask].mean(axis=0)
        plt.figure(figsize=(6,4)); plt.bar(np.arange(N_PATCHES),spec); plt.title(f'Patch Spectrum {lbl_name}'); plt.xlabel('Patch'); plt.ylabel('Mean Amp'); plt.tight_layout(); plt.savefig(os.path.join(ANALYSIS_DIR,f'patch_spec_{lbl_name}.png')); plt.close()
    mp_img=mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
    for lbl,lbl_name in LABEL_MAP.items():
        subj= int(np.unique(data['groups'][data['y']==lbl])[0]); vid= sorted(os.listdir(os.path.join(VIDEO_DIR,f's{subj:02d}')))[0]
        cap=cv2.VideoCapture(os.path.join(VIDEO_DIR,f's{subj:02d}',vid)); cap.set(cv2.CAP_PROP_POS_FRAMES,WINDOW_LEN)
        ret,frame=cap.read(); cap.release(); img=frame.copy(); h,w=img.shape[:2]
        res=mp_img.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); lm=res.multi_face_landmarks[0].landmark
        for roi in ROI:
            pts=np.array([[int(lm[i].x*w),int(lm[i].y*h)] for i in roi])
            cv2.polylines(img,[pts],True,(0,255,0),2)
        cv2.imwrite(os.path.join(ANALYSIS_DIR,f'roi_overlay_{lbl_name}.png'),img)

if __name__=='__main__':
    if not os.path.exists(INTERMEDIATE): extract_and_save_windows()
    run_csp_pipeline()
    emotion_analysis()
    print("✔ All Done. Results in",OUTPUT_DIR,"and",ANALYSIS_DIR)
