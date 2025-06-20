import os, re, threading, time, subprocess
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, constraints, callbacks
from sklearn.model_selection import train_test_split
from scipy.signal import detrend, butter, filtfilt, welch
import cv2
import mediapipe as mp
from collections import Counter, deque

# 경로 설정
EEG_CSP_DIR = "/home/bcml1/2025_EMOTION/DEAP_eeg_new_label_CSP"
VIDEO_ROOT  = "/home/bcml1/2025_EMOTION/face_video"
WEIGHTS_PATH = "multimodal_emotion.h5"

# 하이퍼파라미터
WINDOW_SECS = 1.0
FPS = 30.0
NUM_RPPG_CHANNELS = 3
PATCH_SIZE = 20
NUM_PATCHES = 100
NUM_SAMPLES = 5  # 1초당 샘플링 프레임 수

# GPU 메모리 설정
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

# MediaPipe FaceMesh (EOG용)
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False,
                             max_num_faces=1,
                             refine_landmarks=True,
                             min_detection_confidence=0.5,
                             min_tracking_confidence=0.5)

# ----------------- rPPG 전처리 유틸 -----------------
def bandpass_filter(data, fs, lowcut=0.65, highcut=4.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)

# def chrom_method(rgb):
#     R, G, B = rgb
#     X = 3*R - 2*G
#     Y = 1.5*R + G - 1.5*B
#     alpha = np.std(Y)/(np.std(X)+1e-8)
#     return detrend(Y - alpha*X)

def chrom_method(rgb):
    # 입력을 float64 배열로 변환
    rgb = np.array(rgb, dtype=np.float64)

    # NaN/Inf 보간
    for i in range(rgb.shape[0]):
        ch = rgb[i]
        mask = ~np.isfinite(ch)
        if mask.any():
            valid = ~mask
            if valid.sum() > 0:
                ch[mask] = np.interp(
                    np.flatnonzero(mask),
                    np.flatnonzero(valid),
                    ch[valid]
                )
            else:
                ch[mask] = 0.0
        rgb[i] = ch

    R, G, B = rgb
    X = 3*R - 2*G
    Y = 1.5*R + G - 1.5*B

    # alpha 계산 시 0 분모 방지
    alpha = np.std(Y) / (np.std(X) + 1e-8)

    # 신호 생성 후 남은 NaN/Inf 제거
    sig = Y - alpha*X
    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)

    return detrend(sig)

# ----------------- raw 패치 RGB 트레이스 추출 -----------------
def extract_patch_traces(frames):
    # 1초 윈도우에서 NUM_SAMPLES 프레임 등간 샘플링
    idxs = np.linspace(0, len(frames)-1, NUM_SAMPLES, dtype=int)
    traces = [[] for _ in range(NUM_PATCHES)]
    landmark_idxs = None
    for i in idxs:
        frame = frames[i]
        res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            for buf in traces:
                buf.append([np.nan]*3)
            continue
        lm = res.multi_face_landmarks[0].landmark
        pts = [(int(p.x*frame.shape[1]), int(p.y*frame.shape[0])) for p in lm]
        if landmark_idxs is None:
            landmark_idxs = np.linspace(0, len(pts)-1, NUM_PATCHES, dtype=int)
        for j, pi in enumerate(landmark_idxs):
            x, y = pts[pi]
            x1, y1 = max(0, x-PATCH_SIZE//2), max(0, y-PATCH_SIZE//2)
            x2, y2 = min(frame.shape[1], x1+PATCH_SIZE), min(frame.shape[0], y1+PATCH_SIZE)
            patch = frame[y1:y2, x1:x2]
            if patch.size==0:
                traces[j].append([np.nan]*3)
            else:
                traces[j].append(np.mean(patch.reshape(-1,3), axis=0).tolist())
    return np.array(traces), FPS

# ----------------- 최적 rPPG 채널(패치) 추출 -----------------
def extract_rppg_signals(frames):
    raw_traces, fps = extract_patch_traces(frames)
    signals = np.stack([chrom_method(tr.T) for tr in raw_traces])
    stds = signals.std(axis=1)
    chosen = np.argsort(stds)[:NUM_RPPG_CHANNELS]
    return signals[chosen], fps

# ----------------- EOG 피처 추출 -----------------
def your_extract_eog_features(frames, fps=FPS, window_secs=WINDOW_SECS):
    # 1초 윈도우에서 NUM_SAMPLES 프레임 샘플링
    idxs = np.linspace(0, len(frames)-1, NUM_SAMPLES, dtype=int)
    sampled = [frames[i] for i in idxs]
    detected = 0
    missed = 0
    segX, segY, blink_marks = [], [], []
    for img in sampled:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            missed += 1
            continue
        detected += 1
        lm = res.multi_face_landmarks[0].landmark
        left = np.array([[lm[i].x*img.shape[1], lm[i].y*img.shape[0]] for i in range(468,473)])
        right= np.array([[lm[i].x*img.shape[1], lm[i].y*img.shape[0]] for i in range(473,478)])
        pts = np.vstack([left,right]); cx, cy = pts.mean(0)
        segX.append(cx); segY.append(cy)
        def dist(a,b): return np.linalg.norm(np.array([lm[a].x,lm[a].y])-np.array([lm[b].x,lm[b].y]))
        ear = (dist(160,144)+dist(158,153))/(2*dist(33,133))
        blink_marks.append(ear<0.2)
    if len(segX)<2:
        feats = np.zeros((31,1),dtype=np.float32)
        img_feat = np.zeros((64,64,3),dtype=np.uint8)
        return feats, img_feat
    segX, segY = np.array(segX), np.array(segY)
    dX, dY = np.diff(segX), np.diff(segY)
    dispX = np.abs(dX); dispY = np.abs(dY)
    mag   = np.sqrt(dX**2 + dY**2)

    # --- fixation: 속도 < 문턱치(v_thr)인 구간 ---
    # --- dynamic threshold based on this window's speed statistics ---
    v = mag * fps                  # instantaneous velocity in px/sec
    if v.size >= 2:
        v_mean, v_std = v.mean(), v.std()
        v_thr = v_mean + v_std
    elif v.size == 1:
        v_mean = float(v)
        v_std  = 0.0
        v_thr  = v_mean
    else:
        v_mean = 0.0
        v_std  = 0.0
        v_thr  = 0.0
    print(f"[EOG] window={window_secs}s, v_mean={v_mean:.1f}, v_std={v_std:.1f}, v_thr={v_thr:.1f}")            # 조정 필요
    fix_mask = v < v_thr
    # 연속 True 세그먼트 길이를 프레임 단위로 구함
    fix_durs = []
    cnt = 0
    for f in fix_mask:
        if f: cnt += 1
        elif cnt>0:
            fix_durs.append(cnt/fps)
            cnt = 0
    if cnt>0: fix_durs.append(cnt/fps)

    # --- blink: 연속 True 구간 길이 ---
    blink_durs = []
    cnt=0
    for b in blink_marks:
        if b: cnt+=1
        elif cnt>0:
            blink_durs.append(cnt/fps)
            cnt=0
    if cnt>0: blink_durs.append(cnt/fps)

    # --- saccade: fixation 아닌 구간 (속도 ≥ v_thr) ---
    sac_durs = []
    sac_amps = []
    cnt=0; start_x=0; start_y=0
    for i, f in enumerate(fix_mask):
        if not f:  # saccade 구간
            if cnt==0:
                start_x, start_y = segX[i], segY[i]
            cnt+=1
        elif cnt>0:
            end_x, end_y = segX[i], segY[i]
            sac_durs.append(cnt/fps)
            sac_amps.append(np.hypot(end_x-start_x, end_y-start_y))
            cnt=0
    if cnt>0:
        end_x, end_y = segX[-1], segY[-1]
        sac_durs.append(cnt/fps)
        sac_amps.append(np.hypot(end_x-start_x, end_y-start_y))

    # --- 31개 feature 계산 ---
    feats = np.zeros((31,), dtype=np.float32)
    feats[0]  = segX.mean()
    feats[1]  = segY.mean()
    feats[2]  = segX.std()
    feats[3]  = segY.std()
    feats[4]  = dX.mean()
    feats[5]  = dX.std()
    feats[6]  = dX.max()
    feats[7]  = dX.min()
    feats[8]  = dY.mean()
    feats[9]  = dY.std()
    feats[10] = dY.max()
    feats[11] = dY.min()
    feats[12] = np.mean(fix_durs)   if fix_durs else 0.0
    feats[13] = np.std(fix_durs)    if fix_durs else 0.0
    feats[14] = np.max(fix_durs)    if fix_durs else 0.0
    feats[15] = len(fix_durs)/(len(frames)/fps)
    feats[16] = dispX.mean()
    feats[17] = dispY.mean()
    feats[18] = dispX.std()
    feats[19] = dispY.std()
    feats[20] = mag.sum()
    feats[21] = mag.max()
    feats[22] = len(blink_durs)/(len(frames)/fps)
    feats[23] = np.mean(sac_durs)   if sac_durs else 0.0
    feats[24] = np.std(sac_durs)    if sac_durs else 0.0
    feats[25] = np.median(sac_durs) if sac_durs else 0.0
    feats[26] = np.mean(sac_amps)   if sac_amps else 0.0
    feats[27] = np.std(sac_amps)    if sac_amps else 0.0
    feats[28] = np.median(sac_amps) if sac_amps else 0.0
    feats[29] = len(sac_durs)/(len(frames)/fps)
    # saccade 속도 = 진폭/지속시간 평균
    feats[30] = np.mean([a/d for a,d in zip(sac_amps, sac_durs)]) if sac_durs else 0.0
    # — Debug: print or log the vector so you can see each dimension
    #    you can also write these to a rotating file or stdout.
    feature_names = [
        'pupilX_mean','pupilY_mean','pupilX_std','pupilY_std',
        'dX_mean','dX_std','dX_max','dX_min','dY_mean','dY_std',
        'dY_max','dY_min','fix_dur_mean','fix_dur_std','fix_dur_max',
        'fix_freq','dispX_mean','dispY_mean','dispX_std','dispY_std',
        'mag_sum','mag_max','blink_freq','sac_dur_mean','sac_dur_std',
        'sac_dur_med','sac_amp_mean','sac_amp_std','sac_amp_med',
        'sac_freq','sac_speed'
    ]
    for name, val in zip(feature_names, feats):
        print(f"[EOG] {name:15s} = {val:6.3f}")

    feats = np.zeros((31,1),dtype=np.float32)
    avg = np.mean(sampled,axis=0).astype(np.uint8)
    img_feat = cv2.resize(avg,(64,64))
    return feats, img_feat

# ----------------- 데이터 로딩 -----------------
def load_multimodal_features():
    X_eeg, X_eog, X_rppg, X_img, Y = [], [], [], [], []
    for fn in os.listdir(EEG_CSP_DIR):
        if not fn.endswith('.npy'): continue
        m = re.match(r'folder\d+_subject(\d+)_sample(\d+)_label(\d+)\.npy$', fn)
        if not m: continue
        subj, samp, label = map(int, m.groups())
        if label==4: continue
        eeg = np.load(os.path.join(EEG_CSP_DIR, fn))
        vid = os.path.join(VIDEO_ROOT, f"s{subj:02d}", f"s{subj:02d}_trial{samp:02d}.avi")
        if not os.path.isfile(vid): continue
        cap = cv2.VideoCapture(vid); frames=[]
        while True:
            ret, fr = cap.read()
            if not ret: break
            frames.append(fr)
        cap.release()
        win_n    = int(WINDOW_SECS * FPS)
        n_video  = len(frames) // win_n
        if n_video == 0:
            continue
        # EEG 전체 길이를 윈도우 개수로 나눠 고정 길이 세그먼트 크기 계산
        eeg_win_len = eeg.shape[1] // n_video

        for i in range(n_video):
                # 1) 프레임 세그먼트 (항상 win_n 프레임)
                seg_frames = frames[i*win_n:(i+1)*win_n]

                # 2) EEG 세그먼트 (항상 eeg_win_len 시점)
                seg_eeg = eeg[:, i*eeg_win_len:(i+1)*eeg_win_len, :]
                X_eeg.append(seg_eeg)

                # 3) EOG 추출
                eog_feat, img_eog = your_extract_eog_features(seg_frames)
                X_eog.append(eog_feat.squeeze())
                X_img.append(img_eog)

                # 4) rPPG 추출
                rppg_signals, _ = extract_rppg_signals(seg_frames)
                X_rppg.append(rppg_signals)

                # 5) 라벨
                Y.append(label)
    return (np.stack(X_eeg), np.stack(X_eog), np.stack(X_rppg), np.stack(X_img), np.array(Y))

# ----------------- 분기 모델 및 멀티모달 구성 -----------------
def build_branch_models():
    inp_eog = layers.Input((31,)); out_eog = layers.GaussianNoise(0.05)(inp_eog)
    m_eog = models.Model(inp_eog,out_eog)
    win_n=int(WINDOW_SECS*FPS)
    inp_r = layers.Input((NUM_RPPG_CHANNELS,win_n))
    x=layers.Permute((2,1))(inp_r)
    x=layers.Conv1D(16,3,padding='same',activation='relu')(x)
    out_r=layers.GlobalAveragePooling1D()(x)
    m_r = models.Model(inp_r,out_r)
    return m_eog, m_r

def build_multimodal_model(num_classes=4):
    win_n=int(WINDOW_SECS*FPS)
    in_eeg=layers.Input((4,win_n,8)); x_eeg=layers.Flatten()(in_eeg)
    x_eeg=layers.Dense(64,activation='relu')(x_eeg)
    in_eog=layers.Input((31,)); in_img=layers.Input((64,64,3)); in_r=layers.Input((NUM_RPPG_CHANNELS,win_n))
    m_eog,m_r=build_branch_models()
    x_eog=m_eog(in_eog)
    x_img=layers.TimeDistributed(layers.Conv2D(16,(3,3),activation='relu'))(layers.Reshape((64,64,3,1))(in_img))
    x_img=layers.GlobalAveragePooling2D()(x_img)
    x_r=m_r(in_r)
    x=layers.Concatenate()([x_eeg,x_eog,x_img,x_r])
    x=layers.Dense(64,activation='relu')(x); x=layers.Dropout(0.3)(x)
    out=layers.Dense(num_classes,activation='softmax')(x)
    return models.Model([in_eeg,in_eog,in_img,in_r],out)

# ----------------- 학습 -----------------
def train_and_save_model():
    Xe, Eo, Xr, Xi, Y=load_multimodal_features()
    Xe_tr,Xe_te,Eo_tr,Eo_te,Xr_tr,Xr_te,Xi_tr,Xi_te,y_tr,y_te=\
        train_test_split(Xe,Eo,Xr,Xi,Y,test_size=0.3, stratify=Y, random_state=42)
    train_ds=tf.data.Dataset.from_tensor_slices(((Xe_tr,Eo_tr,Xi_tr,Xr_tr),y_tr))\
        .shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    test_ds=tf.data.Dataset.from_tensor_slices(((Xe_te,Eo_te,Xi_te,Xr_te),y_te))\
        .batch(32).prefetch(tf.data.AUTOTUNE)
    model=build_multimodal_model(num_classes=len(np.unique(Y)))
    model.compile(optimizers.Adam(1e-4),'sparse_categorical_crossentropy',['accuracy'])
    model.fit(train_ds,epochs=20,validation_data=test_ds,callbacks=[callbacks.EarlyStopping('val_accuracy',patience=5,restore_best_weights=True)])
    model.save_weights(WEIGHTS_PATH)

# ----------------- 실시간 추론 -----------------
def run_live_inference():
    model=build_multimodal_model(); model.load_weights(WEIGHTS_PATH)
    cap=cv2.VideoCapture(0)
    while True:
        frames=[]
        for _ in range(int(WINDOW_SECS*FPS)):
            ret,fr=cap.read();
            if not ret:break
            frames.append(fr)
        if len(frames)<int(WINDOW_SECS*FPS):break
        eog_f,_=your_extract_eog_features(frames)
        r_s,_=extract_rppg_signals(frames)
        eeg_d=np.zeros((1,4,int(WINDOW_SECS*FPS),8),dtype=np.float32)
        img_m=cv2.resize(np.mean(frames,axis=0).astype(np.uint8),(64,64))
        preds=model.predict([eeg_d,eog_f.T[None],img_m[None],r_s[None]],verbose=0)
        print("Predicted:",np.argmax(preds,axis=1)[0])
    cap.release()

if __name__=='__main__':
    train_and_save_model()
    #run_live_inference()
