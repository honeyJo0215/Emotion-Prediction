import os, re, subprocess, threading
import numpy as np
import tensorflow as tf
#print(tf.__version__)                # TF 버전
#print("Built with CUDA:",            tf.test.is_built_with_cuda())
#print("Physical GPUs:",              tf.config.list_physical_devices('GPU'))
#print("Available GPUs:", tf.config.list_physical_devices('GPU'))
from tensorflow.keras import layers, models, optimizers, callbacks, constraints
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import mediapipe as mp
import time
from typing import Dict
from collections import Counter, deque

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# ... (other imports and feature extraction functions remain unchanged) ...

# ── Mediapipe FaceMesh (정밀 iris 모델) 초기화 ──────────────────────────────
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

FFPLAY_PATH = r"C:\ffmpeg-7.1.1-full_build\ffmpeg-7.1.1-full_build\bin\ffplay.exe"

# 미디어 파일 경로
WAITING_MEDIA      = "media/introduction_hell.mp4"
WAITING2_MEDIA     = "media/calibration.mp4"
PREDICT_MEDIA = "media/predict_intro.mp4"
instructions_intro_media = {
    0: "media/Neutral_intro.mp4",
    1: "media/Sad_intro.mp4",
    2: "media/fear_intro.mp4",
    3: "media/happy_intro.mp4"
}
instructions_media      = {
    0: "media/gray.png",
    1: "media/blue.png",
    2: "media/red.png",
    3: "media/yellow.png"
}
feedback_media         = {
    0: "media/neutral.png",
    1: "media/sad.png",
    2: "media/fear.png",
    3: "media/happy.png"
}
CAMERA_ID = 0
WINDOW_SECONDS = 1.0
CALIBRATE_SECONDS = 70.0
MAX_DEBUG_LINES    = 10
debug_lines = []

class DebugCallback(Callback):
    def __init__(self, cap_cam, win, video_path, cam_inset_size=(1/8,1/8)):
        super().__init__()
        self.cap_cam = cap_cam
        self.win     = win
        self.cap_med = cv2.VideoCapture(video_path)
        self._start_audio(video_path)
        self.inset_w_frac, self.inset_h_frac = cam_inset_size
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.win, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

    def _start_audio(self, path):
        def _play():
            cmd = [FFPLAY_PATH, '-i', path, '-autoexit','-nodisp','-volume','100','-loglevel','warning']
            try: subprocess.run(cmd, check=True)
            except: pass
        threading.Thread(target=_play, daemon=True).start()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        debug_lines.append(
            f"[E{epoch+1}] loss={logs.get('loss',0):.4f} acc={logs.get('accuracy',0):.4f}"
        )
        if len(debug_lines) > MAX_DEBUG_LINES:
            debug_lines[:] = debug_lines[-MAX_DEBUG_LINES:]

        # video frame
        ret_v, med_frame = self.cap_med.read()
        if not ret_v:
            self.cap_med.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_v, med_frame = self.cap_med.read()
        bg = (np.zeros((1080,1920,3),np.uint8)
              if med_frame is None
              else cv2.resize(med_frame,(1920,1080)))

        # camera inset
        ret_c, cam_frame = self.cap_cam.read()
        if ret_c and cam_frame is not None:
            h,w = bg.shape[:2]
            iw,ih = int(w*self.inset_w_frac), int(h*self.inset_h_frac)
            small = cv2.resize(cam_frame,(iw,ih))
            bg[h-ih-10:h-10, w-iw-10:w-10] = small

        # debug text
        for i,txt in enumerate(debug_lines):
            cv2.putText(bg, txt,
                        (10,1020-(MAX_DEBUG_LINES-i)*18),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),1)
        cv2.imshow(self.win, bg)
        cv2.waitKey(1)


# ── EOG feature 추출 함수 (변경 없음) ───────────────────────────────────────
def your_extract_eog_features(frames, fps=30.0, window_secs=1.0):
    """
    frames: List[np.ndarray(BGR)]  한 윈도우(예:1초) 동안의 프레임
    fps:     float                 초당 프레임 수
    +    window_secs: float            frames 윈도우 길이 (초)
    +    return:  np.ndarray shape (31,1)
    """
    # --- 디버그용 카운트 초기화 ---
    detected = 0
    missed   = 0
    segX, segY = [], []
    blink_marks = []
    # Fixation/saccade용 속도
    for img in frames:
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            missed += 1
            continue
        detected += 1
        lm = res.multi_face_landmarks[0].landmark

        # 1) iris center: left iris landmarks 474~477
        # pts = np.array([[lm[i].x * w, lm[i].y * h] for i in (474,475,476,477)])
        # 1) iris center: 
        #    Left iris = 5 points 468–472, Right iris = 473–477
        #    Let’s average both eyes to be robust:
        # left_pts  = np.array([[lm[i].x * w, lm[i].y * h] for i in range(468,473)])
        # right_pts = np.array([[lm[i].x * w, lm[i].y * h] for i in range(473,478)])
        # pts = np.vstack([left_pts, right_pts])
        # cx, cy = pts.mean(axis=0)
        # segX.append(cx); segY.append(cy)
        try:
            left_pts  = np.array([[lm[i].x*w, lm[i].y*h] for i in range(468,473)])
            right_pts = np.array([[lm[i].x*w, lm[i].y*h] for i in range(473,478)])
        except Exception:
            # fallback: eye corners indices
            left_pts  = np.array([[lm[i].x*w, lm[i].y*h] for i in (33,133)])
            right_pts = np.array([[lm[i].x*w, lm[i].y*h] for i in (362,263)])

        pts = np.vstack([left_pts, right_pts])
        cx, cy = pts.mean(axis=0)
        segX.append(cx); segY.append(cy)


        # 2) blink mark: eye aspect ratio 이용 (left eye)
        #   p1=33, p2=160, p3=158, p4=133, p5=153, p6=144
        def dist(a,b):
            return np.linalg.norm(np.array([lm[a].x, lm[a].y]) - np.array([lm[b].x, lm[b].y]))
        ear = (dist(160,144) + dist(158,153)) / (2.0 * dist(33,133))
        blink_marks.append(ear < 0.2)  # threshold

    # 디버그 출력
    print(f"[EOG DEBUG] detected_landmarks={detected}, missed={missed}")
    # 배열 변환
    segX = np.array(segX); segY = np.array(segY)
    if len(segX) < 2:
        zero_feat = np.zeros((31,1), dtype=np.float32)
        zero_img  = np.zeros((64,64,3), dtype=np.float32)
        return zero_feat, zero_img
    else:
        segX = np.array(segX); segY = np.array(segY)
        dX = np.diff(segX); dY = np.diff(segY)
        dispX, dispY = np.abs(dX), np.abs(dY)
        mag = np.hypot(dX,dY)
        v = mag * fps
        v_mean, v_std = (v.mean(),v.std()) if v.size>=2 else (float(v),0.0)
        v_thr = v_mean + v_std
        fix_mask = v < v_thr
    # 차분
    dX = np.diff(segX)
    dY = np.diff(segY)
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
    feats = feats[:,None]
    # 평균 eye 이미지
    avg = np.mean(frames, axis=0).astype(np.uint8)
    avg_resized = cv2.resize(avg, (64,64))
    avg_norm = avg_resized.astype(np.float32)/255.0

    return feats, avg_norm


# ── 샘플 제너레이터 정의 ──────────────────────────────────────────────────
def make_generator(Xe, Xy, y):
    # Xe: numpy array (N,4,200,8), Xy: (N,31), y: (N,)
    for eeg_sample, eye_sample, label in zip(Xe, Xy, y):
        yield (eeg_sample.astype(np.float32),
               eye_sample.astype(np.float32)), np.int32(label)



# ── Modality Dropout 함수 ───────────────────────────────────────────────
def modality_dropout(inputs, label, p_drop=0.2):
    """
    inputs: (tuple) (eeg, eye)
    label:   (tensor) 정답 레이블
    p_drop:  (float) 하나의 모달리티를 마스킹할 확률
    """
    eeg, eye = inputs

    # 랜덤 숫자 하나 뽑아서 [0,1) 범위로
    r = tf.random.uniform(())
    # 20% 확률로 EEG 마스킹
    eeg = tf.cond(r < p_drop,
                  lambda: tf.zeros_like(eeg),
                  lambda: eeg)
    # 또다른 랜덤 숫자로 Eye 마스킹 (20%)
    r2 = tf.random.uniform(())
    eye = tf.cond(r2 < p_drop,
                  lambda: tf.zeros_like(eye),
                  lambda: eye)

    return (eeg, eye), label


# ── Custom Layers (학습 시 사용한 것과 동일해야 합니다) ─────────────────
class BidirectionalSSMLayer(layers.Layer):
    def __init__(self, units, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.fwd_conv = layers.Conv1D(1, kernel_size, padding='causal')
        self.bwd_conv = layers.Conv1D(1, kernel_size, padding='causal')
    def call(self, x):
        seq = tf.expand_dims(x, -1)
        fwd = self.fwd_conv(seq)
        rev = tf.reverse(seq, axis=[1])
        bwd = self.bwd_conv(rev)
        bwd = tf.reverse(bwd, axis=[1])
        return tf.squeeze(fwd + bwd, axis=-1)

class DiffuSSMLayer(layers.Layer):
    def __init__(self, model_dim, hidden_dim, alpha_init=0.0, **kwargs):
        super().__init__(**kwargs)
        self.norm    = layers.LayerNormalization()
        self.hg_down = layers.Dense(hidden_dim, activation='swish')
        self.ssm     = BidirectionalSSMLayer(hidden_dim, kernel_size=3)
        self.hg_up   = layers.Dense(model_dim, activation=None)
        self.alpha   = self.add_weight(
            name="fusion_scale_alpha", shape=(1,),
            initializer=tf.keras.initializers.Constant(alpha_init),
            trainable=True,
            constraint=constraints.MaxNorm(1.0)
        )
    def call(self, x):
        x_ln = self.norm(x)
        h    = self.hg_down(x_ln)
        h    = self.ssm(h)
        h    = self.hg_up(h)
        return x + self.alpha * h

# ── 노이즈 추가 유틸 ───────────────────────────────────────────────────────
def add_diffusion_noise(x, stddev=0.05):
    return x + tf.random.normal(tf.shape(x), 0., stddev)

# ── 데이터 로딩 (EEG CSP + Eye feature 윈도우) ─────────────────────────────
def load_multimodal_features(csp_dir, eye_dir):
    X_eeg, X_eye, Y, S, F = [], [], [], [], []
    for fn in os.listdir(csp_dir):
        if not fn.endswith(".npy"): continue
        m = re.match(r'folder(\d+)_subject(\d+)_sample(\d+)_label(\d+)\.npy$', fn)
        if not m: continue
        f, subj, sample, label = m.groups()
        eeg = np.load(os.path.join(csp_dir, fn))  # (4, T, 8)
        eye_path = os.path.join(eye_dir,
            f"folder{f}_subject{subj}_sample{sample}.npy")
        if not os.path.exists(eye_path): continue
        eye = np.load(eye_path)                   # (31, T_eye)

        n_w = eeg.shape[1] // 200
        for i in range(n_w):
            win = eeg[:, i*200:(i+1)*200, :]      # (4,200,8)
            vec = eye[:, i] if i < eye.shape[1] else eye[:, -1]
            X_eeg.append(win)
            X_eye.append(vec)
            Y.append(int(label))
            S.append(int(subj))
            F.append(f"{fn}_win{i}")

    return (np.stack(X_eeg), np.stack(X_eye),
            np.array(Y), np.array(S), np.array(F))

# ── 주파수 CNN 브랜치 ───────────────────────────────────────────────────────
def build_freq_branch(input_shape=(200,8,4), noise_std=0.05):
    inp = layers.Input(shape=input_shape)
    x = layers.GaussianNoise(noise_std)(inp)
    x   = layers.Conv2D(16,(3,3),padding='same',activation='relu')(x)
    x   = layers.MaxPooling2D((2,2))(x)
    x   = layers.Conv2D(64,(3,3),padding='same',activation='relu')(x)
    x   = layers.GlobalAveragePooling2D()(x)         # (64,)
    x = layers.GaussianNoise(noise_std)(x)
    x   = DiffuSSMLayer(64,64)(x)
    return models.Model(inp, x, name="FreqBranchDiff")

# ── 채널 CNN 브랜치 ────────────────────────────────────────────────────────
def build_chan_branch(input_shape=(200,4,8), noise_std=0.05):
    inp = layers.Input(shape=input_shape)
    x = layers.GaussianNoise(noise_std)(inp)
    x   = layers.Conv2D(16,(3,3),padding='same',activation='relu')(x)
    x   = layers.MaxPooling2D((2,2))(x)
    x   = layers.Conv2D(64,(3,3),padding='same',activation='relu')(x)
    x   = layers.GlobalAveragePooling2D()(x)         # (64,)
    x = layers.GaussianNoise(noise_std)(x)
    x   = DiffuSSMLayer(64,64)(x)
    return models.Model(inp, x, name="ChanBranchDiff")

# ── Eye feature 브랜치 ─────────────────────────────────────────────────────
def build_eye_branch(input_shape=(31,), noise_std=0.05):
    inp = layers.Input(shape=input_shape)
    x = layers.GaussianNoise(noise_std)(inp)
    x   = layers.Dense(64, activation='swish')(x)
    x = layers.GaussianNoise(noise_std)(x)
    x   = DiffuSSMLayer(64,64)(x)
    return models.Model(inp, x, name="EyeBranchDiff")

def build_eye_image_branch(input_shape=(64,64,3), noise_std=0.05):
    inp = layers.Input(shape=input_shape)
    x = layers.GaussianNoise(noise_std)(inp)
    x = layers.Conv2D(16,(3,3),padding='same',activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(32,(3,3),padding='same',activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.GaussianNoise(noise_std)(x)
    x = DiffuSSMLayer(32,32)(x)
    return models.Model(inp,x,name="EyeImageBranchDiff")

# ── 크로스모달 퓨전 레이어 ─────────────────────────────────────────────────
class CrossModalFusion(layers.Layer):
    def __init__(self, embed_dim, num_heads=2, **kwargs):
        super().__init__(**kwargs)
        self.mha  = layers.MultiHeadAttention(num_heads=num_heads,
                                               key_dim=embed_dim//num_heads)
        self.norm = layers.LayerNormalization()
    def call(self, eeg_feat, eye_feat):
        q = tf.expand_dims(eeg_feat,1)  # (b,1,d)
        k = tf.expand_dims(eye_feat,1)
        v = k
        x = self.mha(query=q, key=k, value=v)
        x = tf.squeeze(x,1)
        return self.norm(x + eeg_feat)

def build_multimodal_model(noise_std=0.05, num_classes=4):
    # inputs
    eeg_in      = layers.Input((4,200,8), name="EEG_Input")
    eye_feat_in = layers.Input((31,),   name="EyeFeat_Input")
    eye_img_in  = layers.Input((64,64,3),name="EyeImg_Input")

    # EEG 빈도/채널 분기
    freq_x = layers.Permute((2,3,1))(eeg_in)  # (200,8,4)
    chan_x = layers.Permute((2,1,3))(eeg_in)  # (200,4,8)
    f_feat = build_freq_branch(noise_std=noise_std)(freq_x)
    c_feat = build_chan_branch(noise_std=noise_std)(chan_x)

    # Eye feature & Eye image 분기
    e_feat     = build_eye_branch(noise_std=noise_std)(eye_feat_in)
    e_img_feat = build_eye_image_branch(noise_std=noise_std)(eye_img_in)

    # 두 Eye 모달리티 합치기
    e_all = layers.Concatenate()([e_feat, e_img_feat])  # (b,96)
    e_all = layers.Dense(64,activation='relu')(e_all)
    e_all = layers.Dropout(0.2)(e_all)

    # EEG 합치기
    eeg_concat = layers.Concatenate()([f_feat, c_feat])  # (b,128)
    fused_eeg  = layers.Dense(64,activation='relu')(eeg_concat)
    fused_eeg  = layers.Dropout(0.2)(fused_eeg)

    # 게이트 퓨전 (EEG vs Eye_all)
    gate_logits = layers.Dense(2,name="fusion_gate")(
        layers.Concatenate()([fused_eeg, e_all])
    )
    gate = layers.Activation('softmax',name="fusion_weight")(gate_logits)
    w_eeg = layers.Lambda(lambda x: x[:, :1])(gate)
    w_eye = layers.Lambda(lambda x: x[:, 1:])(gate)

    fused = layers.Add()([
        layers.Multiply()([fused_eeg, w_eeg]),
        layers.Multiply()([e_all,      w_eye])
    ])
    fused = layers.LayerNormalization()(fused)

    # 분류기
    x = layers.Dense(64,activation='relu')(fused)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes,activation='softmax')(x)

    return models.Model([eeg_in, eye_feat_in, eye_img_in], out,
                        name="MultiModal_CrossDiffuSSM")


# ── 1) 학습 + 저장 (원본 단순 학습 로직으로 복원) ─────────────────────────
def train_and_save_model():
    # 1) EEG CSP + Eye-vector 불러오기
    X_eeg, X_eye, Y, _, _ = load_multimodal_features(
        "/home/bcml1/2025_EMOTION/SEED_IV/eeg_EM-CSP",
        "/home/bcml1/2025_EMOTION/SEED_IV/eye_feature_smooth_1s"
    )

    # 2) 더미 eye-image 생성 (모두 0)
    #    shape = (샘플 수, 64, 64, 3)
    X_img_dummy = np.zeros((X_eeg.shape[0], 64, 64, 3), dtype=np.float32)

    # 3) train/test split (EEG, eye-vector, eye-image, label)
    Xe_tr, Xe_te, Xy_tr, Xy_te, Xi_tr, Xi_te, y_tr, y_te = train_test_split(
        X_eeg, X_eye, X_img_dummy, Y,
        test_size=0.3, random_state=42, stratify=Y
    )

    # 4) Dataset 생성
    bs = 64
    train_ds = tf.data.Dataset.from_tensor_slices(
        ((Xe_tr, Xy_tr, Xi_tr), y_tr)
    ).shuffle(1000).batch(bs).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices(
        ((Xe_te, Xy_te, Xi_te), y_te)
    ).batch(bs).prefetch(tf.data.AUTOTUNE)

    # 5) 모델 빌드/컴파일
    model = build_multimodal_model(noise_std=0.05,
                                   num_classes=len(np.unique(Y)))
    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 6) 학습
    model.fit(
        train_ds,
        epochs=10,
        validation_data=test_ds,
        callbacks=[EarlyStopping("val_accuracy", patience=10, restore_best_weights=True)],
        verbose=2
    )

    # 7) 가중치 저장
    model.save_weights("_emotion_weights.weights.h5")
    print("모델을 _emotion_weights.weights.h5 에 저장했습니다.")

# ── 가중치 로드 및 예측 유틸 ─────────────────────────────────────────────
# ── 가중치 로드 (by_name로 일부 로딩) ───────────────────────────────────
def load_emotion_model(weights_path="_emotion_weights.weights.h5"):
    model = build_multimodal_model(noise_std=0.05, num_classes=4)
    # by_name 인자 제거
    model.load_weights(weights_path)
    return model

def classify_eog_sequence(eog_seq, model):
    preds = []
    for t in range(eog_seq.shape[1]):
        eye_vec   = eog_seq[:,t][None,:].astype(np.float32)
        dummy_eeg = np.zeros((1,4,200,8),dtype=np.float32)
        probs     = model.predict([dummy_eeg, eye_vec], verbose=0)
        preds.append(int(probs.argmax(axis=1)[0]))
    counts = np.bincount(preds, minlength=model.output_shape[-1])
    return counts.argmax(), counts

def find_working_camera(max_idx=3):
    for i in range(max_idx+1):
        cap = cv2.VideoCapture(i)
        if cap.isOpened(): return cap, i
    return None, None

# --- 동영상 풀재생 유틸 함수 ---

def play_video(window, media_path, camera_id_input=None, img_display_ms=1000):
    """
    media_path가 비디오(.mp4 등)면 기존대로 ffplay + OpenCV,
    이미지(.png/.jpg 등)면 OpenCV로 img_display_ms 밀리초 동안 화면에 띄움.
    """
    # 1) 이미지 파일일 때
    ext = os.path.splitext(media_path)[1].lower()
    if ext in (".png", ".jpg", ".jpeg", ".bmp"):
        img = cv2.imread(media_path)
        if img is None:
            print(f"⚠️ 이미지 로드 실패: {media_path}")
            return
        # 창 설정
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # 리사이즈해서 표시
        disp = cv2.resize(img, (1920, 1080))
        cv2.imshow(window, disp)
        cv2.waitKey(img_display_ms)           # 예: 1000ms = 1초 동안
        cv2.destroyWindow(window)
        return

    # 2) 비디오일 때 (기존 로직)
    camera_id = camera_id_input if camera_id_input is not None else CAMERA_ID
    # --- 1) 비디오 캡처 준비 ---
    cap = cv2.VideoCapture(media_path)
    cam = cv2.VideoCapture(camera_id)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_duration = 1.0 / fps

    # --- 2) audio 쓰레드와 동기용 Event/TS ---
    start_event = threading.Event()
    start_ts = {}
    def _play_audio(path):
        t0 = time.time()
        start_ts['t0'] = t0
        start_event.set()
        cmd = [
            FFPLAY_PATH,
            '-i', path,
            '-autoexit', '-nodisp',
            '-volume', '100', '-loglevel', 'warning'
        ]
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            print("⚠️ ffplay를 찾을 수 없습니다.")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ ffplay 실행 오류: {e}")

    audio_thread = threading.Thread(target=_play_audio, args=(media_path,), daemon=True)
    audio_thread.start()

    if not start_event.wait(timeout=1.0):
        start_time = time.time()
    else:
        start_time = start_ts['t0']

    # --- 4) OpenCV 창 설정 ---
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # --- 5) 비디오 프레임 루프 (싱크) ---
    frame_idx = 0
    while True:
        ret_v, frame = cap.read()
        if not ret_v:
            break
        ideal_t = start_time + frame_idx * frame_duration
        now = time.time()
        if ideal_t > now:
            time.sleep(ideal_t - now)

        frame = cv2.resize(frame, (1920, 1080))
        ret_c, cam_frame = cam.read()
        if ret_c:
            h, w = frame.shape[:2]
            small = cv2.resize(cam_frame, (w//6, h//6))
            frame[h - h//6 - 10 : h - 10, w - w//6 - 10 : w - 10] = small

        cv2.imshow(window, frame)
        frame_idx += 1
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cam.release()
    cv2.destroyWindow(window)
    
# --- Calibration with media timeline 수정 ---
def calibrate_with_media_timeline(model,
                                  intro_dict,
                                  calib_dict,
                                  calibrate_secs=CALIBRATE_SECONDS,
                                  window_secs=WINDOW_SECONDS,
                                  camera_id=CAMERA_ID):
    WIN       = "Calibrate"    # 지시 이미지 + inset
    LIVE_WIN  = "LiveCam"      # livecam + 테두리 + 피처 overlay
    INTRO_WIN = "IntroWindow"  # Intro 비디오 전용

    cap_cam = cv2.VideoCapture(camera_id)
    fps_cam = cap_cam.get(cv2.CAP_PROP_FPS) or 30.0
    win_f   = int(window_secs * fps_cam)
    stride  = max(1, win_f // 2)

    label2color = {
        0: (128,128,128),
        1: (255,0,0),
        2: (0,0,255),
        3: (0,255,255),
    }

    X_feat, X_img, y_cal = [], [], []

    # 1) 윈도우 한 번만 생성 & 풀스크린
    for w in (WIN, LIVE_WIN):
        cv2.namedWindow(w, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(w, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    for label in sorted(calib_dict.keys()):
        # 매 label마다 풀스크린 속성 재적용
        for w in (WIN, LIVE_WIN):
            cv2.setWindowProperty(w, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # --- 2) Intro 비디오 (전용 IntroWindow) ---
        cap_intro = cv2.VideoCapture(intro_dict[label])
        if cap_intro.isOpened():
            cv2.namedWindow(INTRO_WIN, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(INTRO_WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            fps_i = cap_intro.get(cv2.CAP_PROP_FPS) or 30.0
            delay = max(1, int(1000 / fps_i))
            while True:
                ret_i, fr_i = cap_intro.read()
                if not ret_i: break
                fr_i = cv2.resize(fr_i, (1920,1080))
                cv2.imshow(INTRO_WIN, fr_i)
                if cv2.waitKey(delay) == ord('q'):
                    cap_intro.release()
                    cap_cam.release()
                    cv2.destroyWindow(INTRO_WIN)
                    return model
            cap_intro.release()
            cv2.destroyWindow(INTRO_WIN)
        else:
            print(f"⚠️ Intro open failed: {intro_dict[label]}")

        # --- 3) Calibration 수집 루프 ---
        buf = deque(maxlen=win_f)
        cnt = 0
        t0  = time.time()
        while time.time() - t0 < calibrate_secs:
            ret, fr = cap_cam.read()
            if not ret:
                cv2.imshow(LIVE_WIN, np.zeros((1080,1920,3),np.uint8))
                if cv2.waitKey(1) == ord('q'):
                    cap_cam.release()
                    return model
                continue

            buf.append(fr); cnt += 1

            # LiveCam 창: 프레임 + 테두리
            disp = cv2.resize(fr, (1920,1080))
            cv2.rectangle(disp, (0,0), (1919,1079),
                          label2color[label], thickness=20)

            # stride마다 피처 뽑고 overlay & print
            if cnt >= stride and len(buf) == win_f:
                feats, img = your_extract_eog_features(list(buf), fps=fps_cam, window_secs=window_secs)
                X_feat.append(feats[:,0]); X_img.append(img); y_cal.append(label)
                cnt = 0
                # 터미널 출력
                print(f"[Calib] Label={label}, feats={feats.squeeze().tolist()}")
                # 화면 오버레이: 첫 5개
                feat5 = feats.squeeze()[:5]
                txt = "F:" + ",".join(f"{v:.2f}" for v in feat5)
                cv2.putText(disp, txt, (10,200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.imshow(LIVE_WIN, disp)

            # Calibrate 창: 배경+inset
            bg = cv2.imread(calib_dict[label])
            if bg is None:
                bg = np.zeros((1080,1920,3),np.uint8)
            bg = cv2.resize(bg, (1920,1080))
            small = cv2.resize(fr, (1920//8,1080//8))
            h,w = bg.shape[:2]
            bg[h-1080//8-10:h-10, w-1920//8-10:w-10] = small
            cv2.putText(bg, f"Calib Label={label}", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow(WIN, bg)

            if cv2.waitKey(1) == ord('q'):
                cap_cam.release()
                return model

        # 다음 label 전 짧게 대기
        time.sleep(0.2)

    # 4) 끝난 뒤 창 닫기
    cap_cam.release()
    cv2.destroyWindow(WIN)
    cv2.destroyWindow(LIVE_WIN)

    if not X_feat:
        raise RuntimeError("캘리브레이션 데이터가 없습니다.")

    # 나머지 fine-tuning 로직 (이전과 동일) …
    X_feat = np.stack(X_feat).astype(np.float32)
    X_img  = np.stack(X_img).astype(np.float32)
    y_cal  = np.array(y_cal, dtype=np.int32)

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.GaussianNoise):
            layer.stddev = 0.0     # 노이즈 제거
        if isinstance(layer, tf.keras.layers.Dropout):
            layer.rate = 0.0       # 드롭아웃 제거

    for layer in model.layers:
        layer.trainable = True
    for layer in model.layers:
        if layer.name.startswith("fusion_") or \
           layer.name.startswith("dense")   or \
           layer.name.startswith("EyeBranch") or \
           layer.name.startswith("EyeImageBranch"):
            layer.trainable = True

    model.compile(optimizer=optimizers.Adam(1e-4),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    debug_cb = DebugCallback(cap_cam=cv2.VideoCapture(camera_id),
                             win=WIN,
                             video_path=WAITING2_MEDIA)
    model.fit(
        x=[np.zeros((len(X_feat),4,200,8),dtype=np.float32),
           X_feat,
           X_img],
        y=y_cal,
        batch_size=8,
        epochs=300,
        callbacks=[debug_cb,
                   EarlyStopping('accuracy', patience=5000, restore_best_weights=True)],
        verbose=2
    )
    return model

# ── 실시간 추론 및 반복 루프 ────────────────────────────────────────────
def classify_eog_window(feats, eye_img, model):
    dummy_eeg = np.zeros((1,4,200,8),dtype=np.float32)
    feats_in  = feats[None,:]
    img_in    = eye_img[None,:,:,:]
    probs     = model.predict([dummy_eeg, feats_in, img_in], verbose=0)
    return int(probs.argmax(axis=1)[0])

def run_live_inference_timeline(weights_path="_emotion_weights.weights.h5"):
    model = load_emotion_model(weights_path)
    play_video("Intro", WAITING_MEDIA)

    model = calibrate_with_media_timeline(
        model,
        instructions_intro_media,
        instructions_media
    )

    while True:
        play_video("PredictIntro", PREDICT_MEDIA)

        cap = cv2.VideoCapture(CAMERA_ID)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        win_n = int(fps * WINDOW_SECONDS)

        cv2.namedWindow("Live",cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Live",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

        buf, start = [], time.time()
        while time.time()-start < 5.0:
            ret, frame = cap.read()
            if not ret: break
            buf.append(frame)
            cv2.imshow("Live",frame)
            if cv2.waitKey(1)==ord('q'):
                cap.release(); cv2.destroyAllWindows(); return

        cap.release(); cv2.destroyWindow("Live")

        if len(buf) < win_n:
            continue

        preds = []
        for i in range(0, len(buf)-win_n+1, win_n):
            feats, img = your_extract_eog_features(
                buf[i:i+win_n], fps=fps, window_secs=WINDOW_SECONDS
            )
            preds.append(classify_eog_window(feats[:,0], img, model))

        if not preds:
            continue

        label = Counter(preds).most_common(1)[0][0]
        play_video("Feedback", feedback_media[label], img_display_ms=3000)


if __name__ == "__main__":
    #train_and_save_model()
    run_live_inference_timeline()
