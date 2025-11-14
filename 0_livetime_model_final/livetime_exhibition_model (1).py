#!/usr/bin/env python3
import sys
import os
import random
import tempfile
import cv2
import numpy as np
import tensorflow as tf
import time
from collections import deque, Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import QUrl
import keras
keras.config.enable_unsafe_deserialization()#!/usr/bin/env python3

# Multimodal 1s model imports
# from multimodal_training2 import SplitModality
from Face_feature import extract_face_features_from_video
from EYE_feature import extract_features_from_video
from Mouth_feature import extract_features_from_video_mouth

# MicroExp3D 15s model imports
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Conv3D, MaxPool3D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.callbacks import Callback
import mediapipe as mp
from scipy.signal import medfilt
import threading

# 4 modal model imports
from PyQt6.QtWidgets import QSizePolicy  # 기존 import 옆에 추가
from tensorflow.keras.layers import Input, Dense, Concatenate, Multiply, Flatten, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.utils import register_keras_serializable
# --- Global Configuration ---
EMOTIONS         = ['neutral','sad','fear','happy']
FUSION_META_FILE = 'fusion_meta.npz'
# 1s model
SEG_SEC_1        = 1
CAL_HEAD_EPOCH   = 5
CAL_EPOCHS_1     = 50
CAL_BATCH_SIZE_1 = 8
CAL_LR_1         = 1e-5
# MODEL_FILE_1     = 'best_4_modality.keras'
MODEL_FILE_1     = 'best_0803.keras'
MEAN_FACE_FILE   = 'mean_face.npy'
STD_FACE_FILE    = 'std_face.npy'
MEAN_EYE_FILE    = 'mean_eye.npy'
STD_EYE_FILE     = 'std_eye.npy'
MEAN_RPPG_FILE   = 'mean_rppg.npy'
STD_RPPG_FILE    = 'std_rppg.npy'
MEAN_MOUTH_FILE  = 'mean_mouth.npy'
STD_MOUTH_FILE   = 'std_mouth.npy'
CALIB_1_PATH     = 'calib_data_1.npz'
# 15s model
SEG_SEC_15       = 15
FPS              = 30
BUFFER_SIZE_15   = SEG_SEC_15 * FPS
ONOFF_LEN        = FPS
CAL_EPOCHS_15    = 50
CAL_BATCH_15     = 4
CAL_LR_15        = 1e-5
MODEL_FILE_15    = 'final_microexp3dstcnn.keras'
CALIB_15_PATH    = 'calib_data_15.npz'
# FaceMesh & preprocessing params for 15s model
OUT_SIZE             = 112
PAD                  = 20
BRIGHTNESS_THRESHOLD = 40
N_LM                 = 478
FB_PARAMS = {
    'pyr_scale':0.5, 'levels':3, 'winsize':15,
    'iterations':3, 'poly_n':5, 'poly_sigma':1.2, 'flags':0
}

# MediaPipe Face Mesh for 15s
mp_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
last_bbox = None

# Utility
path_join = os.path.join

# --- 1s model feature extraction wrappers ---
def extract_face_features_from_frames_1s(frames, fps=30):
    with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as tmp:
        tmp_path = tmp.name
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w,h))
    for f in frames:
        out.write(f)
    out.release()
    feats, meta = extract_face_features_from_video(tmp_path)
    try: os.remove(tmp_path)
    except: pass
    return feats, meta


def extract_eye_features_from_frames_1s(frames, fps=30):
    with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as tmp:
        tmp_path = tmp.name
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w,h))
    for f in frames:
        out.write(f)
    out.release()
    feats, meta = extract_features_from_video(tmp_path)
    try: os.remove(tmp_path)
    except: pass
    return feats, meta

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_rppg_features_from_frames_1s(frames, fps=30):
    gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    if len(faces)==0:
        x,y,w,h = 0,0,frames[0].shape[1],frames[0].shape[0]
    else:
        x,y,w,h = faces[0]
    ph, pw = h//4, w//5
    T = len(frames)
    raw = np.zeros((20, T), float)
    for t, f in enumerate(frames):
        roi = f[y:y+h, x:x+w]
        for idx in range(20):
            r, c = divmod(idx, 5)
            patch = roi[r*ph:(r+1)*ph, c*pw:(c+1)*pw]
            raw[idx, t] = np.nanmean(patch[:,:,1])
    old_t = np.arange(T)
    new_t = np.linspace(0, T-1, 50)
    up = np.vstack([np.interp(new_t, old_t, raw[ch]) for ch in range(20)])
    return up, None

def extract_mouth_features_from_frames_1s(frames, fps=30):
    with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as tmp:
        tmp_path = tmp.name
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w,h))
    for f in frames:
        out.write(f)
    out.release()
    # segment_sec=1로 설정하여 1초 단위로 특징 추출
    feats, meta = extract_features_from_video_mouth(tmp_path, segment_sec=1, stride_sec=1)
    try: os.remove(tmp_path)
    except: pass
    return feats, meta

# --- 15s model preprocessing & model builder ---
def align_frame_15(frame):
    global last_bbox
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = mp_face.process(rgb)
    det = bool(res.multi_face_landmarks)
    if not det:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l,a,b = cv2.split(lab)
        cl = clahe.apply(l)
        merged = cv2.merge((cl,a,b))
        frame = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        frame = cv2.bilateralFilter(frame,5,75,75)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_face.process(rgb)
        det = bool(res.multi_face_landmarks)
    if not det and last_bbox is None:
        return np.zeros((OUT_SIZE,OUT_SIZE,3),np.uint8), False, np.full((N_LM,2),np.nan)
    if not det:
        x1,y1,x2,y2 = last_bbox
        crop = frame[y1:y2, x1:x2]
        aligned = cv2.resize(crop,(OUT_SIZE,OUT_SIZE))
        return aligned, False, np.full((N_LM,2),np.nan)
    lm = res.multi_face_landmarks[0].landmark
    xs = np.array([p.x*w for p in lm]); ys = np.array([p.y*h for p in lm])
    x1 = max(0,int(xs.min())-PAD); y1 = max(0,int(ys.min())-PAD)
    x2 = min(w,int(xs.max())+PAD); y2 = min(h,int(ys.max())+PAD)
    last_bbox = (x1,y1,x2,y2)
    crop = frame[y1:y2, x1:x2]
    aligned = cv2.resize(crop,(OUT_SIZE,OUT_SIZE))
    if aligned.mean()<BRIGHTNESS_THRESHOLD:
        return aligned, False, np.full((N_LM,2),np.nan)
    lm_arr = np.array([[p.x*w,p.y*h] for p in lm],dtype=np.float32)
    return aligned, True, lm_arr


def compute_combined_scores_15(frames, mask, lms,
                               n_base=5, w_pixel=1.0,
                               w_flow=0.5, w_landmark=0.5,
                               smooth_kernel=5):
    valid = np.where(mask)[0]
    if valid.size==0: return np.array([]), valid
    gray = np.dot(frames[valid][...,:3], [0.299,0.587,0.114])
    p = np.mean(np.abs(gray - ((gray[:n_base].mean(0)+gray[-n_base:].mean(0))/2)), axis=(1,2))
    f = np.zeros_like(p); prev = gray[0]
    for i,c in enumerate(gray[1:],1):
        flow = cv2.calcOpticalFlowFarneback(prev,c,None,**FB_PARAMS)
        f[i] = np.mean(np.linalg.norm(flow,axis=2)); prev = c
    lm_v = lms[valid]
    l = np.zeros_like(p); prev_l = lm_v[0]
    for i,cur in enumerate(lm_v[1:],1):
        l[i] = np.mean(np.linalg.norm(cur-prev_l,axis=1)); prev_l=cur
    def norm(x): return (x-x.min())/(x.max()-x.min()+1e-6)
    comb = w_pixel*norm(p) + w_flow*norm(f) + w_landmark*norm(l)
    if smooth_kernel>1 and smooth_kernel%2==1:
        comb = medfilt(comb, smooth_kernel)
    comb = np.convolve(comb, np.ones(5)/5, mode='same')
    return comb, valid


def find_top_segment_15(comb, valid, length=FPS):
    if comb.size==0: return 0,-1
    idx = np.argmax(comb); peak = valid[idx]; st = peak-length//2
    return int(st), int(peak)


def pad_or_crop_15(frames, start, length):
    T = frames.shape[0]; end = start+length
    s,e = max(start,0), min(end,T)
    clip = frames[s:e]
    pb,pa = max(0,-start), max(0,end-T)
    if pb or pa:
        clip = np.pad(clip, ((pb,pa),(0,0),(0,0),(0,0)), mode='constant')
    if clip.shape[0]>length: clip = clip[:length]
    if clip.shape[0]<length:
        clip = np.pad(clip, ((0,length-clip.shape[0]),(0,0),(0,0),(0,0)), mode='constant')
    return clip


def preprocess_clip_15(frames):
    arr = np.stack(frames).astype(np.float32)/255.0
    gray = np.dot(arr[...,:3], [0.299,0.587,0.114])
    out = np.zeros((gray.shape[0],64,64),dtype=np.float32)
    for i in range(gray.shape[0]):
        out[i] = cv2.resize(gray[i], (64,64))
    return out[...,np.newaxis]


@register_keras_serializable(package='custom_layers')
class SplitModality(tf.keras.layers.Layer):
    def __init__(self, idx, **kwargs):
        super().__init__(**kwargs)
        self.idx = idx
    def call(self, inputs):
        return tf.expand_dims(inputs[:, self.idx], -1)
    def get_config(self):
        cfg = super().get_config()
        cfg.update({'idx': self.idx})
        return cfg
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def build_gated4_model(num_classes,
                       seg_len_eye, feat_dim_eye,
                       seg_len_face, feat_dim_face,
                       seg_len_rppg, feat_dim_rppg,
                       seg_len_mouth, feat_dim_mouth,
                       eye_ext, face_ext, mouth_ext
):
    # (1) Inputs
    in_e = Input((seg_len_eye, feat_dim_eye),  name='eye_input')
    in_f = Input((seg_len_face, feat_dim_face),name='face_input')
    in_r = Input((seg_len_rppg, feat_dim_rppg),name='rppg_input')
    in_m = Input((seg_len_mouth, feat_dim_mouth),name='mouth_input')

    # 2) Extractor embeddings
    fe = eye_ext(in_e)
    ff = face_ext(in_f)
    fm = mouth_ext(in_m)

    # ───────────────────────────────────────────────────────────────
    # (A) rPPG 입력 전에 LayerNorm
    x_rn    = LayerNormalization(name="layer_norm")(in_r)
    xr_flat = Flatten(name='rppg_flat')(x_rn)
    pdim    = int(fe.shape[-1])
    feat_r  = Dense(pdim, activation='relu', name='rppg_proj')(xr_flat)

    # 4) Concatenate → Gate
    merged = Concatenate(name='concat')([fe, ff, fm, feat_r])

    # ───────────────────────────────────────────────────────────────
    # (B) 병합 직후 LayerNorm
    x_gn   = LayerNormalization(name="layer_norm_1")(merged)
    gates  = Dense(4, activation='softmax', name='gate')(x_gn)

    we = SplitModality(0, name='we')(gates)
    wf = SplitModality(1, name='wf')(gates)
    wm = SplitModality(2, name='wm')(gates)
    wr = SplitModality(3, name='wr')(gates)

    pe = Multiply(name='pe')([fe, we])
    pf = Multiply(name='pf')([ff, wf])
    pm = Multiply(name='pm')([fm, wm])
    pr = Multiply(name='pr')([feat_r, wr])

    # 6) 최종 fusion
    fused = Concatenate(name='fused')([pe, pf, pm, pr])

    # ───────────────────────────────────────────────────────────────
    # (C) 최종 fusion 뒤 LayerNorm + 두 단계 Dense
    x_dn = LayerNormalization(name="layer_norm_2")(fused)
    x1   = Dense(128, activation='relu', name='dense1')(x_dn)
    x2   = Dense(64,  activation='relu', name='dense2')(x1)
    out  = Dense(num_classes, activation='softmax', name='out')(x2)

    return Model([in_e, in_f, in_r, in_m], out, name='gated4')


def build_microexp3dstcnn(
    input_shape=(ONOFF_LEN*3,64,64,1),
    num_classes=len(EMOTIONS),
    backbone_trainable: bool = True,
    dropout_rate=0.3
):
    inp = Input(shape=input_shape, name='video_in')
    # ── “backbone” 부분 ──
    x = Conv3D(32,(15,3,3),strides=(1,1,1),
               padding='valid',activation='relu',
               name='backbone_conv1')(inp)
    x = MaxPool3D((3,3,3),strides=(3,3,3),
                  padding='valid',name='backbone_pool1')(x)
    x = Dropout(dropout_rate, name='backbone_dropout')(x)

    # ── head 부분 ──
    x = Flatten(name='head_flatten')(x)
    x = Dense(128, activation='relu', name='head_fc1')(x)
    x = Dropout(dropout_rate, name='head_dropout2')(x)
    out = Dense(num_classes, activation='softmax', name='head_output')(x)

    model = KerasModel(inp, out, name='MicroExp3DSTCNN')

    # backbone_trainable=False 일 때 freeze
    if not backbone_trainable:
        for layer in model.layers:
            if layer.name.startswith('backbone_'):
                layer.trainable = False
        # head 레이어만 학습 가능
        for layer in model.layers:
            if layer.name.startswith('head_'):
                layer.trainable = True

    return model


def load_micro_model(model_dir):
    model_path = path_join(model_dir, MODEL_FILE_15)
    if os.path.exists(model_path):
        return load_model(model_path, compile=False)
    return build_microexp3dstcnn(backbone_trainable=False)

# --- helper: 순차적으로 두 영상 읽는 캡처 ---
class SequentialCapture:
    """두 개 이상의 비디오를 순차적으로 하나의 스트림처럼 읽음."""
    def __init__(self, paths):
        self.paths = list(paths)
        self.cap = None
        self._open_next()

    def _open_next(self):
        if self.cap:
            self.cap.release()
        if not self.paths:
            self.cap = None
            return
        next_path = self.paths.pop(0)
        self.cap = cv2.VideoCapture(next_path)

    def read(self):
        if self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        if not ret:
            self._open_next()
            if self.cap is None:
                return False, None
            ret, frame = self.cap.read()
        return ret, frame

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()

    def get_fps(self):
        if self.cap:
            return self.cap.get(cv2.CAP_PROP_FPS)
        return None

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None


# --- helper: 1초 모델 입력 안전 추출 ---
def safe_extract_1s_features(frames, fps, me, se, mf, sf, mm, sm, mr, sr):
    f_feats, _ = extract_face_features_from_frames_1s(frames, fps)
    e_feats, _ = extract_eye_features_from_frames_1s(frames, fps)
    m_feats, _ = extract_mouth_features_from_frames_1s(frames, fps)
    r_feats, _ = extract_rppg_features_from_frames_1s(frames, fps)

    if f_feats.size == 0 or e_feats.size == 0 or m_feats.size == 0 or r_feats.size == 0:
        return None

    try:
        inp_eye = ((e_feats[:,0] - me) / se)[None, None, :]
        inp_face = ((f_feats[:,0] - mf) / sf)[None, None, :]
        inp_mouth = ((m_feats[:,0] - mm) / sm)[None, None, :]
        inp_rppg = ((r_feats.T - mr) / sr)[None, :, :]
    except Exception:
        return None
    return [inp_eye, inp_face, inp_rppg, inp_mouth]

# Epoch callback for both models
class QtEpochCallback(tf.keras.callbacks.Callback):
    def __init__(self, thread):
        super().__init__()
        self.thread = thread
    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        acc = logs.get('accuracy')
        self.thread.progress.emit(f"[Epoch {epoch+1}] loss:{loss:.4f}, acc:{acc:.4f}")

# WorkerThread handles both calibrate & evaluate for 1s and 15s
class WorkerThread(QtCore.QThread):
    progress      = QtCore.pyqtSignal(str)
    finished      = QtCore.pyqtSignal(str)
    video_to_play = QtCore.pyqtSignal(str)

    def __init__(self, mode, data_dir, model_dir):
        super().__init__()
        self.mode = mode; self.data_dir = data_dir; self.model_dir = model_dir

    def run(self):
        self.progress.emit(f"[WorkerThread] start mode={self.mode}")
        try:
            if self.mode == 'calibrate':
                self.calibrate()
                res = 'Calibration complete for both models'
            else:
                self.evaluate()
                res = 'Evaluation complete for both models'
        except Exception as e:
            # 오류도 로그에 남기고
            self.progress.emit(f"[WorkerThread] ERROR: {e}")
            res = f"Error: {e}"
        finally:
            self.finished.emit(res)

    def calibrate(self):
        # 디버그: calibrate 진입
        self.progress.emit("[Calibration] 시작")
        # Load existing calibration if available
        os.makedirs(self.model_dir, exist_ok=True)
        calib1_path = path_join(self.model_dir, CALIB_1_PATH)
        if os.path.exists(calib1_path):
            data1 = np.load(calib1_path)
            Xf_old, Xe_old, Xm_old, Xr_old, y1_old = data1['Xf'], data1['Xe'], data1.get('Xm'), data1['Xr'], data1['y']
            self.progress.emit(f"[Calibration] Loaded {len(y1_old)} old 1s samples")
            # --- OLD DATA TEST TRAINING ---
            if Xf_old.size and Xe_old.size and Xm_old.size and Xr_old.size and y1_old.size:
                self.progress.emit("[Calibration] Testing training on OLD 1s calibration data")
                # 정규화 통계
                mf_o, sf_o = Xf_old.mean(0), Xf_old.std(0) + 1e-6
                me_o, se_o = Xe_old.mean(0), Xe_old.std(0) + 1e-6
                mm_o, sm_o = Xm_old.mean(0), Xm_old.std(0) + 1e-6
                fr_flat = Xr_old.reshape(-1, Xr_old.shape[2])
                mr_o, sr_o = fr_flat.mean(0), fr_flat.std(0) + 1e-6

                # 모델 빌드 (1s)
                from Eye_model import build_eye_model
                from Face_model import build_face_model
                from Mouth_model import build_mouth_model
                eye_base_o   = build_eye_model(num_classes=4, seq_length=1, feature_dim=me_o.shape[0],backbone_trainable=False)
                face_base_o  = build_face_model(num_classes=4, seq_length=1, feature_dim=mf_o.shape[0],backbone_trainable=False)
                mouth_base_o = build_mouth_model(num_classes=4, seq_length=1, feature_dim=mm_o.shape[0],backbone_trainable=False)
                eye_ext_o    = Model(eye_base_o.input, eye_base_o.layers[-2].output)
                face_ext_o   = Model(face_base_o.input, face_base_o.layers[-2].output)
                mouth_ext_o  = Model(mouth_base_o.input, mouth_base_o.layers[-2].output)

                # Gated4 모델 생성
                model_o = build_gated4_model(
                    num_classes=4,
                    seg_len_eye=1,    feat_dim_eye=me_o.shape[0],
                    seg_len_face=1,   feat_dim_face=mf_o.shape[0],
                    seg_len_rppg=50,  feat_dim_rppg=mr_o.shape[0],
                    seg_len_mouth=1,  feat_dim_mouth=mm_o.shape[0],
                    eye_ext=eye_ext_o, face_ext=face_ext_o, mouth_ext=mouth_ext_o
                )
                model_o.compile(
                    optimizer=tf.keras.optimizers.Adam(CAL_LR_1),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                # 한 epoch 정도 빠르게 돌려보기
                model_o.fit(
                    [
                        ((Xe_old - me_o) / se_o)[:, None, :],
                        ((Xf_old - mf_o) / sf_o)[:, None, :],
                        ((Xr_old - mr_o) / sr_o),
                        ((Xm_old - mm_o) / sm_o)[:, None, :]
                    ],
                    y1_old,
                    epochs=5,
                    batch_size=CAL_BATCH_SIZE_1,
                    callbacks=[QtEpochCallback(self)],
                    verbose=1
                )
                model_o.save_weights(path_join(self.model_dir, 'old_best_0803.weights.h5'))
                self.progress.emit("[Calibration] OLD data test training completed")
        else:
            Xf_old = Xe_old = Xm_old = Xr_old = y1_old = None
        calib15_path = path_join(self.model_dir, CALIB_15_PATH)
        if os.path.exists(calib15_path):
            data2 = np.load(calib15_path)
            X15_old, y15_old = data2['X'], data2['y']
            self.progress.emit(f"[Calibration] Loaded {len(y15_old)} old 15s samples")
            # --- OLD DATA TEST TRAINING for 15s model (5 epochs) ---
            if X15_old.size and y15_old.size:
                self.progress.emit("[Calibration] Testing training on OLD 15s calibration data")
                # build head-only 15s model
                model15_o = load_micro_model(self.model_dir)
                model15_o.compile(
                    optimizer=tf.keras.optimizers.Adam(1e-4),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                model15_o.fit(
                    X15_old, y15_old,
                    epochs=5,
                    batch_size=CAL_BATCH_15,
                    callbacks=[QtEpochCallback(self)],
                    verbose=1
                )
                # (원한다면 weights 저장)
                model15_o.save(path_join(self.model_dir, 'old_microexp3dstcnn_0803.keras'))
                self.progress.emit("[Calibration] OLD 15s data test training completed")
        else:
            X15_old = y15_old = None
        

        # Prepare new calibration lists
        Xf_new, Xe_new, Xm_new, Xr_new, y1_new = [], [], [], [], []
        X15_new, y15_new = [], []

        # Ensure tempvideo folder exists
        temp_folder = path_join(os.getcwd(), 'tempvideo')
        os.makedirs(temp_folder, exist_ok=True)

        # Open camera for recording
        cap_c = cv2.VideoCapture(0)
        if not cap_c.isOpened():
            raise RuntimeError("Camera failed to open for calibration recording")
        ret, frame0 = cap_c.read()
        if not ret:
            cap_c.release()
            raise RuntimeError("Failed to read frame from camera")
        h, w = frame0.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        # Fixed-duration playback and recording per emotion
        for label, emo in enumerate(EMOTIONS):
            emo_dir = path_join(self.data_dir, emo)
            vids = [f for f in os.listdir(emo_dir) if f.lower().endswith(('.mp4','.avi'))]
            if not vids:
                self.progress.emit(f"[Calibration] No videos for {emo}, skipping")
                continue
            vid_path = path_join(emo_dir, random.choice(vids))
            # Start playback
            self.video_to_play.emit(vid_path)
            time.sleep(0.5)
            # Prepare writer
            out_path = path_join(temp_folder, f"{emo}.avi")
            out_cam = cv2.VideoWriter(out_path, fourcc, FPS, (w, h))
            end_time = time.time() + 60
            while time.time() < end_time:
                rv_c, cam_frame = cap_c.read()
                if rv_c:
                    out_cam.write(cam_frame)
            out_cam.release()
            # Stop playback
            self.video_to_play.emit("")
            time.sleep(0.2)
            self.progress.emit(f"[Calibration] Recorded 60s for {emo} at {out_path}")
        cap_c.release()

        # Extract segments for each emotion
        for label, emo in enumerate(EMOTIONS):
            rec_path = path_join(temp_folder, f"{emo}.avi")
            cap_rec = cv2.VideoCapture(rec_path)
            recorded = []
            while True:
                ret, frm = cap_rec.read()
                if not ret:
                    break
                recorded.append(frm)
            cap_rec.release()
            total = len(recorded)
            # 1s segments
            seg1 = int(SEG_SEC_1 * FPS)
            for start in range(0, total - seg1 + 1, seg1):
                seg = recorded[start:start+seg1]
                print(f"[{emo}-{start}] >> Extracting Face...")
                f_feats,_ = extract_face_features_from_frames_1s(seg, FPS)
                print(f"[{emo}-{start}] << Face done.")
                print(f"[{emo}-{start}] >> Extracting Eye…")
                e_feats,_ = extract_eye_features_from_frames_1s(seg, FPS)
                print(f"[{emo}-{start}] << Eye done.")
                print(f"[{emo}-{start}] >> Extracting Mouth…")
                m_feats,_ = extract_mouth_features_from_frames_1s(seg, FPS)
                print(f"[{emo}-{start}] << Mouth done.")
                print(f"[{emo}-{start}] >> Extracting rPPG…")
                r_feats,_ = extract_rppg_features_from_frames_1s(seg, FPS)
                print(f"[{emo}-{start}] << rPPG done.")
                
                if f_feats.size and e_feats.size and m_feats.size and r_feats.size:
                    Xf_new.append(f_feats[:,0])
                    Xe_new.append(e_feats[:,0])
                    Xm_new.append(m_feats[:,0])
                    Xr_new.append(r_feats.T)
                    y1_new.append(label)
                else:
                    self.progress.emit(f"[Calibration] Skipped 1s seg {start}-{start+seg1} for {emo}")

            # 15s segments
            seg15 = int(SEG_SEC_15 * FPS)
            for start in range(0, total - seg15 + 1, seg15):
                clip = recorded[start:start+seg15]
                arr = np.stack(clip)
                comb, valid = compute_combined_scores_15(arr, np.ones(len(arr),bool), np.zeros((len(arr),N_LM,2)))
                if comb.size == 0:
                    self.progress.emit(f"[Calibration] Skipped 15s seg {start}-{start+seg15} for {emo}")
                    continue
                st, _ = find_top_segment_15(comb, valid, length=FPS)
                window = pad_or_crop_15(arr, st, ONOFF_LEN * 3)
                sample = preprocess_clip_15(window)[None,...]
                X15_new.append(sample[0])
                y15_new.append(label)

        self.progress.emit(f"Combine start")

        # Combine old and new data
        def _combine(old, new, empty_shape):
            new_exists = bool(new)
            old_exists = old is not None and old.size > 0
            if new_exists:
                new_stack = np.stack(new)
                if old_exists:
                    if old.ndim == new_stack.ndim:
                        return np.concatenate([old, new_stack], axis=0)
                    else:
                        print(f"Warning: Dimension mismatch between old({old.shape}) and new({new_stack.shape}) data. Using new data only.")
                        return new_stack
                return new_stack
            return old if old_exists else np.empty(empty_shape)

        Xf = _combine(Xf_old, Xf_new, (0, 68))
        Xe = _combine(Xe_old, Xe_new, (0, 30))
        Xm = _combine(Xm_old, Xm_new, (0, 20))
        Xr = _combine(Xr_old, Xr_new, (0, 50, 20))
        y1 = _combine(y1_old, y1_new, (0,))

        X15 = _combine(X15_old, X15_new, (0, ONOFF_LEN*3, 64, 64, 1))
        y15 = _combine(y15_old, y15_new, (0,))

        # 로그 shape
        self.progress.emit(f"[Calibration] shapes: Xf {Xf.shape}, Xe {Xe.shape}, Xm {Xm.shape}, Xr {Xr.shape}, y1 {y1.shape}")
        self.progress.emit(f"[Calibration] shapes: X15 {X15.shape}, y15 {y15.shape}")

        # Save calibration sets
        os.makedirs(self.model_dir, exist_ok=True)
        np.savez(path_join(self.model_dir, CALIB_1_PATH), Xf=Xf, Xe=Xe, Xm=Xm, Xr=Xr, y=y1)
        np.savez(path_join(self.model_dir, CALIB_15_PATH), X=X15, y=y15)
        self.progress.emit(f"[Calibration] Saved calib data: {CALIB_1_PATH}, {CALIB_15_PATH}")

        # --- 정규화 통계 계산 및 저장 (이 시점에 me/mf 등 정의) ---
        mf, sf = Xf.mean(0), Xf.std(0) + 1e-6
        me, se = Xe.mean(0), Xe.std(0) + 1e-6
        mm, sm = Xm.mean(0), Xm.std(0) + 1e-6
        fr = Xr.reshape(-1, Xr.shape[2]) if Xr.size else np.empty((0, Xr.shape[2]))
        mr, sr = fr.mean(0), fr.std(0) + 1e-6

        np.save(path_join(self.model_dir, MEAN_FACE_FILE), mf)
        np.save(path_join(self.model_dir, STD_FACE_FILE), sf)
        np.save(path_join(self.model_dir, MEAN_EYE_FILE), me)
        np.save(path_join(self.model_dir, STD_EYE_FILE), se)
        np.save(path_join(self.model_dir, MEAN_MOUTH_FILE), mm)
        np.save(path_join(self.model_dir, STD_MOUTH_FILE), sm)
        np.save(path_join(self.model_dir, MEAN_RPPG_FILE), mr)
        np.save(path_join(self.model_dir, STD_RPPG_FILE), sr)

        # --- 1s 모델 구성 및 fine-tune ---
        self.progress.emit("[Calibration] Training 1s model...")
        num_classes = 4
        from Eye_model import build_eye_model
        from Face_model import build_face_model
        from Mouth_model import build_mouth_model

        eye_base = build_eye_model(num_classes=num_classes, seq_length=1, feature_dim=me.shape[0],backbone_trainable=False)
        face_base = build_face_model(num_classes=num_classes, seq_length=1, feature_dim=mf.shape[0],backbone_trainable=False)
        mouth_base = build_mouth_model(num_classes=num_classes, seq_length=1, feature_dim=mm.shape[0],backbone_trainable=False)

        eye_ext = Model(eye_base.input, eye_base.layers[-2].output)
        face_ext = Model(face_base.input, face_base.layers[-2].output)
        mouth_ext = Model(mouth_base.input, mouth_base.layers[-2].output)

        

        weights_path = path_join(self.model_dir, 'best_0803.weights.h5')
        self.progress.emit("[Calibration] 1s model TL phase 1 (head-only)...")
        model1 = build_gated4_model(
            num_classes=num_classes,
            seg_len_eye=1, feat_dim_eye=me.shape[0],
            seg_len_face=1, feat_dim_face=mf.shape[0],
            seg_len_rppg=50, feat_dim_rppg=mr.shape[0],
            seg_len_mouth=1, feat_dim_mouth=mm.shape[0],
            eye_ext=eye_ext, face_ext=face_ext, mouth_ext=mouth_ext
        )
        # model1.load_weights(weights_path)  # by_name, skip_mismatch 없이
        # 가중치 로드: 이름(by_name)으로 매칭, 변수 개수 다르면(skip_mismatch) 그냥 무시
        try:
            model1.load_weights(weights_path, skip_mismatch=True)
            self.progress.emit("[Calibration] 1s weights loaded (skip_mismatch=True)")
            #self.progress.emit("[Calibration] 1s weights loaded (mismatches skipped)")
        except Exception as e:
            self.progress.emit(f"[Calibration] Warning: 1s load_weights skipped: {e}")
        model1.compile(optimizer=tf.keras.optimizers.Adam(CAL_LR_1),
                    loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        

        model1.fit([
            ((Xe - me) / se)[:, None, :],
            ((Xf - mf) / sf)[:, None, :],
            ((Xr - mr) / sr),
            ((Xm - mm) / sm)[:, None, :]
        ], y1, epochs=CAL_HEAD_EPOCH, batch_size=CAL_BATCH_SIZE_1,
            callbacks=[QtEpochCallback(self)], verbose=0)

        for layer in face_base.layers[-20:]:
            layer.trainable = True
        for layer in eye_base.layers[-10:]:
            layer.trainable = True
        for layer in mouth_base.layers[-10:]:
            layer.trainable = True
        self.progress.emit("[Calibration] 1s model TL phase 2 (fine-tune)...")
        model1.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),  # 낮은 LR
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        model1.fit(
            [
                ((Xe - me) / se)[:, None, :],
                ((Xf - mf) / sf)[:, None, :],
                ((Xr - mr) / sr),
                ((Xm - mm) / sm)[:, None, :]
            ],
            y1,
            epochs=CAL_EPOCHS_1,       # 본래 설정대로
            batch_size=CAL_BATCH_SIZE_1,
            callbacks=[QtEpochCallback(self)],
            verbose=0
        )

        model1.save_weights(path_join(self.model_dir, 'calib_best_0803.weights.h5'))
        #model1.save(path_join(self.model_dir, MODEL_FILE_1), save_format='keras')
        self.progress.emit("[Calibration] 1s model trained")

        # --- 15s 모델 ---
        if X15.size:
            self.progress.emit("[Calibration] 15s model TL phase 1 (head-only)...")

            # Phase1: backbone frozen, head만 학습
            model2 = load_micro_model(self.model_dir)
            model2.compile(
                optimizer=tf.keras.optimizers.Adam(1e-4),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            model2.fit(
                X15, y15,
                epochs=CAL_HEAD_EPOCH,
                batch_size=CAL_BATCH_15,
                callbacks=[QtEpochCallback(self)],
                verbose=0
            )

            self.progress.emit("[Calibration] 15s model TL phase 2 (fine-tune)...")

            # Phase2: backbone unfreeze + 낮은 LR 로 fine-tune
            # (model2.layers 가 이미 backbone frozen 상태)
            for layer in model2.layers:
                # head는 이미 trainable=True, backbone은 False → 전체 trainable=True 로 변경
                layer.trainable = True

            model2.compile(
                optimizer=tf.keras.optimizers.Adam(1e-5),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            model2.fit(
                X15, y15,
                epochs=CAL_EPOCHS_15,
                batch_size=CAL_BATCH_15,
                callbacks=[QtEpochCallback(self)],
                verbose=0
            )

            model2.save(path_join(self.model_dir, MODEL_FILE_15))
            self.progress.emit("[Calibration] 15s model fine-tuned")
        else:
            self.progress.emit("[Calibration] No 15s data to train")
        
        # ========== [FUSION] 두 모델 정확도 측정 → 가중치 저장 ==========
        # 1) 1초 모델 calibration set 정확도
        try:
            p1_cal = model1.predict(
                [
                    ((Xe - me) / se)[:, None, :],
                    ((Xf - mf) / sf)[:, None, :],
                    ((Xr - mr) / sr),
                    ((Xm - mm) / sm)[:, None, :]
                ],
                batch_size=CAL_BATCH_SIZE_1, verbose=0
            )
            acc1 = float((p1_cal.argmax(1) == y1).mean()) if y1.size else 0.0
        except Exception as e:
            self.progress.emit(f"[Calibration][FUSION] 1s acc calc fail: {e}")
            acc1 = 0.0

        # 2) 15초 모델 calibration set 정확도
        acc15 = 0.0
        if X15.size:
            try:
                p15_cal = model2.predict(X15, batch_size=CAL_BATCH_15, verbose=0)
                acc15 = float((p15_cal.argmax(1) == y15).mean()) if y15.size else 0.0
            except Exception as e:
                self.progress.emit(f"[Calibration][FUSION] 15s acc calc fail: {e}")
                acc15 = 0.0

        # 3) 정확도를 기반으로 가중치 산출 (합이 1이 되도록 정규화)
        eps = 1e-6
        s = acc1 + acc15 + eps
        w1 = acc1 / s
        w15 = acc15 / s

        np.savez(path_join(self.model_dir, FUSION_META_FILE), w1=w1, w15=w15, acc1=acc1, acc15=acc15)
        self.progress.emit(f"[Calibration][FUSION] weights saved: w1={w1:.3f}, w15={w15:.3f} (acc1={acc1:.3f}, acc15={acc15:.3f})")

    def evaluate(self):
        # Load models and stats
        # Load normalization stats
        mf, sf = np.load(path_join(self.model_dir, MEAN_FACE_FILE)), np.load(path_join(self.model_dir, STD_FACE_FILE))
        me, se = np.load(path_join(self.model_dir, MEAN_EYE_FILE)), np.load(path_join(self.model_dir, STD_EYE_FILE))
        mr, sr = np.load(path_join(self.model_dir, MEAN_RPPG_FILE)), np.load(path_join(self.model_dir, STD_RPPG_FILE))
        mm, sm = np.load(path_join(self.model_dir, MEAN_MOUTH_FILE)), np.load(path_join(self.model_dir, STD_MOUTH_FILE))

        # gated4 모델을 학습할 때와 동일하게 재구성한 뒤 weights-only 로드
        from Eye_model import build_eye_model
        from Face_model import build_face_model
        from Mouth_model import build_mouth_model

        # 감정 라벨 수 (학습 때와 같아야 함). 예제에서는 4개.
        num_classes = 4  

        # extractor base 모델 생성 (feature_dim은 mean/std 길이로 추론)
        eye_base = build_eye_model(num_classes=num_classes, seq_length=1, feature_dim=me.shape[0])
        face_base = build_face_model(num_classes=num_classes, seq_length=1, feature_dim=mf.shape[0])
        mouth_base = build_mouth_model(num_classes=num_classes, seq_length=1, feature_dim=mm.shape[0])

        # embedding extractor로 래핑
        eye_ext = Model(eye_base.input, eye_base.layers[-2].output)
        face_ext = Model(face_base.input, face_base.layers[-2].output)
        mouth_ext = Model(mouth_base.input, mouth_base.layers[-2].output)

        # gated4 구조 생성 (rPPG feature dim은 mr.shape[0], seg_len_rppg=50)
        model1 = build_gated4_model(
            num_classes=num_classes,
            seg_len_eye=1, feat_dim_eye=me.shape[0],
            seg_len_face=1, feat_dim_face=mf.shape[0],
            seg_len_rppg=50, feat_dim_rppg=mr.shape[0],
            seg_len_mouth=1, feat_dim_mouth=mm.shape[0],
            eye_ext=eye_ext, face_ext=face_ext, mouth_ext=mouth_ext
        )
        model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # 학습된 weights-only 체크포인트 로드
        #model1.load_weights(path_join(self.model_dir, 'calib_best_0803.weights.h5'))
        wpath = path_join(self.model_dir, 'calib_best_0803.weights.h5')
        if os.path.exists(wpath):
            try:
                model1.load_weights(wpath, skip_mismatch=True)
                self.progress.emit("[Evaluate] calib weights loaded (mismatches skipped)")
            except TypeError:
                # (환경에 따라 by_name/skip_mismatch 인자 미지원일 수 있어 안전 장치)
                model1.load_weights(wpath)
                self.progress.emit("[Evaluate] calib weights loaded (no by_name/skip_mismatch)")
        else:
            self.progress.emit("[Evaluate] Warning: calib weights not found; using random init")
        model2 = load_micro_model(self.model_dir)
        model2.compile()

        # 같은 감정 폴더에서 두 영상 선택
        emo = 'fear' # 평가할 감정 랜덤선택 또는 임시: 'fear' 고정
        emo_dir = path_join(self.data_dir, emo)
        vids = [f for f in os.listdir(emo_dir) if f.lower().endswith(('.mp4', '.avi'))]
        if not vids:
            self.progress.emit(f"[Evaluate] No videos for {emo}")
            return
        if len(vids) >= 2:
            v1, v2 = random.sample(vids, 2)
        else:
            v1 = v2 = vids[0]
        path1 = path_join(emo_dir, v1)
        path2 = path_join(emo_dir, v2)

        # GUI 재생: 먼저 첫 영상, 뒤이어 두번째 영상으로 자동 전환 (백그라운드에서 시간 기반)
        self.video_to_play.emit(path1)

        def switch_to_second():
            # 첫 영상 길이 계산
            cap_tmp = cv2.VideoCapture(path1)
            fps1 = cap_tmp.get(cv2.CAP_PROP_FPS) or FPS
            frame_count1 = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_tmp.release()
            duration1 = frame_count1 / fps1 if fps1 > 0 else 0
            # 살짝 여유 두기
            time.sleep(duration1 + 0.2)
            self.video_to_play.emit(path2)

        threading.Thread(target=switch_to_second, daemon=True).start()

        # 내부 스트림: 두 영상을 순차적으로 읽는 캡처
        seq_cap = SequentialCapture([path1, path2])

        # 카메라 열기
        cap_c = cv2.VideoCapture(0)
        if not seq_cap.is_opened() or not cap_c.isOpened():
            self.progress.emit("[Evaluate] Video or camera failed to open")
            seq_cap.release()
            return

        fps = seq_cap.get_fps() or FPS
        votes1, votes2 = [], []
        # [FUSION] 확률 평균을 위해 확률을 모을 리스트
        probs_1s_a = []   # 1초 - 첫 15회
        probs_15_a = []   # 15초 - 첫 1회
        probs_1s_b = []   # 1초 - 추가 15회
        probs_15_b = []   # 15초 - 두 번째 1회
        # 1) 1초 모델 15회 예측
        for i in range(15):
            buf1 = deque(maxlen=int(fps * SEG_SEC_1))
            start_wait = time.time()
            while len(buf1) < buf1.maxlen:
                rv_v, _ = seq_cap.read()
                rv_c, frame = cap_c.read()
                if not rv_v or not rv_c:
                    if time.time() - start_wait > 5:
                        break
                    continue
                buf1.append(frame)
            if len(buf1) < buf1.maxlen:
                self.progress.emit(f"[Evaluate] Skipped 1s #{i+1} due to insufficient frames")
                continue
            inputs = safe_extract_1s_features(list(buf1), fps, me, se, mf, sf, mm, sm, mr, sr)
            if inputs is None:
                self.progress.emit(f"[Evaluate] Skipped 1s #{i+1} due to failed feature extraction")
                continue
            pred1 = model1.predict(inputs, verbose=0)
            label1 = pred1.argmax()
            votes1.append(label1)
            probs_1s_a.append(pred1[0])   # [FUSION] 확률 추가 저장
            self.progress.emit(f"[Evaluate] 1s #{i+1}: {EMOTIONS[label1]}")

        # 2) 15초 모델 1회 예측
        buf15 = deque(maxlen=BUFFER_SIZE_15)
        start_wait = time.time()
        while len(buf15) < buf15.maxlen:
            rv_v, _ = seq_cap.read()
            rv_c, frame = cap_c.read()
            if not rv_v or not rv_c:
                if time.time() - start_wait > 5:
                    break
                continue
            buf15.append(frame)
        if len(buf15) == BUFFER_SIZE_15:
            arr15 = np.stack(buf15)
            comb, valid = compute_combined_scores_15(arr15, np.ones(len(arr15), bool),
                                                    np.zeros((len(arr15), N_LM, 2)))
            st, _ = find_top_segment_15(comb, valid, length=FPS)
            clip15 = pad_or_crop_15(arr15, st, ONOFF_LEN * 3)
            p15 = model2.predict(preprocess_clip_15(clip15)[None, ...], verbose=0)
            votes2.append(p15.argmax())
            probs_15_a.append(p15[0])    # [FUSION] 확률 추가 저장
            self.progress.emit(f"[Evaluate] 15s #1: {EMOTIONS[p15.argmax()]}")
        else:
            self.progress.emit("[Evaluate] Skipped 15s #1 due to insufficient frames")

        # 3) 1초 모델 추가 15회 예측
        for i in range(15):
            buf1 = deque(maxlen=int(fps * SEG_SEC_1))
            start_wait = time.time()
            while len(buf1) < buf1.maxlen:
                rv_v, _ = seq_cap.read()
                rv_c, frame = cap_c.read()
                if not rv_v or not rv_c:
                    if time.time() - start_wait > 5:
                        break
                    continue
                buf1.append(frame)
            if len(buf1) < buf1.maxlen:
                self.progress.emit(f"[Evaluate] Skipped extra 1s #{i+1} due to insufficient frames")
                continue
            inputs = safe_extract_1s_features(list(buf1), fps, me, se, mf, sf, mm, sm, mr, sr)
            if inputs is None:
                self.progress.emit(f"[Evaluate] Skipped extra 1s #{i+1} due to failed feature extraction")
                continue
            pred1 = model1.predict(inputs, verbose=0)
            label1 = pred1.argmax()
            votes1.append(label1)
            probs_1s_b.append(pred1[0])   # ★ 1초 블록 B
            self.progress.emit(f"[Evaluate] 1s extra #{i+1}: {EMOTIONS[label1]}")

        # 4) 15초 모델 두 번째 예측
        buf15 = deque(maxlen=BUFFER_SIZE_15)
        start_wait = time.time()
        while len(buf15) < buf15.maxlen:
            rv_v, _ = seq_cap.read()
            rv_c, frame = cap_c.read()
            if not rv_v or not rv_c:
                if time.time() - start_wait > 5:
                    break
                continue
            buf15.append(frame)
        if len(buf15) == BUFFER_SIZE_15:
            arr15 = np.stack(buf15)
            comb, valid = compute_combined_scores_15(arr15, np.ones(len(arr15), bool),
                                                    np.zeros((len(arr15), N_LM, 2)))
            st, _ = find_top_segment_15(comb, valid, length=FPS)
            clip15 = pad_or_crop_15(arr15, st, ONOFF_LEN * 3)
            p15 = model2.predict(preprocess_clip_15(clip15)[None, ...], verbose=0)
            votes2.append(p15.argmax())
            probs_15_b.append(p15[0])     # ★ 15초 블록 B
            self.progress.emit(f"[Evaluate] 15s #2: {EMOTIONS[p15.argmax()]}")
        else:
            self.progress.emit("[Evaluate] Skipped 15s #2 due to insufficient frames")

        seq_cap.release()
        cap_c.release()

        # 최종 집계
        if votes1:
            final1 = Counter(votes1).most_common(1)[0][0]
            self.progress.emit(f"[Evaluate] 1s Final: {EMOTIONS[final1]}")
        else:
            self.progress.emit("[Evaluate] 1s Final: No predictions")
        if votes2:
            for idx, l2 in enumerate(votes2, 1):
                self.progress.emit(f"[Evaluate] 15s Cycle {idx} Final: {EMOTIONS[l2]}")
        else:
            self.progress.emit("[Evaluate] 15s Final: No predictions")

        # ========== [FUSION] 가중치 로드 (전역 모델 가중치) ==========
        w1, w15 = 0.5, 0.5
        meta_path = path_join(self.model_dir, FUSION_META_FILE)
        if os.path.exists(meta_path):
            meta = np.load(meta_path)
            w1 = float(meta.get('w1', 0.5))    # 1초 모델 전역 가중치
            w15 = float(meta.get('w15', 0.5))  # 15초 모델 전역 가중치
            self.progress.emit(f"[Evaluate][FUSION] loaded weights: w1={w1:.3f}, w15={w15:.3f}")
        else:
            self.progress.emit("[Evaluate][FUSION] fusion weights not found; using 0.5/0.5")

        # --- 블록별 평균 확률 (없으면 0벡터) ---
        C = len(EMOTIONS)
        def avg_or_zero(lst):
            return np.mean(np.stack(lst, axis=0), axis=0) if len(lst)>0 else np.zeros(C, dtype=float)

        p1a  = avg_or_zero(probs_1s_a)   # 1초 A
        p15a = avg_or_zero(probs_15_a)   # 15초 A
        p1b  = avg_or_zero(probs_1s_b)   # 1초 B
        p15b = avg_or_zero(probs_15_b)   # 15초 B

        # --- 엔트로피 기반 블록 신뢰도(0~1) ---
        def entropy(p):
            p = np.clip(p, 1e-8, 1.0)
            return float(-np.sum(p * np.log(p)))
        maxH = np.log(C)

        def conf(p):   # 확률 분포가 날카로울수록(낮은 엔트로피) 신뢰↑
            if p.sum() == 0: return 0.0
            return 1.0 - (entropy(p)/maxH)

        c1a, c15a, c1b, c15b = conf(p1a), conf(p15a), conf(p1b), conf(p15b)

        # --- 전역 모델 가중치 × 블록 신뢰도 => 블록별 알파 ---
        a1a  = w1  * c1a
        a15a = w15 * c15a
        a1b  = w1  * c1b
        a15b = w15 * c15b

        # (옵션) 후반 블록에 시간가중(최근성) 주고 싶으면 ↓ 주석 해제
        # time_boost = 1.0    # 예: 1.0이면 부스트 없음, 1.2면 최근 블록 20% 가산
        # a1b  *= time_boost
        # a15b *= time_boost

        s = a1a + a15a + a1b + a15b + 1e-6
        a1a, a15a, a1b, a15b = a1a/s, a15a/s, a1b/s, a15b/s

        # --- 최종 확률 = 블록 가중합 ---
        p_final = a1a*p1a + a15a*p15a + a1b*p1b + a15b*p15b
        final_label = int(np.argmax(p_final))

        # 로그 출력(디버깅용)
        def topk_str(p):
            if p.sum()==0: return "N/A"
            k = int(np.argmax(p))
            return f"{EMOTIONS[k]} (max {p.max():.2f})"
        self.progress.emit(
            "[Evaluate][FUSION] "
            f"1s-A={topk_str(p1a)} w={a1a:.2f} | "
            f"15s-A={topk_str(p15a)} w={a15a:.2f} | "
            f"1s-B={topk_str(p1b)} w={a1b:.2f} | "
            f"15s-B={topk_str(p15b)} w={a15b:.2f}"
        )
        self.progress.emit(f"[Evaluate][FUSION] Final: {EMOTIONS[final_label]}")


# --- GUI Application ---
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Dual Realtime Emotion GUI')
        self.resize(1200, 800)

        self.video_widget = QVideoWidget()
        # 영상 영역을 크게, 확장 가능하게
        self.video_widget.setMinimumSize(960, 540)
        self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setVideoOutput(self.video_widget)
        self.player.setAudioOutput(self.audio_output)

        self.log = QtWidgets.QTextEdit(readOnly=True)
        # 로그 높이를 상대적으로 작게 만들기 위한 최소/최대 지정
        self.log.setMinimumHeight(150)
        self.log.setMaximumHeight(300)

        btn_cal = QtWidgets.QPushButton('Calibrate Models')
        btn_eval= QtWidgets.QPushButton('Evaluate Models')
        # 버튼 높이 통일
        btn_cal.setFixedHeight(30)
        btn_eval.setFixedHeight(30)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.video_widget, 6)   # 영상 비중을 더 크게
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(btn_cal)
        btn_layout.addWidget(btn_eval)
        layout.addLayout(btn_layout)
        layout.addWidget(self.log, 2)            # 로그 비중 줄임

        container = QtWidgets.QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        btn_cal.clicked.connect(lambda: self.start_task('calibrate'))
        btn_eval.clicked.connect(lambda: self.start_task('evaluate'))

        # 전체화면 토글 (더블클릭)
        self.video_widget.mouseDoubleClickEvent = self.toggle_fullscreen
        self._is_fullscreen = False

        self.worker = None

    def toggle_fullscreen(self, event):
        if not self._is_fullscreen:
            self.video_widget.setFullScreen(True)
            self._is_fullscreen = True
        else:
            self.video_widget.setFullScreen(False)
            self._is_fullscreen = False
            
    def start_task(self, mode):
        if self.worker and self.worker.isRunning(): return
        self.log.clear()
        base = os.path.dirname(os.path.abspath(__file__))
        data_dir = path_join(base,'data')
        model_dir= path_join(base,'model')
        self.worker = WorkerThread(mode, data_dir, model_dir)
        self.worker.progress.connect(self.log.append)
        self.worker.finished.connect(self.log.append)
        self.worker.video_to_play.connect(self.play_video)
        self.worker.start()

    def play_video(self, path):
        # Handle empty path as stop command
        if not path:
            self.player.stop()
            return
        self.player.stop()
        url = QUrl.fromLocalFile(path)
        self.player.setSource(url)
        self.player.play()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

