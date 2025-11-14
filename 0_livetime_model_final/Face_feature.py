import cv2
import mediapipe as mp
import numpy as np
import subprocess
import os
import av
import math
from av.error import InvalidDataError
import keras
keras.config.enable_unsafe_deserialization()

# Mediapipe Face Mesh 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 30개 얼굴 표정 피처 이름
feature_names = [
    "inner_brow_raise", "outer_brow_raise", "horizontal_forehead_wrinkle", "cheek_raise", "brow_slope_angle",
    "nasolabial_fold_depth", "mouth_width", "nostril_flare", "mouth_area", "lip_corner_pull",
    "lip_corner_depress", "lip_stretch", "lip_press", "visible_teeth_ratio", "gum_visibility",
    "mouth_stretch", "mouth_opening_height", "jaw_drop", "head_yaw", "chin_raise",
    "chin_depression", "lower_lip_depressor", "cheek_dimpler", "lip_tightener", "lips_part",
    "facial_symmetry_index", "nasolabial_asymmetry", "lip_curvature", "head_pitch", "jaw_lateral_shift"
]

# 랜드마크 인덱스 매핑 (Mediapipe FaceMesh)
LM = {
    'left_brow_inner': 55, 'right_brow_inner': 285,
    'left_brow_outer': 65, 'right_brow_outer': 295,
    'forehead_left': 10, 'forehead_right': 338,
    'left_cheek': 50, 'right_cheek': 280,
    'nasolabial_left': 61, 'nasolabial_right': 291,
    'mouth_left_corner': 61, 'mouth_right_corner': 291,
    'upper_lip_top': 13, 'upper_lip_bottom': 14,
    'lower_lip_top': 14, 'lower_lip_bottom': 17,
    'teeth_top': 0, 'teeth_bottom': 17,
    'gums_top': 13, 'gums_bottom': 14,
    'jaw_lower': 152,
    'nose_left': 98, 'nose_right': 327,
    'chin': 199,
    'face_contour': list(range(10))
}

# 기본 거리, 각도 계산

def euclidean(p, q):
    return np.linalg.norm(p - q)

def angle(p, q, r):
    v1 = p - q
    v2 = r - q
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

# 피처 계산 함수들

def compute_inner_brow_raise(lm, w, h):
    pL = np.array([lm[LM['left_brow_inner']].x * w, lm[LM['left_brow_inner']].y * h])
    pR = np.array([lm[LM['right_brow_inner']].x * w, lm[LM['right_brow_inner']].y * h])
    return ((h/2 - pL[1]) + (h/2 - pR[1]))/2

def compute_outer_brow_raise(lm, w, h):
    pL = np.array([lm[LM['left_brow_outer']].x * w, lm[LM['left_brow_outer']].y * h])
    pR = np.array([lm[LM['right_brow_outer']].x * w, lm[LM['right_brow_outer']].y * h])
    return ((h/2 - pL[1]) + (h/2 - pR[1]))/2

def compute_horizontal_forehead_wrinkle(lm, w, h):
    y1 = lm[LM['forehead_left']].y * h
    y2 = lm[LM['forehead_right']].y * h
    return abs(y1 - y2)

def compute_cheek_raise(lm, w, h):
    cL = np.array([lm[LM['left_cheek']].x * w, lm[LM['left_cheek']].y * h])
    cR = np.array([lm[LM['right_cheek']].x * w, lm[LM['right_cheek']].y * h])
    return ((h/2 - cL[1]) + (h/2 - cR[1]))/2

def compute_brow_slope_angle(lm, w, h):
    p1 = np.array([lm[LM['left_brow_inner']].x * w, lm[LM['left_brow_inner']].y * h])
    p2 = np.array([lm[LM['right_brow_inner']].x * w, lm[LM['right_brow_inner']].y * h])
    return angle(p1, p2, np.array([w/2, h/2]))

def compute_nasolabial_fold_depth(lm, w, h):
    nl = np.array([lm[LM['nasolabial_left']].x * w, lm[LM['nasolabial_left']].y * h])
    nr = np.array([lm[LM['nasolabial_right']].x * w, lm[LM['nasolabial_right']].y * h])
    return (h/2 - nl[1] + h/2 - nr[1]) / 2

def compute_mouth_width(lm, w, h):
    p1 = np.array([lm[LM['mouth_left_corner']].x * w, lm[LM['mouth_left_corner']].y * h])
    p2 = np.array([lm[LM['mouth_right_corner']].x * w, lm[LM['mouth_right_corner']].y * h])
    return euclidean(p1, p2)

def compute_nostril_flare(lm, w, h):
    nl = np.array([lm[LM['nose_left']].x * w, lm[LM['nose_left']].y * h])
    nr = np.array([lm[LM['nose_right']].x * w, lm[LM['nose_right']].y * h])
    return (nr[0] - nl[0])

def compute_mouth_area(lm, w, h):
    xs = [lm[i].x * w for i in [61,146,91,181,84,17,314,405,321,375]]
    ys = [lm[i].y * h for i in [61,146,91,181,84,17,314,405,321,375]]
    pts = np.vstack((xs, ys)).T
    return 0.5 * abs(np.dot(pts[:,0], np.roll(pts[:,1],1)) - np.dot(pts[:,1], np.roll(pts[:,0],1)))

def compute_lip_corner_pull(lm, w, h):
    top = np.array([lm[LM['upper_lip_top']].x * w, lm[LM['upper_lip_top']].y * h])
    lc = np.array([lm[LM['mouth_left_corner']].x * w, lm[LM['mouth_left_corner']].y * h])
    rc = np.array([lm[LM['mouth_right_corner']].x * w, lm[LM['mouth_right_corner']].y * h])
    return ((top[1] - lc[1]) + (top[1] - rc[1]))/2

def compute_lip_corner_depress(lm, w, h):
    bot = np.array([lm[LM['lower_lip_bottom']].x * w, lm[LM['lower_lip_bottom']].y * h])
    lc = np.array([lm[LM['mouth_left_corner']].x * w, lm[LM['mouth_left_corner']].y * h])
    rc = np.array([lm[LM['mouth_right_corner']].x * w, lm[LM['mouth_right_corner']].y * h])
    return ((bot[1] - lc[1]) + (bot[1] - rc[1]))/2

def compute_lip_stretch(lm, w, h):
    return compute_mouth_width(lm,w,h)

def compute_lip_press(lm, w, h):
    return -compute_mouth_opening_height(lm,w,h)

def compute_visible_teeth_ratio(lm, w, h):
    mouth_h = compute_mouth_opening_height(lm,w,h)
    width = compute_mouth_width(lm,w,h)
    return mouth_h/width if width>0 else 0

def compute_gum_visibility(lm, w, h):
    gum = np.array([lm[LM['gums_top']].x*w, lm[LM['gums_top']].y*h])
    teeth = np.array([lm[LM['teeth_top']].x*w, lm[LM['teeth_top']].y*h])
    return max(0, teeth[1]-gum[1])

def compute_mouth_stretch(lm, w, h):
    return compute_mouth_width(lm,w,h)/ (h/2)

def compute_mouth_opening_height(lm, w, h):
    top = np.array([lm[LM['upper_lip_bottom']].x*w, lm[LM['upper_lip_bottom']].y*h])
    bot = np.array([lm[LM['lower_lip_top']].x*w, lm[LM['lower_lip_top']].y*h])
    return abs(bot[1]-top[1])

def compute_jaw_drop(lm, w, h):
    jaw = np.array([lm[LM['jaw_lower']].x*w, lm[LM['jaw_lower']].y*h])
    mid = np.array([(lm[LM['mouth_left_corner']].x+lm[LM['mouth_right_corner']].x)/2*w,
                    (lm[LM['mouth_left_corner']].y+lm[LM['mouth_right_corner']].y)/2*h])
    return max(0, mid[1]-jaw[1])

def compute_head_yaw(lm, w, h):
    cl = np.array([lm[LM['left_cheek']].x*w, lm[LM['left_cheek']].y*h])
    cr = np.array([lm[LM['right_cheek']].x*w, lm[LM['right_cheek']].y*h])
    return (cl[0] - cr[0])

def compute_chin_raise(lm, w, h):
    chin = np.array([lm[LM['chin']].x*w, lm[LM['chin']].y*h])
    return (h/2 - chin[1])

def compute_chin_depression(lm, w, h):
    return -compute_chin_raise(lm,w,h)

def compute_lower_lip_depressor(lm, w, h):
    bot = np.array([lm[LM['lower_lip_bottom']].x*w, lm[LM['lower_lip_bottom']].y*h])
    return (bot[1] - h/2)

def compute_cheek_dimpler(lm, w, h):
    cl = np.array([lm[LM['left_cheek']].x*w, lm[LM['left_cheek']].y*h])
    lc = np.array([lm[LM['mouth_left_corner']].x*w, lm[LM['mouth_left_corner']].y*h])
    cr = np.array([lm[LM['right_cheek']].x*w, lm[LM['right_cheek']].y*h])
    rc = np.array([lm[LM['mouth_right_corner']].x*w, lm[LM['mouth_right_corner']].y*h])
    return (euclidean(cl, lc) + euclidean(cr, rc))/2

def compute_lip_tightener(lm, w, h):
    return -compute_mouth_width(lm,w,h)

def compute_lips_part(lm, w, h):
    return compute_mouth_opening_height(lm,w,h)/ (compute_mouth_width(lm,w,h)+1e-6)

def compute_facial_symmetry_index(lm, w, h):
    cl = np.array([lm[LM['left_cheek']].x*w, lm[LM['left_cheek']].y*h])
    cr = np.array([lm[LM['right_cheek']].x*w, lm[LM['right_cheek']].y*h])
    return abs(euclidean(cl, np.array([w/2,h/2])) - euclidean(cr, np.array([w/2,h/2])))

def compute_nasolabial_asymmetry(lm, w, h):
    nl = np.array([lm[LM['nasolabial_left']].x*w, lm[LM['nasolabial_left']].y*h])
    nr = np.array([lm[LM['nasolabial_right']].x*w, lm[LM['nasolabial_right']].y*h])
    return abs((h/2 - nl[1]) - (h/2 - nr[1]))

def compute_lip_curvature(lm, w, h):
    lc = np.array([lm[LM['mouth_left_corner']].x*w, lm[LM['mouth_left_corner']].y*h])
    rc = np.array([lm[LM['mouth_right_corner']].x*w, lm[LM['mouth_right_corner']].y*h])
    top = np.array([lm[LM['upper_lip_top']].x*w, lm[LM['upper_lip_top']].y*h])
    return (angle(top, lc, rc) + angle(top, rc, lc))/2

def compute_head_pitch(lm, w, h):
    fh = np.array([lm[LM['forehead_left']].x*w, lm[LM['forehead_left']].y*h])
    chin = np.array([lm[LM['chin']].x*w, lm[LM['chin']].y*h])
    return (chin[1] - fh[1])

def compute_jaw_lateral_shift(lm, w, h):
    jaw = np.array([lm[LM['jaw_lower']].x*w, lm[LM['jaw_lower']].y*h])
    return jaw[0] - w/2

# 프레임 처리 및 세그먼트 평균 계산

def process_frames(frames, fps):
    T = len(frames)
    feats = np.zeros((len(feature_names), T), dtype=np.float32)
    funcs = [globals()[f"compute_{name}"] for name in feature_names]
    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            feats[:, i] = 0
            continue
        lm = res.multi_face_landmarks[0].landmark
        for j, func in enumerate(funcs):
            feats[j, i] = func(lm, w, h)
    return feats

# 메타데이터 프레임 수 가져오기

def get_metadata_count(path):
    cap = cv2.VideoCapture(path)
    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return cnt

# 비디오에서 얼굴 피처 추출 (세그먼트 패딩/컷 적용)

def extract_face_features_from_video(path, segment_sec=10.0):
    # 메타데이터
    meta_frames = get_metadata_count(path)
    cap = None
    frames = []
    fps = None
    # PyAV 시도
    try:
        container = av.open(path)
        vs = container.streams.video[0]
        fps = float(vs.average_rate or 30.0)
        for packet in container.demux(vs):
            try:
                for frame in packet.decode():
                    frames.append(frame.to_ndarray(format="bgr24"))
            except InvalidDataError:
                continue
    except Exception:
        # fallback to OpenCV
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        while True:
            ret, fr = cap.read()
            if not ret: break
            frames.append(fr)
        cap.release()
    # 실제 디코딩된 프레임 수
    dec_frames = len(frames)
    if dec_frames == 0:
        # 빈 반환
        return np.zeros((len(feature_names), 0), dtype=np.float32), feature_names
    # 프레임별 피처
    all_feat = process_frames(frames, fps)
    # 세그먼트 길이 프레임 수
    seg_len = max(1, int(round(segment_sec * fps)))
    # 예상 세그먼트 개수
    expected_segs = math.ceil(meta_frames / seg_len)
    # 실제 세그먼트 평균 계산
    actual_segs = math.ceil(all_feat.shape[1] / seg_len)
    seg_feats = np.zeros((len(feature_names), expected_segs), dtype=np.float32)
    for si in range(expected_segs):
        s = si * seg_len
        e = min((si+1) * seg_len, all_feat.shape[1])
        if s < all_feat.shape[1]:
            seg_feats[:, si] = np.nanmean(all_feat[:, s:e], axis=1)
        else:
            # 패딩: 마지막 유효 세그먼트
            seg_feats[:, si] = seg_feats[:, si-1]
    # nan, inf -> 0
    seg_feats = np.nan_to_num(seg_feats, nan=0.0, posinf=0.0, neginf=0.0)
    return seg_feats, feature_names

if __name__ == "__main__":
    video_path = "path/to/face_video.mp4"
    seg_feats, names = extract_face_features_from_video(video_path, segment_sec=1.0)
    print(f"Extracted shape: {seg_feats.shape}  # (30, expected_seg) per metadata")
    for idx, nm in enumerate(names):
        print(f"{idx+1}. {nm}: {seg_feats[idx,0]:.4f}")
