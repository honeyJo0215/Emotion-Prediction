import cv2
import mediapipe as mp
import numpy as np
import subprocess
import shlex
import os
import av
import math
from av.error import InvalidDataError
import keras
keras.config.enable_unsafe_deserialization()
import tensorflow as tf
from tensorflow.keras import layers as _layers

# ── 디버깅용 몽키패치 ───────────────────────────────────────────────
_orig_lambda_from_config = _layers.Lambda.from_config
def _debug_lambda_from_config(cls, config, custom_objects=None):
    # 역직렬화 시도되는 Lambda 레이어의 이름과 function 직렬화 문자열을 찍어본다
    layer_name = config.get('name', '<unknown>')
    func_str   = config.get('function', '<no-func-in-config>')
    print(f"[DEBUG] Deserializing Lambda layer '{layer_name}', function repr: {func_str}")
    return _orig_lambda_from_config(config, custom_objects=custom_objects)
_layers.Lambda.from_config = classmethod(_debug_lambda_from_config)

# Mediapipe 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

SEGMENT = 1
LEFT_EYE   = [33,160,158,133,153,144]
RIGHT_EYE  = [362,385,387,263,373,380]
LEFT_IRIS  = [474,475,476,477]
RIGHT_IRIS = [469,470,471,472]

# 이제 30개 이름
feature_names = [
    "pupil_size", "dilation_speed", "constriction_speed",
    "EAR", "blink_flag", "blink_rate", "blink_duration",
    "fixation_duration", "fixation_count", "fixation_flag",
    "saccade_amplitude", "saccade_velocity",
    "microsaccade_flag", "microsaccade_amplitude", "microsaccade_velocity",
    "first_saccade_latency", "pupil_variability", "blink_interval_variability",
    "upper_eyelid_opening", "eyelid_velocity",
    "regression_count", "heatmap_entropy",
    "post_blink_refix_latency",
    "long_fixation_ratio", "synchrony", "direction_bias",
    "pre_blink_size", "post_blink_size", "post_blink_saccade_velocity",
    "path_stability"
]

def eye_aspect_ratio(lm, idx):
    pts = np.array([(lm[i].x, lm[i].y) for i in idx])
    A = np.linalg.norm(pts[1]-pts[5])
    B = np.linalg.norm(pts[2]-pts[4])
    C = np.linalg.norm(pts[0]-pts[3])
    return (A + B) / (2.0 * C)

def pupil_diameter(lm, idx, dims):
    pts = np.array([(lm[i].x * dims[1], lm[i].y * dims[0]) for i in idx])
    return np.linalg.norm(pts[0] - pts[2])

def read_frames_pyav(path):
    container = av.open(path)
    video_stream = container.streams.video[0]
    frames = []
    for packet in container.demux(video_stream):
        try:
            for frame in packet.decode():
                img = frame.to_ndarray(format="bgr24")
                frames.append(img)
        except InvalidDataError:
            # 문제 있는 패킷은 건너뜁니다
            continue
    fps = float(video_stream.average_rate or 30.0)
    return frames, fps

def get_metadata_count(path):
    cap = cv2.VideoCapture(path)
    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return cnt


def read_frames_ffmpeg(path):
    # 메타데이터 얻기
    cap = cv2.VideoCapture(path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 50.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    cmd = [
        'ffmpeg', '-err_detect', 'ignore_err',
        '-i', path,
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24', '-'
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_size = width * height * 3
    frames = []
    while True:
        buf = proc.stdout.read(frame_size)
        if len(buf) < frame_size:
            break
        frames.append(
            np.frombuffer(buf, np.uint8).reshape((height, width, 3))
        )
    proc.wait()
    return frames, fps

def get_metadata_count(path):
    cap = cv2.VideoCapture(path)
    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return cnt

def transcode(path):
    base, _ = os.path.splitext(path)
    out_mp4 = base + '_fixed.mp4'
    cmd = [
        'ffmpeg', '-y',
        '-err_detect', 'ignore_err',
        '-i', path,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        out_mp4
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_mp4


def process_frames(frames, fps, window_sec):
    pupil, ear, centers, times = [], [], [], []
    for i, frame in enumerate(frames, start=1):
        h, w = frame.shape[:2]
        res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            pupil.append(np.nan)
            ear.append(np.nan)
            centers.append([np.nan, np.nan])
        else:
            lm = res.multi_face_landmarks[0].landmark
            pd = (pupil_diameter(lm, LEFT_IRIS, (h,w)) +
                  pupil_diameter(lm, RIGHT_IRIS,(h,w))) / 2.0
            pupil.append(pd)
            e = (eye_aspect_ratio(lm, LEFT_EYE) +
                 eye_aspect_ratio(lm, RIGHT_EYE)) / 2.0
            ear.append(e)
            cL = np.mean([(lm[j].x*w, lm[j].y*h) for j in LEFT_IRIS], axis=0)
            cR = np.mean([(lm[j].x*w, lm[j].y*h) for j in RIGHT_IRIS], axis=0)
            centers.append(((cL + cR)/2).tolist())
        times.append(i/fps)

    T = len(pupil)
    arr_pd  = np.array(pupil, dtype=np.float32)
    arr_ear = np.array(ear,   dtype=np.float32)
    arr_ctr = np.array(centers, dtype=np.float32)
    if arr_ctr.ndim == 1:
        arr_ctr = arr_ctr.reshape(-1,2)
    t = np.array(times, dtype=np.float32)

    # 결측치 보간
    for arr in (arr_pd, arr_ear, arr_ctr[:,0], arr_ctr[:,1]):
        nans = np.isnan(arr)
        idxs = np.arange(T)
        arr[nans] = np.interp(idxs[nans], idxs[~nans], arr[~nans])

    # dt 계산 & zero-division 방지
    dt = np.diff(t, prepend=t[0])
    dt[dt == 0] = 1.0 / fps

    # 변화율, 이동량
    dp   = np.diff(arr_pd,  prepend=arr_pd[0])  / dt
    de   = np.diff(arr_ear, prepend=arr_ear[0]) / dt
    disp = np.linalg.norm(np.diff(arr_ctr, axis=0, prepend=arr_ctr[0:1]), axis=1)
    vel  = disp / dt

    ear_mean = np.nanmean(arr_ear)
    ear_std  = np.nanstd(arr_ear)
    blink_thresh = ear_mean - 1.5 * ear_std
    blink_flag = arr_ear < blink_thresh
    blink_events = []
    start = None
    for i, b in enumerate(blink_flag):
        if b and start is None:
            start = i
        elif not b and start is not None:
            blink_events.append((start, i-1))
            start = None
    if start is not None:
        blink_events.append((start, T-1))

    vel_med = np.median(vel)
    fix_flag = vel < vel_med
    fix_events = []
    start = None
    for i, f in enumerate(fix_flag):
        if f and start is None:
            start = i
        elif not f and start is not None:
            fix_events.append((start, i-1))
            start = None
    if start is not None:
        fix_events.append((start, T-1))

    blink_intervals = np.diff([s for s,_ in blink_events]) / fps if len(blink_events)>1 else np.array([])
    #pupil_var  = np.std(arr_pd)
    #blink_ints = np.diff([s for s,_ in blink_events]) / fps if len(blink_events)>1 else np.array([0])
    #blink_int_var  = np.std(blink_ints)
    #heat_entropy   = -np.sum((disp/disp.sum()) * np.log2(disp/disp.sum()+1e-8))
    #sync           = np.std(arr_ctr[:,0] - arr_ctr[:,1])
    #dir_bias       = np.mean(np.sign(np.diff(arr_ctr[:,1], prepend=arr_ctr[0,1])))
    #path_stab      = np.var(arr_ctr.reshape(T,-1), axis=0).mean()

    # 슬라이딩 헬퍼
    w = int(window_sec * fps)
    def sliding(func):
        return np.array([func(slice(max(0,i-w//2), min(T,i+w//2+1))) for i in range(T)])

    blink_rate = sliding(lambda sl: blink_flag[sl].sum()/window_sec)
    blink_dur  = sliding(lambda sl:
        np.mean([(e-s+1)/fps for s,e in blink_events if s>=sl.start and e<sl.stop])*1000
        if any(s>=sl.start and e<sl.stop for s,e in blink_events) else 0
    )
    blink_int_var_win = sliding(
        lambda sl: np.nanstd(blink_intervals[max(0, sl.start-1):sl.stop])
        if blink_intervals[max(0, sl.start-1):sl.stop].size > 0 else 0
    )
    post_blink_refix = sliding(lambda sl:
        np.mean([(fs-fe)/fps*1000
                 for (s,fe),(fs,_) in zip(blink_events, fix_events)
                 if fe>=sl.start and fs<sl.stop])
        if blink_events and fix_events else 0
    )
    long_fix_ratio = sliding(
        lambda sl: (sum((e-s+1)/fps >= 0.5 for s,e in fix_events if s>=sl.start and e<sl.stop)
                   / max(1, sum(1 for s,e in fix_events if s>=sl.start and e<sl.stop)))
    )
    # --- 윈도우별 동적 계산으로 변경 ---
    pupil_var_win    = sliding(lambda sl: np.std(arr_pd[sl]))
    blink_intervals  = np.diff([s for s,_ in blink_events]) / fps if len(blink_events)>1 else np.zeros(T)
    # blink_int_var_win= sliding(lambda sl: np.std(blink_intervals[max(0, sl.start-1):sl.stop]))  # window 내 인터벌 분산
    EPS = 1e-6
    def safe_entropy(disp_slice):
        probs = disp_slice / (disp_slice.sum() + EPS)
        return -np.sum(probs * np.log2(probs + EPS))
   
    # 2) sliding 내부 heat_entropy_win 에 EPS 적용
    heat_entropy_win = sliding(lambda sl: safe_entropy(disp[sl]))
    sync_win         = sliding(lambda sl: np.std(arr_ctr[sl,0] - arr_ctr[sl,1]))
    dir_bias_win     = sliding(lambda sl: np.mean(np.sign(np.diff(arr_ctr[sl,1], prepend=arr_ctr[sl.start,1]))))

    # 기존에 이미 정의된 나머지 슬라이딩 채널들
    # blink_rate            = sliding(lambda sl: blink_flag[sl].sum()/window_sec)
    # blink_dur             = sliding(lambda sl: np.mean([(e-s+1)/fps for s,e in blink_events if s>=sl.start and e<sl.stop])*1000
                                    #if any(s>=sl.start and e<sl.stop for s,e in blink_events) else 0)
    fix_dur               = sliding(lambda sl: np.mean([(e-s+1)/fps for s,e in fix_events if s>=sl.start and e<sl.stop])*1000
                                    if any(s>=sl.start and e<sl.stop for s,e in fix_events) else 0)
    fix_count             = sliding(lambda sl: sum(1 for s,e in fix_events if s>=sl.start and e<sl.stop))
    # post_blink_refix      = sliding(lambda sl: np.mean([(fs-fe)/fps*1000
    #                                 for (s,fe),(fs,_) in zip(blink_events, fix_events)
    #                                 if fe>=sl.start and fs<sl.stop]) if blink_events and fix_events else 0)
    # long_fix_ratio        = sliding(lambda sl: sum((e-s+1)/fps>=0.5 for s,e in fix_events if s>=sl.start and e<sl.stop)
    #                                 / max(1, sum(1 for s,e in fix_events if s>=sl.start and e<sl.stop)))
    path_stab_win         = sliding(lambda sl: np.var(arr_ctr[sl], axis=0).mean())
    first_saccade_latency = sliding(lambda sl: sl.start*1000/fps)
    blink_int_var_win = np.nan_to_num(blink_int_var_win, nan=0.0)
    long_fix_ratio    = np.nan_to_num(long_fix_ratio, nan=0.0)
    features = np.vstack([
        arr_pd,
        dp.clip(min=0),
        (-dp).clip(min=0),
        arr_ear,
        blink_flag.astype(float),
        blink_rate,
        blink_dur,
        fix_dur,
        fix_count,
        fix_flag.astype(float),
        disp,
        vel,
        (disp<=np.percentile(disp,20)).astype(float),
        disp*(disp<=np.percentile(disp,20)),
        vel*(disp<=np.percentile(disp,20)),
        first_saccade_latency,
        pupil_var_win,           # 변경
        blink_int_var_win,       # 변경
        arr_ear,                 # upper_eyelid_opening
        np.abs(de),              # eyelid_velocity
        sliding(lambda sl: np.sum(np.diff(arr_ctr[sl,0])<0)),  # regression_count
        heat_entropy_win,        # 변경
        post_blink_refix,
        long_fix_ratio,
        sync_win,                # 변경
        dir_bias_win,            # 변경
        np.concatenate(([arr_pd[0]], arr_pd[:-1])),            # pre_blink_size
        np.concatenate((arr_pd[1:], [arr_pd[-1]])),            # post_blink_size
        np.concatenate(([vel[0]], vel[:-1])),                  # post_blink_saccade_velocity
        path_stab_win
    ])
    return features


def extract_features_from_video(path, segment_sec=SEGMENT):
    # 메타데이터 프레임 수와 예상 세그먼트 수 계산
    meta_frames = get_metadata_count(path)
    frames, fps = read_frames_pyav(path)
    print(f"▶ metadata frames = {meta_frames}, decoded frames = {len(frames)}")
    if not frames:
        return np.empty((len(feature_names), 0), dtype=np.float32), feature_names

    all_feat = process_frames(frames, fps, segment_sec)
    actual_frames = all_feat.shape[1]
    seg_len = max(1, int(round(segment_sec * fps)))
    expected_segs = math.ceil(meta_frames / seg_len)
    actual_segs = math.ceil(actual_frames / seg_len)

    # 실제 세그먼트 평균 계산
    seg_feats = np.zeros((len(feature_names), actual_segs), dtype=np.float32)
    for i in range(actual_segs):
        s = i * seg_len
        e = min((i + 1) * seg_len, actual_frames)
        seg_feats[:, i] = np.mean(all_feat[:, s:e], axis=1)

    # 부족한 세그먼트 패딩
    if actual_segs < expected_segs:
        pad_count = expected_segs - actual_segs
        last_col = seg_feats[:, -1:].copy()
        pad = np.repeat(last_col, pad_count, axis=1)
        seg_feats = np.concatenate([seg_feats, pad], axis=1)
    elif actual_segs > expected_segs:
        # 초과 시 자르기
        seg_feats = seg_feats[:, :expected_segs]
    

    # (필요하다면 다른 슬라이딩 채널에도 동일 기법 적용)

    # 3) 최종 seg_feats 에 nan, inf 가 남아있다면 모두 0으로 치환
    seg_feats = np.nan_to_num(
        seg_feats,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )
    print(f"✔ returning {seg_feats.shape[1]} segments (expected {expected_segs})")
    return seg_feats, feature_names

def extract_features_from_frames(frame_list, fps: float, segment_sec: float) -> np.ndarray:
    """
    실시간으로 수집한 프레임 리스트에서 eye 특징을 추출합니다.

    Args:
        frame_list (List[np.ndarray]): OpenCV로 읽어온 BGR 형식의 프레임 리스트
        fps (float): 초당 프레임 수 (웹캠 설정값)
        segment_sec (float): 하나의 세그먼트 길이(초)

    Returns:
        feats (np.ndarray): shape (feat_dim, n_segments) 의 특징 매트릭스
    """
    # 1) BGR -> RGB 변환
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frame_list]

    # 2) numpy array 로 변환: shape (n_frames, H, W, C)
    frames_np = np.stack(rgb_frames, axis=0)

    # 3) 학습 시 extract_features_from_video 에서 사용하던 process_frames() 호출
    #    process_frames(frames: np.ndarray, fps: float, segment_sec: float)
    feats = process_frames(frames_np, fps, segment_sec)

    return feats

if __name__ == "__main__":
    video_path = "/home/bcml1/2025_EMOTION/face_video/s13/s13_trial27.avi"
    segment_sec = SEGMENT

    segs, names = extract_features_from_video(video_path, segment_sec)
    # segs shape: (30, num_segments)
    if segs.size == 0 or segs.shape[1] == 0:
        print(f"Error: '{video_path}'에서 프레임을 읽어오지 못했거나 세그먼트를 생성하지 못했습니다.")
        print("파일 경로와 파일 형식을 확인하세요.")
    else:
        print(f"features shape = {segs.shape}  # (30, 세그먼트 개수)")
        first = segs[:, 0]
        print(f"-- 첫 번째 세그먼트 ({segment_sec}s) 평균 30채널 데이터 --")
        for i, name in enumerate(names):
            print(f"{i+1:2d}. {name} : {first[i]:.4f}")
