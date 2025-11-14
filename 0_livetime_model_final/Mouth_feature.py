import cv2
import cv2
import mediapipe as mp
import numpy as np
import os
import math
from tqdm import tqdm

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

SEGMENT_DURATION_SEC = 10  # 세그먼트 길이

# 랜드마크 인덱스
OUTER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
MOUTH_CORNERS = [61, 291]      # 좌, 우 입꼬리
LIP_VERTICAL_POINTS = [12, 15] # 윗입술, 아랫입술 중앙
EYE_CORNERS = [33, 263]        # 좌, 우 눈꼬리
LIP_SYMMETRY_PAIRS = [
    (61, 291), (146, 375), (91, 321), (181, 405), (84, 314),
    (78, 308), (191, 415), (80, 310), (81, 311), (82, 312),
    (185, 409), (40, 270), (39, 269), (37, 267),
    (95, 324), (88, 318), (178, 402), (87, 317)
]
# 입 윤곽
MOUTH_OUTLINE = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 270, 269, 267, 0, 37, 39, 40, 185, 61]


# 전체 특징 이름
feature_names = [
    "mouth_width",                 # 입 너비 (좌우 입꼬리 사이 거리)
    "mouth_height_inner",          # 내부 입술 높이 (내부 입술 상하 중앙점 사이 거리)
    "mouth_height_outer",          # 외부 입술 높이 (입술 외곽선의 최대 수직 높이)
    "mouth_aspect_ratio_mar",      # 입 종횡비 (내부 높이 / 너비)


    "lip_curvature",               # 윗입술 곡률
    "mouth_corner_dist",           # 입꼬리 사이 거리 (mouth_width와 동일할 수 있으나 별도 함수 존재)
    "mouth_corner_angle",          # 입꼬리와 입 중앙이 이루는 각도
    

    "lip_symmetry_angle_avg",


    "mouth_outline_area",          # 입술 외곽선이 감싸는 면적
    "mouth_outline_perimeter",     # 입술 외곽선의 둘레 길이
    "mouth_outline_circularity",   # 입술 외곽선의 원형성 (모양이 원에 가까운 정도)


    "mouth_eye_dist_left",         # 왼쪽 입꼬리-눈꼬리 거리
    "mouth_eye_dist_right",        # 오른쪽 입꼬리-눈꼬리 거리


    "mar_change",               # MAR의 프레임 간 차이값
    "width_change",             # 입 너비의 프레임 간 차이값
    "height_change",            # 내부 입술 높이의 프레임 간 차이값
    "outline_area_change",      # 외곽선 면적의 프레임 간 차이값
    

    "lip_contour_change",       # 입술 윤곽점들의 평균 위치 변화량


    "mouth_open_flag",             # 입 벌림 상태 플래그 (임계값 이상으로 벌어졌는지)
    "lip_press_flag",              # 입술 다묾 상태 플래그 (임계값 이하로 닫혔는지)
]



# ==============================================================================
# 2. Feature 추출 함수들
# ==============================================================================


# 입 너비(양쪽 입꼬리 사이 거리), 높이(윗입술 중앙과 아랫입술 중앙 사이 거리)를 통해 MAR 계산
# 입을 다물면 높이가 0에 가까워져 MAR ↓, 입을 크게 벌리면 높이가 커져 MAR ↑.
def get_mouth_dimensions(lm, w, h):
    p_left = lm[MOUTH_CORNERS[0]]
    p_right = lm[MOUTH_CORNERS[1]]
    mouth_width = np.linalg.norm(np.array([p_left.x, p_left.y]) - np.array([p_right.x, p_right.y])) * w
    
    p_upper = lm[LIP_VERTICAL_POINTS[0]]
    p_lower = lm[LIP_VERTICAL_POINTS[1]]
    mouth_height_inner = np.linalg.norm(np.array([p_upper.x, p_upper.y]) - np.array([p_lower.x, p_lower.y])) * h
    
    mar = mouth_height_inner / (mouth_width + 1e-6)
    return mouth_width, mouth_height_inner, mar


# 윗입술 곡률 계산
# 왼쪽 입꼬리, 오른쪽 입꼬리, 윗입술 중앙 Y좌표를 이용해 곡률 계산 (입꼬리의 평균 높이에서 윗입술 중앙의 높이를 빼기) 
def get_lip_curvature(lm):
    y_corner_left = lm[MOUTH_CORNERS[0]].y
    y_corner_right = lm[MOUTH_CORNERS[1]].y
    y_lip_center = lm[LIP_VERTICAL_POINTS[0]].y
    curvature = ((y_corner_left + y_corner_right) / 2) - y_lip_center
    return curvature


# 입꼬리 사이의 거리 계산
# 왼쪽 입꼬리와 오른쪽 입꼬리 사이의 직선 거리를 픽셀 단위로 계산
def get_mouth_corner_dist(lm, w, h):
    p_left = np.array([lm[61].x * w, lm[61].y * h])
    p_right = np.array([lm[291].x * w, lm[291].y * h])
    return np.linalg.norm(p_left - p_right)


# 입꼬리와 입 중앙이 이루는 각도 계산
# 입 모양이 넓은지, 좁고 둥근지 판별 가능
def get_mouth_corner_angle(lm, w, h):
    p_left = np.array([lm[61].x * w, lm[61].y * h])
    p_right = np.array([lm[291].x * w, lm[291].y * h])
    p_center = np.array([lm[13].x * w, lm[13].y * h])
    
    vec_left = p_left - p_center
    vec_right = p_right - p_center
    cos_angle = np.dot(vec_left, vec_right) / (np.linalg.norm(vec_left) * np.linalg.norm(vec_right) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle


# 외부 입술 윤곽 전체의 수직 높이를 픽셀단위로 계산
# 입술의 가장 낮은 지점(max) - 가장 높은 지점(min) * 이미지의 높이(h)
def get_outer_lip_height(lm, h):
    outer_lip_y = [lm[i].y for i in MOUTH_OUTLINE]
    return (max(outer_lip_y) - min(outer_lip_y)) * h if outer_lip_y else 0


# 왼쪽/오른쪽 각각의 입꼬리, 눈꼬리 사이 직선 거리를 픽셀 단위로 계산
def get_mouth_eye_corner_dists(lm, w, h):
    p_mouth_left = np.array([lm[61].x * w, lm[61].y * h])
    p_mouth_right = np.array([lm[291].x * w, lm[291].y * h])
    p_eye_left = np.array([lm[33].x * w, lm[33].y * h])
    p_eye_right = np.array([lm[263].x * w, lm[263].y * h])
    
    dist_left = np.linalg.norm(p_mouth_left - p_eye_left)
    dist_right = np.linalg.norm(p_mouth_right - p_eye_right)
    return dist_left, dist_right


# 신발끈 공식을 통한 다각형 면적 계산
def get_polygon_area(points):
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

# 세 점의 좌표(p1, p2, p3)를 받아 p2를 꼭짓점으로 하는 각도를 계산
def calculate_angle(p1, p2, p3):
    vec1 = p1 - p2
    vec2 = p3 - p2
    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    cosine_angle = dot_product / (magnitude1 * magnitude2 + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle_rad = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

# LIP_SYMMETRY_PAIRS의 모든 쌍과 윗입술 중앙점(12) 사이의 각도를 계산
def get_lip_symmetry_angle_avg(lm, w, h):
    center_vertex_idx = LIP_VERTICAL_POINTS[0]  # 12
    p_center = np.array([lm[center_vertex_idx].x * w, lm[center_vertex_idx].y * h])
    
    all_angles = []
    for left_idx, right_idx in LIP_SYMMETRY_PAIRS:
        p_left = np.array([lm[left_idx].x * w, lm[left_idx].y * h])
        p_right = np.array([lm[right_idx].x * w, lm[right_idx].y * h])
        
        angle = calculate_angle(p_left, p_center, p_right)
        all_angles.append(angle)
        
    return np.mean(all_angles) if all_angles else 0.0

# 다각형 둘레 계산
def get_polygon_perimeter(points):
    if len(points) < 2: return 0.0
    return np.sum(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1))

# 입술 외곽선 특징 추출 (면적, 둘레, 원형성)
def get_mouth_outline_features(lm, w, h):
    points = np.array([(lm[i].x * w, lm[i].y * h) for i in MOUTH_OUTLINE])
    
    area = get_polygon_area(points) # 면적
    perimeter = get_polygon_perimeter(points) # 둘레
    
    # 원형성 계산: 4 * pi * Area / Perimeter^2
    circularity = (4 * np.pi * area) / (perimeter**2 + 1e-6) if perimeter > 0 else 0.0
    return area, perimeter, circularity


# ==============================================================================
# 3. process_frames
# ==============================================================================

def process_frames(frames, fps, window_sec):
    if not frames:
        return np.empty((len(feature_names), 0)), []

    # 1. 프레임별 Feature 추출
    static_features = {
        "mouth_width": [], "mouth_height_inner": [], "mouth_height_outer": [], "mouth_aspect_ratio_mar": [],
        "lip_curvature": [], "mouth_corner_dist": [], "mouth_corner_angle": [],
        "mouth_outline_area": [], "mouth_outline_perimeter": [], "mouth_outline_circularity": [],
        "mouth_eye_dist_left": [], "mouth_eye_dist_right": [], "lip_symmetry_angle_avg": []
    }
    list_contour_pts = []

    for frame in frames:
        h, w, _ = frame.shape
        res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            
            width, height, mar = get_mouth_dimensions(lm, w, h)
            static_features["mouth_width"].append(width)
            static_features["mouth_height_inner"].append(height)
            static_features["mouth_aspect_ratio_mar"].append(mar)
            static_features["mouth_height_outer"].append(get_outer_lip_height(lm, h))
            static_features["lip_curvature"].append(get_lip_curvature(lm))
            static_features["mouth_corner_dist"].append(get_mouth_corner_dist(lm, w, h))
            static_features["mouth_corner_angle"].append(get_mouth_corner_angle(lm, w, h))
            
            area, perimeter, circularity = get_mouth_outline_features(lm, w, h)
            static_features["mouth_outline_area"].append(area)
            static_features["mouth_outline_perimeter"].append(perimeter)
            static_features["mouth_outline_circularity"].append(circularity)
            
            dist_l, dist_r = get_mouth_eye_corner_dists(lm, w, h)
            static_features["mouth_eye_dist_left"].append(dist_l)
            static_features["mouth_eye_dist_right"].append(dist_r)

            avg_angle = get_lip_symmetry_angle_avg(lm, w, h)
            static_features["lip_symmetry_angle_avg"].append(avg_angle)

            list_contour_pts.append(np.array([(lm[i].x * w, lm[i].y * h) for i in MOUTH_OUTLINE]))
        
        else:
            for key in static_features:
                static_features[key].append(np.nan)
            list_contour_pts.append(np.full((len(MOUTH_OUTLINE), 2), np.nan))

    # static_features의 NaN 값 보간
    for name, lst in static_features.items():
        arr = np.array(lst, dtype=np.float32)
        nans = np.isnan(arr)
        if np.all(nans): arr.fill(0)
        elif np.any(nans): arr[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), arr[~nans])
        static_features[name] = arr

    arr_contour = np.array(list_contour_pts, dtype=np.float32)

    for i in range(arr_contour.shape[1]):
        for j in range(arr_contour.shape[2]):
            arr = arr_contour[:, i, j]
            nans = np.isnan(arr)
            if np.all(nans): arr.fill(0)
            elif np.any(nans): arr[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), arr[~nans])


    all_features = static_features.copy()

    # 5. 프레임 간 변화량
    all_features["mar_change"] = np.abs(np.diff(all_features["mouth_aspect_ratio_mar"], prepend=all_features["mouth_aspect_ratio_mar"][0]))
    all_features["width_change"] = np.abs(np.diff(all_features["mouth_width"], prepend=all_features["mouth_width"][0]))
    all_features["height_change"] = np.abs(np.diff(all_features["mouth_height_inner"], prepend=all_features["mouth_height_inner"][0]))
    all_features["outline_area_change"] = np.abs(np.diff(all_features["mouth_outline_area"], prepend=all_features["mouth_outline_area"][0]))

    # 6. 입술 윤곽점 위치 변화량
    contour_disp = np.linalg.norm(np.diff(arr_contour, axis=0, prepend=arr_contour[0:1]), axis=2)
    lip_contour_change = np.mean(contour_disp, axis=1)
    all_features["lip_contour_change"] = lip_contour_change
    
    # 이벤트 기반 특징
    mar_arr = all_features["mouth_aspect_ratio_mar"]
    height_arr = all_features["mouth_height_inner"]
    open_thresh = np.nanmean(mar_arr) + 1.0 * np.nanstd(mar_arr)
    press_thresh = np.percentile(height_arr[height_arr > 0], 10) if np.any(height_arr > 0) else 0.1
    all_features["mouth_open_flag"] = (mar_arr > open_thresh).astype(float)
    all_features["lip_press_flag"] = (height_arr < press_thresh).astype(float)

    # 3. 최종 특징 배열 생성
    try:
        features_stacked = np.vstack([all_features[name] for name in feature_names])
    except KeyError as e:
        print(f"오류: '{e.args[0]}' 특징이 계산되지 않았습니다. feature_names 리스트나 계산 로직을 확인하세요.")
        return np.empty((len(feature_names), 0)), []
    
    return features_stacked, feature_names

# ==============================================================================
# 4. 메인
# ==============================================================================

# 비디오 파일에서 프레임 리스트를 읽어오기
def read_frames_opencv(path):
    frames = []
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error: Could not open video file with OpenCV: {path}")
        return [], 30.0
    
    # fps를 비디오에서 읽기 (실패 시 30)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        
    cap.release()
    return frames, fps

# 비디오 파일의 총 프레임 수를 메타데이터에서 읽어오기
def get_metadata_count(path):
    cap = cv2.VideoCapture(path)
    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return cnt

def extract_features_from_video_mouth(path, segment_sec=10, stride_sec=1):
    frames, fps = read_frames_opencv(path)
    if not frames:
        return np.empty((20, 0)), []

    all_feat, feature_names = process_frames(frames, fps, segment_sec)
    
    if all_feat.size == 0 or not feature_names:
        return np.empty((20, 0)), []

    actual_frames = all_feat.shape[1]

    window_frames = max(1, int(round(segment_sec * fps)))
    stride_frames = max(1, int(round(stride_sec * fps)))

    seg_feats_list = []
    start_frame = 0
    while start_frame + window_frames <= actual_frames:
        s = start_frame
        e = start_frame + window_frames
        
        segment_features = np.mean(all_feat[:, s:e], axis=1)
        seg_feats_list.append(segment_features)

        start_frame += stride_frames
    
    if not seg_feats_list:
        return np.empty((len(feature_names), 0)), feature_names
    
    seg_feats = np.stack(seg_feats_list, axis=1)

    return np.nan_to_num(seg_feats, nan=0.0, posinf=0.0, neginf=0.0), feature_names
