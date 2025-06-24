import cv2
import numpy as np
import os
import glob
import mediapipe as mp
import dlib

def extract_eog_from_video(video_path, method='mediapipe', predictor_path=None):
    """
    비디오 파일에서 매 프레임마다 눈의 중심 좌표(수평, 수직)를 추출하여 
    EOG 유사 신호(4채널: [left_x, left_y, right_x, right_y])로 반환합니다.
    
    Parameters:
      video_path (str): 비디오 파일 경로.
      method (str): 'mediapipe' 또는 'dlib' 선택.
      predictor_path (str): dlib 사용 시, 68점 랜드마크 모델 파일 경로.
    
    Returns:
      np.ndarray: (프레임수 x 4) 크기의 numpy 배열.
    """
    cap = cv2.VideoCapture(video_path)
    signals = []  # 각 프레임의 [left_eye_x, left_eye_y, right_eye_x, right_eye_y]
    
    if method == 'mediapipe':
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # RGB 변환 (mediapipe 요구사항)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    h, w, _ = frame.shape
                    landmarks = face_landmarks.landmark
                    
                    # 미디어파이프에서 권장하는 눈 landmark indices
                    left_eye_indices = [33, 133, 160, 159, 158, 157, 173]
                    right_eye_indices = [362, 263, 387, 386, 385, 384, 398]
                    
                    left_eye_coords = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in left_eye_indices])
                    right_eye_coords = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in right_eye_indices])
                    
                    left_center = left_eye_coords.mean(axis=0)
                    right_center = right_eye_coords.mean(axis=0)
                    
                    signals.append([left_center[0], left_center[1],
                                    right_center[0], right_center[1]])
                else:
                    signals.append([np.nan, np.nan, np.nan, np.nan])
    
    elif method == 'dlib':
        # dlib 사용 시 predictor_path 필수
        if predictor_path is None:
            raise ValueError("dlib 방식을 사용하려면 predictor_path를 제공해야 합니다.")
        
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            
            if len(faces) > 0:
                face = faces[0]
                shape = predictor(gray, face)
                
                # dlib 68 랜드마크: 좌측 눈 (36~41), 우측 눈 (42~47)
                left_eye = np.array([[shape.part(i).x, shape.part(i).y] for i in range(36, 42)])
                right_eye = np.array([[shape.part(i).x, shape.part(i).y] for i in range(42, 48)])
                
                left_center = left_eye.mean(axis=0)
                right_center = right_eye.mean(axis=0)
                
                signals.append([left_center[0], left_center[1],
                                right_center[0], right_center[1]])
            else:
                signals.append([np.nan, np.nan, np.nan, np.nan])
    else:
        raise ValueError("method 인자는 'mediapipe' 또는 'dlib' 이어야 합니다.")
    
    cap.release()
    return np.array(signals)

def process_all_videos(input_base, output_base, method='mediapipe', predictor_path=None):
    """
    입력 디렉토리(/home/bcml1/2025_EMOTION/face_video)의 s01~s22 폴더에 있는 비디오들을 순회하며,
    각 비디오에서 EOG 유사 신호(안구 중심의 수평, 수직 좌표)를 추출 후,
    출력 디렉토리(/home/bcml1/2025_EMOTION/DEAP_EOG)에 subject별 폴더를 생성하여 npy 파일로 저장합니다.
    """
    for i in range(10, 23):
        subj = f's{i:02d}'
        subj_input_dir = os.path.join(input_base, subj)
        subj_output_dir = os.path.join(output_base, subj)
        os.makedirs(subj_output_dir, exist_ok=True)
        
        # mp4, avi 확장자 비디오 파일 탐색
        video_files = glob.glob(os.path.join(subj_input_dir, '*.mp4')) + \
                      glob.glob(os.path.join(subj_input_dir, '*.avi'))
        
        if not video_files:
            print(f"{subj_input_dir}에 비디오 파일이 없습니다.")
            continue
        
        for video_path in video_files:
            print(f"{video_path} 처리 중 (방법: {method})...")
            signals = extract_eog_from_video(video_path, method=method, predictor_path=predictor_path)
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(subj_output_dir, base_name + '_eog.npy')
            np.save(output_path, signals)
            print(f"EOG 신호를 {output_path}에 저장 완료.")

if __name__ == "__main__":
    input_base = '/home/bcml1/2025_EMOTION/face_video'
    output_base = '/home/bcml1/2025_EMOTION/DEAP_EOG'
    
    # 사용하고자 하는 방법 선택: 'mediapipe' 또는 'dlib'
    method = 'mediapipe'
    
    # 만약 dlib 방식을 사용할 경우, dlib 모델 파일의 경로를 지정합니다.
    predictor_path = 'shape_predictor_68_face_landmarks.dat' if method == 'dlib' else None
    
    process_all_videos(input_base, output_base, method=method, predictor_path=predictor_path)
