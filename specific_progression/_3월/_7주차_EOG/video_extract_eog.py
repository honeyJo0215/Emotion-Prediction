import cv2
import numpy as np
import os
import glob

def extract_eog_from_video(video_path):
    """
    비디오 파일에서 매 프레임마다 눈 영역을 검출하여,
    좌/우 눈의 중심 좌표를 EOG 유사 신호(4채널: [left_x, left_y, right_x, right_y])로 추출합니다.
    눈 검출에 실패한 경우에는 NaN 값을 기록합니다.
    """
    cap = cv2.VideoCapture(video_path)
    eog_signals = []  # 각 프레임의 [left_eye_x, left_eye_y, right_eye_x, right_eye_y]
    
    # OpenCV 내장 Haar Cascade for eye detection
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 그레이스케일 변환 (검출 성능 향상)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 눈 검출 (scaleFactor와 minNeighbors는 필요에 따라 조정)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        if len(eyes) >= 2:
            # 검출된 눈들을 x좌표 기준으로 정렬하여 좌측과 우측 눈을 구분
            eyes = sorted(eyes, key=lambda e: e[0])
            left_eye = eyes[0]
            right_eye = eyes[1]
            
            # 각 눈의 중심 좌표 계산
            lx = left_eye[0] + left_eye[2] / 2
            ly = left_eye[1] + left_eye[3] / 2
            rx = right_eye[0] + right_eye[2] / 2
            ry = right_eye[1] + right_eye[3] / 2
            
            eog_signals.append([lx, ly, rx, ry])
        else:
            # 눈 검출에 실패하면 NaN 값 기록 (4채널)
            eog_signals.append([np.nan, np.nan, np.nan, np.nan])
    
    cap.release()
    return np.array(eog_signals)

def process_all_videos(input_base, output_base):
    """
    주어진 입력 폴더(/home/bcml1/2025_EMOTION/face_video)의 s01~s22 디렉토리를 순회하며,
    각 비디오 파일에 대해 EOG 유사 신호를 추출한 후, 
    출력 폴더(/home/bcml1/2025_EMOTION/DEAP_EOG) 내의 동일한 sXX 폴더에 npy 파일로 저장합니다.
    """
    # s01부터 s22까지 반복
    for i in range(1, 23):
        subj = f's{i:02d}'
        subj_input_dir = os.path.join(input_base, subj)
        subj_output_dir = os.path.join(output_base, subj)
        os.makedirs(subj_output_dir, exist_ok=True)
        
        # 주로 사용되는 비디오 확장자(mp4, avi)를 찾음
        video_files = glob.glob(os.path.join(subj_input_dir, '*.mp4')) + glob.glob(os.path.join(subj_input_dir, '*.avi'))
        
        if not video_files:
            print(f"{subj_input_dir}에 비디오 파일이 없습니다.")
            continue
        
        for video_path in video_files:
            print(f"Processing {video_path} ...")
            signals = extract_eog_from_video(video_path)
            # 비디오 파일명에 기반하여 npy 파일 이름 지정
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(subj_output_dir, base_name + '_eog.npy')
            np.save(output_path, signals)
            print(f"Saved EOG signals to {output_path}")

if __name__ == "__main__":
    input_base = '/home/bcml1/2025_EMOTION/face_video'
    output_base = '/home/bcml1/2025_EMOTION/DEAP_EOG'
    process_all_videos(input_base, output_base)
