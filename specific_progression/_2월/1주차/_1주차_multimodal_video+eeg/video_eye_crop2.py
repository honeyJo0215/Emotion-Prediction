import cv2
import numpy as np
import os
import torch
import dlib
from tqdm import tqdm

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def limit_gpu_memory(memory_limit_mib=5000):
    """Limit GPU memory usage to the specified amount (in MiB)."""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)  # Convert bytes to MiB
        memory_fraction = memory_limit_mib / total_memory
        torch.cuda.set_per_process_memory_fraction(memory_fraction, 0)
        print(f"GPU memory limited to {memory_limit_mib} MiB ({memory_fraction * 100:.2f}% of total).")
    else:
        print("CUDA is not available. Running on CPU.")

# GPU 메모리 제한 적용
limit_gpu_memory(5000)

# 경로 설정
input_base_path = "/home/bcml1/2025_EMOTION/face_video"
output_base_path = "/home/bcml1/2025_EMOTION/eye_crop"

# 얼굴 및 눈 검출 모델 로드
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # 68개 얼굴 랜드마크 모델

# 세션별 비디오 처리
for session in range(19, 21):
    session_id = f"s{session:02d}"
    input_video_path = os.path.join(input_base_path, session_id)
    output_dir = os.path.join(output_base_path, session_id)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing session: {session_id}")
    
    for video_file in tqdm(os.listdir(input_video_path)):
        if not video_file.endswith(('.mp4', '.avi', '.mov')):
            continue
        
        video_path = os.path.join(input_video_path, video_file)
        cap = cv2.VideoCapture(video_path)
        print(f"Processing video: {video_file}")
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray, 2)  # 얼굴 감지
            
            eye_crops = []
            for face in faces:
                landmarks = landmark_predictor(gray, face)

                # 왼쪽 눈 (Landmark 36~41) & 오른쪽 눈 (Landmark 42~47)
                left_eye_pts = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
                right_eye_pts = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])

                # 눈의 Bounding Box 계산
                lx, ly, lw, lh = cv2.boundingRect(left_eye_pts)
                rx, ry, rw, rh = cv2.boundingRect(right_eye_pts)

                # 눈 감지 예외 처리
                if lw * lh < 10 or rw * rh < 10:
                    print(f"⚠ Warning: Small eye detection at frame {frame_count}. Skipping...")
                    continue  # 눈 크기가 너무 작으면 스킵
                
                # 눈 크롭 & 크기 조정
                left_eye_crop = frame[ly:ly+lh, lx:lx+lw]
                right_eye_crop = frame[ry:ry+rh, rx:rx+rw]

                left_eye_crop = cv2.resize(left_eye_crop, (64, 32))
                right_eye_crop = cv2.resize(right_eye_crop, (64, 32))

                left_eye_crop = torch.tensor(left_eye_crop, device=device)
                right_eye_crop = torch.tensor(right_eye_crop, device=device)

                eye_crops.append(left_eye_crop.cpu().numpy())  # GPU -> CPU 변환 후 저장
                eye_crops.append(right_eye_crop.cpu().numpy())  # GPU -> CPU 변환 후 저장

            # 눈이 감지되지 않은 경우 대체 처리 (중앙 얼굴 크롭)
            if not eye_crops:
                if faces:
                    face = faces[0]  # 첫 번째 감지된 얼굴 사용
                    fx, fy, fw, fh = face.left(), face.top(), face.width(), face.height()

                    eye_crop = frame[fy + fh//3: fy + 2*fh//3, fx + fw//4: fx + 3*fw//4]
                    eye_crop = cv2.resize(eye_crop, (64, 32))
                    eye_crops.append(eye_crop)
                    print("중앙 얼굴 크롭")
                else:
                    print(f"⚠ Warning: No face detected at frame {frame_count}. Skipping...")

            if eye_crops:
                eye_crops_array = np.array(eye_crops)
                save_path = os.path.join(output_dir, f"{video_file}_frame{frame_count:04d}.npy")
                np.save(save_path, eye_crops_array)
                print(f"Saved: {save_path}")
            
            frame_count += 1
        
        cap.release()
    
print("Processing complete!")
