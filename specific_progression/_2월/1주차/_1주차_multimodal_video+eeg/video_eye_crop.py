import cv2
import numpy as np
import os
import torch
from tqdm import tqdm

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def limit_gpu_memory(memory_limit_mib=10000):
    """Limit GPU memory usage to the specified amount (in MiB)."""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)  # Convert bytes to MiB
        memory_fraction = memory_limit_mib / total_memory
        torch.cuda.set_per_process_memory_fraction(memory_fraction, 0)
        print(f"GPU memory limited to {memory_limit_mib} MiB ({memory_fraction * 100:.2f}% of total).")
    else:
        print("CUDA is not available. Running on CPU.")

# GPU 메모리 제한 적용
limit_gpu_memory(10000)

# 경로 설정
input_base_path = "/home/bcml1/2025_EMOTION/face_video"
output_base_path = "/home/bcml1/2025_EMOTION/eye_crop"

# OpenCV Haarcascade 모델 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 세션별 비디오 처리
for session in range(1, 23):
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
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            eye_crops = []
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                
                for (ex, ey, ew, eh) in eyes:
                    eye_crop = frame[y+ey:y+ey+eh, x+ex:x+ex+ew]
                    eye_crop = cv2.resize(eye_crop, (64, 32))  # 눈 크롭 후 크기 조정
                    eye_crop = torch.tensor(eye_crop, device=device)  # GPU 로드
                    eye_crops.append(eye_crop.cpu().numpy())  # GPU -> CPU 변환 후 저장
                    
            if eye_crops:
                eye_crops_array = np.array(eye_crops)
                save_path = os.path.join(output_dir, f"{video_file}_frame{frame_count:04d}.npy")
                np.save(save_path, eye_crops_array)
                print(f"Saved: {save_path}")
            
            frame_count += 1
        
        cap.release()
    
print("Processing complete!")
