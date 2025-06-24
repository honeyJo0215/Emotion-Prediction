import os
import numpy as np

# 경로 설정
label_dir = "/home/bcml1/2025_EMOTION/DEAP_EEG/labels"  # 감정 라벨 파일 경로
data_dir = "/home/bcml1/sigenv/4s_overlap_data"  # EEG 데이터 파일 경로
output_dir = "./processed_labels"  # 변환된 라벨 저장 경로
os.makedirs(output_dir, exist_ok=True)

# 라벨 파일과 데이터 파일 매핑
label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.npy')])
data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])

# 데이터와 라벨 매핑 및 처리
for label_file, data_file in zip(label_files, data_files):
    # 라벨 및 데이터 로드
    label_path = os.path.join(label_dir, label_file)
    data_path = os.path.join(data_dir, data_file)
    
    label_data = np.load(label_path)  # 감정 라벨
    eeg_data = np.load(data_path)  # EEG 데이터 (band, time, channel)
    
    # 라벨을 (band, channel, 1) 형식으로 변환
    band, time, channel = eeg_data.shape
    required_size = band * channel
    
    # 라벨 크기 조정
    if len(label_data) != required_size:
        print(f"Adjusting size for {label_file}: len={len(label_data)}, required={required_size}")
        if len(label_data) > required_size:
            # 초과된 값을 제거
            label_data = label_data[:required_size]
        elif len(label_data) < required_size:
            # 누락된 값을 0으로 채움
            label_data = np.pad(label_data, (0, required_size - len(label_data)), constant_values=0)
    
    # 라벨을 EEG 형식으로 reshape
    emotion_labels = label_data.reshape((band, channel, 1))
    
    # 저장 파일 경로 생성
    output_path = os.path.join(output_dir, f"{os.path.splitext(data_file)[0]}_emotion_label.npy")
    np.save(output_path, emotion_labels)
    print(f"Processed {label_file} -> {output_path}")
