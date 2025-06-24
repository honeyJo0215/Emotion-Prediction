import numpy as np
import os

# 원본 데이터 경로 및 저장할 디렉토리 설정
base_dir = "/home/bcml1/2025_EMOTION/DEAP_npy_files+label"
output_dir = "/home/bcml1/2025_EMOTION/DEAP_three_labels"

# 저장 디렉토리가 없으면 생성
os.makedirs(output_dir, exist_ok=True)

# 대상 subject 리스트
subject_files = [f"s{str(i).zfill(2)}_labels.npy" for i in range(1, 23)]

# 자동으로 계산된 valence 경계값
lower_bound = 4.06  # Negative ≤ 4.06
upper_bound = 6.53  # Neutral: 4.06 < x ≤ 6.53, Positive > 6.53

# 각 subject에 대해 라벨 변환 및 저장
for file_name in subject_files:
    file_path = os.path.join(base_dir, file_name)
    
    if os.path.exists(file_path):
        labels_data = np.load(file_path, allow_pickle=True)
        valence_values = labels_data[:, 0]  # 첫 번째 열 (Valence 값)

        # Valence 값을 3개의 카테고리로 변환
        three_class_labels = np.where(valence_values <= lower_bound, 0,  # Negative
                             np.where(valence_values > upper_bound, 1, 2))  # Positive: 1, Neutral: 2

        # 변환된 라벨 저장 경로
        save_path = os.path.join(output_dir, file_name.replace("_labels.npy", "_three_labels.npy"))

        # npy 파일로 저장
        np.save(save_path, three_class_labels)
        print(f"Saved: {save_path}")
    else:
        print(f"Warning: {file_path} not found!")
