import numpy as np
import os

# 세션별 라벨 (총 24개)
session1_label = [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3]

# subject 파일들이 저장된 폴더 (총 15개)
subject_dir = '/home/bcml1/2025_EMOTION/SEED_IV/eeg_raw_data/1_npy'
# 샘플 파일들을 저장할 폴더
output_dir = '/home/bcml1/2025_EMOTION/SEED_IV/eeg_raw_data/1_npy_sample'
os.makedirs(output_dir, exist_ok=True)

# subject 폴더 내 npy 파일 목록을 정렬하여 가져옵니다.
subject_files = sorted([f for f in os.listdir(subject_dir) if f.endswith('.npy')])

for subject_file in subject_files:
    subject_path = os.path.join(subject_dir, subject_file)
    # npy 파일을 불러오며, 딕셔너리 형태임을 가정합니다.
    subject_data = np.load(subject_path, allow_pickle=True).item()
    
    # 각 세션(샘플)별로 분리 및 저장
    for i in range(24):
        key = f'wq_eeg{i+1}'  # 예: 'cz_eeg1', 'cz_eeg2', ..., 'cz_eeg24'
        if key not in subject_data:
            print(f"Warning: {subject_file}에 {key} 키가 존재하지 않습니다.")
            continue
        
        sample_data = subject_data[key]
        label = session1_label[i]
        
        # 샘플 데이터와 라벨을 딕셔너리로 구성
        sample_dict = {'data': sample_data, 'label': label}
        
        # 파일 이름 구성: 예) 1_20160518_sample_01_label_1.npy
        subject_base = os.path.splitext(subject_file)[0]
        sample_num = str(i+1).zfill(2)
        output_filename = f"{subject_base}_sample_{sample_num}_label_{label}.npy"
        output_path = os.path.join(output_dir, output_filename)
        
        np.save(output_path, sample_dict)
        print(f"Saved: {output_path}")
