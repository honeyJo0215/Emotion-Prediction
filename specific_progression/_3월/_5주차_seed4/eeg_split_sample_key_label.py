import scipy.io
import numpy as np
import os
import re

# .mat 파일이 저장된 폴더 경로 (실제 경로로 변경)
mat_dir = "/home/bcml1/2025_EMOTION/SEED_IV/eeg_feature_smooth/1"  
# 변환된 .npy 파일을 저장할 폴더 경로 (실제 경로로 변경)
save_dir = "/home/bcml1/2025_EMOTION/SEED_IV/eeg_feature_smooth/1_npy_sample"  
os.makedirs(save_dir, exist_ok=True)

# 세션1 라벨 정의 (각 trial과 일대일 대응; 총 24 trial)
session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]

# 폴더 내 모든 .mat 파일 가져오기
mat_files = [f for f in os.listdir(mat_dir) if f.endswith(".mat")]

for mat_file in mat_files:
    # 피실험자 ID 추출 (예: "1_20160518.mat" -> "1_20160518")
    subject_id = os.path.splitext(mat_file)[0]
    
    # .mat 파일 로드
    mat_path = os.path.join(mat_dir, mat_file)
    data = scipy.io.loadmat(mat_path)
    
    # mat 파일 내부의 키 중, 우리가 원하는 feature key만 추출
    # 기존 키 형식은 "de_movingAve20", "de_LDS20", "psd_movingAve20", "psd_LDS20" 형태이므로
    # 정규식은 언더스코어 없이 feature type과 trial 번호를 분리함.
    keys = [key for key in data.keys() if re.match(r"(de_movingAve|de_LDS|psd_movingAve|psd_LDS)(\d+)$", key)]
    
    # 각 trial에 대해 label을 부여 (trial 번호는 1부터 시작한다고 가정)
    for trial_idx in range(len(session1_label)):
        label = session1_label[trial_idx]
        
        for key in keys:
            match = re.match(r"(de_movingAve|de_LDS|psd_movingAve|psd_LDS)(\d+)$", key)
            if match:
                feature_type, trial_num_str = match.groups()
                trial_num = int(trial_num_str)
                
                # trial 번호가 일치하는 경우에만 처리
                if trial_num == trial_idx + 1:
                    trial_data = data[key]  # shape: (62, W, 5)
                    
                    # 출력 파일명: {subject_id}_sampleXX_{feature_type}_{label}.npy
                    trial_name = f"{subject_id}_sample_{trial_idx+1:02d}_{feature_type}_label_{label}.npy"
                    np.save(os.path.join(save_dir, trial_name), trial_data)
                    
print("✅ .mat 파일이 trial별 및 key별로 .npy 파일로 변환되었습니다!")
