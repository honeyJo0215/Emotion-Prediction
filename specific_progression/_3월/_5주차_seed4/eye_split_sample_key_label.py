import scipy.io as sio
import numpy as np
import os
import re

# === 세션별 라벨 정의 ===
session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]

# === 어느 세션의 라벨을 사용할지 설정 (1, 2, 또는 3) ===
SESSION_ID = 3  # 예: 세션1 라벨을 사용

# 세션에 따른 라벨 매핑
if SESSION_ID == 1:
    session_label = session1_label
elif SESSION_ID == 2:
    session_label = session2_label
elif SESSION_ID == 3:
    session_label = session3_label
else:
    raise ValueError("SESSION_ID는 1, 2, 또는 3 중 하나여야 합니다.")

# === .mat 파일이 저장된 폴더 경로 ===
mat_dir = "/home/bcml1/2025_EMOTION/SEED_IV/eye_feature_smooth/3"  # 실제 경로로 변경
save_dir = "/home/bcml1/2025_EMOTION/SEED_IV/eye_feature_smooth/3_npy_sample"  # 저장할 폴더
os.makedirs(save_dir, exist_ok=True)

# === 폴더 내 모든 .mat 파일 가져오기 ===
mat_files = [f for f in os.listdir(mat_dir) if f.endswith(".mat")]

# === 변환 작업 ===
for mat_file in mat_files:
    # 예: "1_20160518.mat" → subject_id = "1_20160518"
    subject_id = os.path.splitext(mat_file)[0]

    # .mat 파일 로드
    mat_path = os.path.join(mat_dir, mat_file)
    mat_data = sio.loadmat(mat_path)

    # trial은 총 24개 (eye_1 ~ eye_24)
    for trial_idx in range(1, 25):
        key_name = f"eye_{trial_idx}"
        if key_name in mat_data:
            # shape = (31, W)인 2D 배열
            trial_data = mat_data[key_name]  # (31, W)
            
            # trial 번호에 대응하는 라벨 (session_label[trial_idx-1])
            label = session_label[trial_idx - 1]
            
            # 출력 파일명: "X_XXXXXXXX_sample_XX_label_X.npy"
            # 예: "1_20160518_sample_01_label_2.npy"
            out_file = f"{subject_id}_sample_{trial_idx:02d}_label_{label}.npy"
            
            # .npy 파일로 저장
            np.save(os.path.join(save_dir, out_file), trial_data)

print("✅ Eye crop 데이터를 trial별 .npy로 변환 완료!")
