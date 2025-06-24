import pickle
import numpy as np
import glob
import os

# 📌 데이터 폴더 설정 (여기에 있는 모든 .dat 파일을 변환)
dat_folder_path = "/home/bcml1/2025_EMOTION/data_preprocessed_python"  # 변환할 DAT 파일 폴더
eeg_npy_folder = "/home/bcml1/sigenv/_2주차_data_preprocess/test_DEAP_EEG/"  # EEG NPY 저장 폴더
labels_npy_folder = "/home/bcml1/sigenv/_2주차_data_preprocess/test_DEAP_label/"  # 감정 라벨 NPY 저장 폴더

# 📌 변환할 .dat 파일 리스트 가져오기
dat_files = glob.glob(os.path.join(dat_folder_path, "*.dat"))  # 폴더 내 모든 .dat 파일 찾기
print(f"🔍 총 {len(dat_files)}개의 DAT 파일을 찾았습니다.")

# 📌 저장 폴더 생성 (없을 경우 자동 생성)
os.makedirs(eeg_npy_folder, exist_ok=True)
os.makedirs(labels_npy_folder, exist_ok=True)

# 📌 각 파일을 변환하여 저장
for dat_file in dat_files:
    subject_id = os.path.basename(dat_file).replace(".dat", "")  # 파일명에서 주제 ID 추출 (예: s01)
    
    eeg_npy_path = os.path.join(eeg_npy_folder, f"{subject_id}_eeg.npy")  # EEG 저장 경로
    labels_npy_path = os.path.join(labels_npy_folder, f"{subject_id}_labels.npy")  # 라벨 저장 경로
    
    # 📌 DAT 파일 로드
    with open(dat_file, "rb") as f:
        data = pickle.load(f, encoding="latin1")  # Python 2로 저장된 데이터 호환

    # 📌 EEG 데이터 변환 및 저장
    if "data" in data:
        eeg_data = np.array(data["data"])[:, :32, :]  # (트라이얼, 채널, 샘플)
        eeg_data = np.transpose(eeg_data, (0, 2, 1))  # (트라이얼, 샘플, 채널)로 변환
        np.save(eeg_npy_path, eeg_data)
        print(f"✅ {subject_id} EEG 데이터 저장 완료 → {eeg_npy_path}")
    else:
        print(f"❌ {subject_id} 오류: 'data' 키를 찾을 수 없습니다.")

    # 📌 감정 라벨 변환 및 저장
    if "labels" in data:
        labels = np.array(data["labels"])  # 감정 라벨 (Valence, Arousal 등)
        np.save(labels_npy_path, labels)
        print(f"✅ {subject_id} 라벨 데이터 저장 완료 → {labels_npy_path}")
    else:
        print(f"❌ {subject_id} 오류: 'labels' 키를 찾을 수 없습니다.")

print("🎉 모든 변환이 완료되었습니다!")
