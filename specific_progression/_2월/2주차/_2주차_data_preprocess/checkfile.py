import os
import numpy as np

def print_npy_shapes(folder_path):
    """
    주어진 폴더 내 모든 .npy 파일의 shape을 출력하는 함수.
    
    Args:
        folder_path (str): .npy 파일이 있는 폴더 경로
    """
    if not os.path.exists(folder_path):
        print(f"❌ 폴더가 존재하지 않습니다: {folder_path}")
        return
    
    npy_files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]

    if not npy_files:
        print(f"❌ 폴더에 .npy 파일이 없습니다: {folder_path}")
        return

    print(f"📂 폴더: {folder_path} (총 {len(npy_files)}개 파일)")

    for file_name in npy_files:
        file_path = os.path.join(folder_path, file_name)
        try:
            data = np.load(file_path)
            print(f"🟢 {file_name} - Shape: {data.shape}")
        except Exception as e:
            print(f"❌ 오류 발생: {file_name} - {e}")

# 📌 실행할 폴더 경로 지정
folder_path = "/home/bcml1/2025_EMOTION/DEAP_EEG_10s_1soverlap_labeled/"

# 📌 실행
print_npy_shapes(folder_path)
