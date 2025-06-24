import os
import numpy as np
import scipy.io as sio
from scipy.signal import medfilt
from sklearn.preprocessing import MinMaxScaler

def load_eye_mat_file(file_path):
    """ .mat 파일에서 Eye-tracking 데이터를 로드 """
    mat_data = sio.loadmat(file_path)
    # 24개 영화 클립 데이터를 로드 (cell array 포함 가능)
    eye_data = {key: np.array(value, dtype=object) for key, value in mat_data.items() if not key.startswith("__")}
    return eye_data

def preprocess_eye_data(eye_data, apply_smoothing=False):
    """ 
    Eye-tracking 데이터 전처리:
    - 결측값 처리 (NaN -> 평균값 대체)
    - Min-Max 정규화
    - 스무딩 처리 (옵션)
    """
    for key in eye_data:
        raw_values = eye_data[key]

        # 🔹 데이터 타입 확인 (디버깅)
        print(f"Processing {key}: Type={type(raw_values)}, Shape={raw_values.shape if isinstance(raw_values, np.ndarray) else 'N/A'}")

        # 🔹 리스트(cell array)인 경우 numpy 배열로 변환
        if isinstance(raw_values, list):
            raw_values = np.array(raw_values, dtype=object)

        # 🔹 빈 데이터 건너뛰기
        if raw_values.size == 0:
            print(f"Skipping {key}: Empty data")
            continue
        
        # 🔹 데이터가 숫자가 아닐 경우 변환 시도
        if raw_values.dtype != np.float64 and raw_values.dtype != np.float32:
            try:
                raw_values = raw_values.astype(float)  # float 형식으로 변환
            except ValueError:
                print(f"Skipping {key}: Cannot convert to float (contains non-numeric data)")
                continue  # 변환 실패 시 해당 데이터 건너뜀

        # 🔹 NaN 값 처리
        nan_mask = np.isnan(raw_values)
        if np.any(nan_mask):
            mean_values = np.nanmean(raw_values, axis=0)
            inds = np.where(nan_mask)
            raw_values[inds] = np.take(mean_values, inds[1])

        # 🔹 Min-Max 정규화 (값의 범위 조정)
        scaler = MinMaxScaler()
        eye_data[key] = scaler.fit_transform(raw_values)

        # 🔹 스무딩 (옵션)
        if apply_smoothing:
            eye_data[key] = medfilt(eye_data[key], kernel_size=3)

    return eye_data

def process_eye_data(input_folder, output_folder, apply_smoothing=False):
    """ Eye-tracking 데이터 전처리 후 저장 """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.mat'):
            file_path = os.path.join(input_folder, file_name)
            eye_data = load_eye_mat_file(file_path)
            processed_data = preprocess_eye_data(eye_data, apply_smoothing=apply_smoothing)

            output_file = os.path.join(output_folder, file_name.replace('.mat', '.npy'))
            np.save(output_file, processed_data)
            print(f"Processed Eye: {file_name} -> {output_file} (Smoothing: {apply_smoothing})")

# 실행 예시
input_eye_dir = "/home/bcml1/2025_EMOTION/SEED_IV/eye_raw_data"
output_eye_dir = "/home/bcml1/2025_EMOTION/SEED_IV/eye_raw_data_npy"
process_eye_data(input_eye_dir, output_eye_dir, apply_smoothing=False)
