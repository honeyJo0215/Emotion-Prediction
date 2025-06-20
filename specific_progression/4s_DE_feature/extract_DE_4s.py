import numpy as np
import os

def calculate_de(segment):
    """
    주어진 EEG 세그먼트에 대한 Differential Entropy (DE)를 계산합니다.
    세그먼트가 Gaussian 분포를 따른다고 가정합니다.
    
    Parameters:
        segment (ndarray): EEG 신호 세그먼트 (1차원 배열)
    
    Returns:
        float: 계산된 DE 값
    """
    variance = np.var(segment)
    if variance == 0:
        variance = 1e-6  # 로그(0) 방지
    return 0.5 * np.log(2 * np.pi * np.e * variance)

def extract_de_features(eeg_data):
    """
    EEG 데이터에서 DE 특징을 추출합니다.
    
    Parameters:
        eeg_data (ndarray): EEG 신호 데이터 (bands x timestamps x channels)
    
    Returns:
        ndarray: 추출된 DE 특징 (bands x channels)
    """
    bands, time_steps, channels = eeg_data.shape
    features = np.zeros((bands, channels))
    
    for band in range(bands):
        for channel in range(channels):
            segment = eeg_data[band, :, channel]
            de = calculate_de(segment)
            features[band, channel] = de
    
    return features

# 파라미터 정의
bands = ['Theta', 'Alpha', 'Beta', 'Gamma']  # 주파수 대역 이름 (필요 시 사용)

# 입력 및 출력 경로 설정
input_dir = "/home/bcml1/2025_EMOTION/DEAP_EEG/overlap4s_0.125_seg_conv_ch_BPF/segment_sample_2D"
output_dir = "./de_features_overlap_0.125s/"
os.makedirs(output_dir, exist_ok=True)

# 입력 디렉토리 내 모든 .npy 파일 목록 가져오기
input_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]

for file in input_files:
    input_path = os.path.join(input_dir, file)
    print(f"파일 처리 중: {input_path}")

    try:
        eeg_data = np.load(input_path)  # EEG 데이터 파일 로드
    except Exception as e:
        print(f"파일 {file}을(를) 로드하는 중 오류 발생: {e}. 건너뜁니다.")
        continue

    # 데이터 형태 확인
    if eeg_data.ndim != 3:
        print(f"파일 {file}의 데이터 형태가 예상과 다릅니다 {eeg_data.shape}. 건너뜁니다.")
        continue

    # DE 특징 추출
    de_features = extract_de_features(eeg_data)  # (bands x channels)

    # 파일 이름 파싱하여 출력 파일 이름 생성
    base_name = os.path.basename(file).replace(".npy", "")
    parts = base_name.split("_")
    
    # 파일 이름이 예상된 형식을 따르는지 확인
    if len(parts) < 6:
        print(f"파일 이름 {file}이 예상된 형식을 따르지 않습니다. 건너뜁니다.")
        continue

    # 각 부분 추출
    subject = parts[0]
    condition = parts[1]
    # parts[2]은 'FB'로 가정
    segment = parts[3]
    sample_label = parts[4]  # 'sample'
    sample = parts[5]

    # 출력 파일명 생성
    output_file_name = f"{subject}_{condition}_segment_{segment}_sample_{sample}_de_features.npy"

    # 특징을 새로운 .npy 파일로 저장
    output_file = os.path.join(output_dir, output_file_name)
    try:
        np.save(output_file, de_features)
        print(f"DE 특징을 {output_file}에 저장했습니다.")
    except Exception as e:
        print(f"파일 {output_file}을(를) 저장하는 중 오류 발생: {e}.")
