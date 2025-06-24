import os
import numpy as np

# 📌 DE Feature가 저장된 폴더 (입력)
input_dir = "/home/bcml1/sigenv/_1주차_data_preprocess/test_DEAP_EEG_de_features"
# 📌 변환된 2D 맵 저장 폴더 (출력)
output_dir = "/home/bcml1/sigenv/_1주차_data_preprocess/test_DEAP_EEG_de_features_2D_mapping/"
os.makedirs(output_dir, exist_ok=True)

# 📌 32개의 EEG 채널을 6x6 Grid로 매핑
def map_32_to_6x6(de_features):
    """
    DE Feature (4, 32)를 (4, 6, 6) 형태의 2D 매트릭스로 변환하고, 꼭짓점(네 모서리)을 0으로 채움.

    Parameters:
        de_features (ndarray): (4, 32) 형태의 DE 특징 벡터
    
    Returns:
        ndarray: (4, 6, 6) 형태로 변환된 데이터
    """
    mapped_features = np.zeros((4, 6, 6))  # (4,6,6) 초기화 (모서리는 0)
    
    # 6x6 채널 배치 매핑 (채널 인덱스 기준)
    channel_map = [
        [ 0,  1,  2,  3,  4,  0],  
        [ 5,  6,  7,  8,  9, 10],  
        [11, 12, 13, 14, 15, 16],  
        [17, 18, 19, 20, 21, 22],  
        [23, 24, 25, 26, 27, 28],  
        [ 0, 29, 30, 31, 32,  0]   
    ]

    # 각 밴드(Theta, Alpha, Beta, Gamma)에 대해 적용
    for band in range(4):
        for row in range(6):
            for col in range(6):
                channel_idx = channel_map[row][col]
                if channel_idx != 0:  # 꼭짓점은 0으로 유지
                    mapped_features[band, row, col] = de_features[band, channel_idx - 1]
    
    return mapped_features

# 📌 입력 디렉토리 내 모든 .npy 파일 목록 가져오기
input_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]

for file in input_files:
    input_path = os.path.join(input_dir, file)
    print(f"파일 처리 중: {input_path}")

    try:
        de_features = np.load(input_path)  # DE Feature 데이터 파일 로드
    except Exception as e:
        print(f"파일 {file}을(를) 로드하는 중 오류 발생: {e}. 건너뜁니다.")
        continue

    # 데이터 형태 확인
    if de_features.shape != (4, 32):
        print(f"파일 {file}의 데이터 형태가 예상과 다릅니다 {de_features.shape}. 건너뜁니다.")
        continue

    # 📌 (4, 32) → (4, 6, 6) 변환
    de_features_2d = map_32_to_6x6(de_features)

    # 📌 파일명 파싱하여 출력 파일 이름 생성
    base_name = os.path.basename(file).replace(".npy", "")
    parts = base_name.split("_")

    # 파일 이름이 예상된 형식을 따르는지 확인
    if len(parts) < 6:
        print(f"파일 이름 {file}이 예상된 형식을 따르지 않습니다. 건너뜁니다.")
        continue

    # 각 부분 추출
    subject = parts[0]  # 예: "s01"
    sample = parts[2]  # 예: "sample_23"
    segment = parts[4]  # 예: "segment_035"
    emotion = parts[5]  # 예: "Excited"

    # 출력 파일명 생성
    output_file_name = f"{subject}_sample_{sample}_segment_{segment}_label_{emotion}_2D.npy"

    # 특징을 새로운 .npy 파일로 저장
    output_file = os.path.join(output_dir, output_file_name)
    try:
        np.save(output_file, de_features_2d)
        print(f"✅ 2D DE 특징을 {output_file}에 저장했습니다.")
    except Exception as e:
        print(f"파일 {output_file}을(를) 저장하는 중 오류 발생: {e}.")
