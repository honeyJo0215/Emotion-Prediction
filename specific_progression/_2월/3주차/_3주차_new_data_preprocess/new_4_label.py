import os
import numpy as np

def label_emotions_four_states(input_dir, output_dir):
    """
    s01_labels.npy ~ s32_labels.npy 파일에서 각 비디오 클립을 4가지 감정 상태로 라벨링합니다.
    라벨 매핑: 
      0: Excited, 
      1: Relaxed, 
      2: Stressed, 
      3: Bored
    
    Args:
        input_dir (str): 레이블 파일(.npy) 경로
        output_dir (str): 라벨링된 결과를 저장할 경로
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    def classify_emotion(valence, arousal):
        """
        Valence와 Arousal 값을 기반으로 4가지 감정 상태로 분류합니다.
        기준은 5로 하며, 5보다 큰지 여부로 분류합니다.
        """
        if valence > 5 and arousal > 5:
            return 0  # Excited
        elif valence > 5 and arousal <= 5:
            return 1  # Relaxed
        elif valence <= 5 and arousal > 5:
            return 2  # Stressed
        else:
            return 3  # Bored

    # 파일 처리 (s01_labels.npy ~ s32_labels.npy)
    for subject_id in range(1, 33):
        file_name = f"s{str(subject_id).zfill(2)}_labels.npy"
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, f"s{str(subject_id).zfill(2)}_emotion_labels.npy")
        
        if os.path.exists(input_path):
            # 레이블 파일 로드 (예: (40, 4) 배열)
            labels = np.load(input_path)
            emotion_labels = []
            # 각 비디오 클립에 대해 첫 두 값(Valence, Arousal)을 이용하여 라벨 결정
            for valence, arousal, _, _ in labels:
                emotion_labels.append(classify_emotion(valence, arousal))
            # 결과 저장 (정수 배열)
            np.save(output_path, np.array(emotion_labels))
            print(f"{file_name}: 감정 상태 라벨링 완료 -> {output_path}")
        else:
            print(f"파일 없음: {input_path}")

# 사용 예시
input_dir = '/home/bcml1/2025_EMOTION/DEAP_npy_files+label'  # 레이블 파일이 저장된 경로
output_dir = '/home/bcml1/2025_EMOTION/DEAP_four_labels'    # 라벨링 결과를 저장할 경로

# 함수 호출
label_emotions_four_states(input_dir, output_dir)
