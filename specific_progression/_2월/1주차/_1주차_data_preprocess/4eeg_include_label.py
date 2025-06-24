import os
import shutil
import numpy as np

# 📌 감정 라벨 매핑 (숫자 → 감정 이름)
EMOTION_LABELS = {
    0: "Excited",  # High Positive
    1: "Relaxed",  # Low Positive
    2: "Stressed",  # High Negative
    3: "Bored",  # Low Negative
    4: "Neutral"  # Neutral
}

def classify_emotion(valence, arousal):
    """ Valence(긍정도)와 Arousal(각성도)를 기반으로 감정 범주를 지정 """
    if valence > 6 and arousal > 6:
        return 0  # 긍정적이고 자극적
    elif valence > 6 and arousal <= 6:
        return 1  # 긍정적이고 차분함
    elif valence <= 6 and arousal > 6:
        return 2  # 부정적이고 자극적
    elif valence <= 6 and arousal <= 6 and valence > 4 and arousal > 4:
        return 4  # 중립 상태
    else:
        return 3  # 부정적이고 차분함
    # if valence >= 6 and arousal >= 6:
    #     return 0  # Excited (High Positive)
    # elif valence >= 6 and arousal < 6:
    #     return 1  # Relaxed (Low Positive)
    # elif valence < 6 and arousal >= 6:
    #     return 2  # Stressed (High Negative)
    # elif valence < 6 and arousal < 6:
    #     return 3  # Bored (Low Negative)
    # else:
    #     return 4  # Neutral

def add_emotion_labels_to_filenames(input_dir, label_dir, output_dir):
    """
    각 EEG 파일의 sample ID에 기반하여 감정 라벨을 가져와 파일명을 업데이트하는 함수.
    
    Args:
        input_dir (str): EEG 밴드 데이터(.npy) 파일이 저장된 폴더.
        label_dir (str): 감정 라벨(.npy) 파일이 저장된 폴더.
        output_dir (str): 새로운 파일명을 적용하여 저장할 폴더.
    """
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(input_dir):
        if not file_name.endswith("_bands.npy"):
            print(f"Skipping invalid file: {file_name}")
            continue

        # 📌 파일명에서 정보 추출
        try:
            subject_id = file_name.split("_")[0]  # e.g., "s01"
            sample_id = int(file_name.split("_sample_")[1].split("_")[0])  # e.g., "23"
            segment_id = int(file_name.split("_segment_")[1].split("_")[0])  # e.g., "035"
        except (IndexError, ValueError) as e:
            print(f"Error parsing file name {file_name}: {e}")
            continue

        # 📌 감정 라벨 파일 찾기
        label_file = os.path.join(label_dir, f"{subject_id}_labels.npy")
        if not os.path.exists(label_file):
            print(f"Label file not found: {label_file}")
            continue

        # 📌 감정 라벨 로드
        emotion_labels = np.load(label_file)  # 감정 라벨 파일 로드

        # 감정 라벨이 (40, 4) 형태인 경우, sample_id에 해당하는 첫 번째 값과 두 번째 값(Valence & Arousal) 사용
        if emotion_labels.ndim == 2 and emotion_labels.shape[1] == 4:
            valence, arousal = emotion_labels[sample_id][:2]  # 첫 번째와 두 번째 값 사용
            emotion_label_index = classify_emotion(valence, arousal)
        elif emotion_labels.ndim == 1:  # 감정 라벨이 (40,) 형태라면 바로 사용
            emotion_label_index = int(emotion_labels[sample_id])
        else:
            print(f"Unexpected label shape: {emotion_labels.shape}")
            continue

        # 📌 감정 라벨 매핑
        emotion_label = EMOTION_LABELS.get(emotion_label_index, "Unknown")

        # 📌 새로운 파일명 생성
        new_file_name = f"{subject_id}_sample_{sample_id:02d}_segment_{segment_id:03d}_{emotion_label}_bands.npy"
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, new_file_name)

        # 📌 파일을 복사하여 저장 (파일 삭제 X)
        shutil.copy2(input_path, output_path)
        print(f"Copied: {input_path} -> {output_path}")

# 📌 경로 설정
input_dir = "/home/bcml1/sigenv/_1주차_data_preprocess/test_DEAP_EEG_sam_seg_bands/"
output_dir = "/home/bcml1/sigenv/_1주차_data_preprocess/test_DEAP_EEG_sam_seg_labeled/"  # 저장 폴더
label_dir = "/home/bcml1/sigenv/_1주차_data_preprocess/test_DEAP_label"  # 감정 라벨 데이터

# 📌 실행
add_emotion_labels_to_filenames(input_dir, label_dir, output_dir)
