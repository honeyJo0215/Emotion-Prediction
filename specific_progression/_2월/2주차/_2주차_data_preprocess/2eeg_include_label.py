import os
import shutil
import numpy as np

# 📌 감정 라벨 매핑 (숫자 → 감정 이름)
EMOTION_LABELS = {
    0: "Negative",  # valence <= 4.06
    1: "Positive",  # valence >= 6.53
    2: "Neutral"    # Neutral 4.06 ~ 6.53
}

# def classify_emotion(valence):
#     """ Valence(긍정도) 값에 따라 감정 범주를 지정 """
#     if valence <= 4.06:
#         return 0  # Negative
#     elif valence > 4.06 and valence < 6.53:
#         return 2  # Neutral
#     else:  # valence >= 6.53
#         return 1  # Positive
    
#위에 감정라벨 매핑은 안해도 됨! 이미 0, 1, 2로 되어있음!

def add_emotion_labels_to_filenames(input_dir, label_dir, output_dir):
    """
    각 EEG 파일의 sample ID에 기반하여 감정 라벨을 가져와 파일명을 업데이트하는 함수.
    
    Args:
        input_dir (str): EEG 세그먼트 데이터(.npy) 파일이 저장된 폴더.
        label_dir (str): 감정 라벨(.npy) 파일이 저장된 폴더.
        output_dir (str): 새로운 파일명을 적용하여 저장할 폴더.
    """
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(input_dir):
        if not file_name.endswith(".npy"):
            print(f"Skipping invalid file: {file_name}")
            continue

        # 📌 파일명에서 정보 추출
        try:
            subject_id = file_name.split("_")[0]  # e.g., "s01"
            sample_id = int(file_name.split("_sample_")[1].split("_")[0])  # e.g., "23"
            segment_id = int(file_name.split("_segment_")[1].replace(".npy", ""))  # e.g., "035"
        except (IndexError, ValueError) as e:
            print(f"Error parsing file name {file_name}: {e}")
            continue

        # 📌 감정 라벨 파일 찾기
        label_file = os.path.join(label_dir, f"{subject_id}_three_labels.npy")
        if not os.path.exists(label_file):
            print(f"Label file not found: {label_file}")
            continue

        # 📌 감정 라벨 로드
        emotion_labels = np.load(label_file)  # 감정 라벨 파일 로드

        # 🎯 sample_id를 0부터 시작하도록 변환
        sample_idx = sample_id - 1  # sample_id는 1~40이므로 0~39로 변환

        # 🎯 감정 라벨이 (40, 4) 형태인 경우, 첫 번째 값(valence)만 사용
        try:
            if emotion_labels.ndim == 2 and emotion_labels.shape[1] == 4:
                emotion_label_index = int(emotion_labels[sample_idx][0])  # 첫 번째 값 (valence)
            elif emotion_labels.ndim == 1 and len(emotion_labels) == 40:  # 감정 라벨이 (40,) 형태라면 바로 사용
                emotion_label_index = int(emotion_labels[sample_idx])
            else:
                print(f"Unexpected label shape: {emotion_labels.shape} for {file_name}")
                continue
        except IndexError:
            print(f"⚠️ IndexError: {file_name} - sample_idx {sample_idx} is out of range.")
            continue

        # 📌 감정 라벨 매핑 (0 → Negative, 1 → Positive, 2 → Neutral)
        emotion_label = EMOTION_LABELS.get(emotion_label_index, "Unknown")

        # 📌 새로운 파일명 생성
        new_file_name = f"{subject_id}_sample_{sample_id:02d}_segment_{segment_id:03d}_label_{emotion_label}.npy"
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, new_file_name)

        # 📌 파일을 복사하여 저장 (파일 삭제 X)
        shutil.copy2(input_path, output_path)
        print(f"Copied: {input_path} -> {output_path}")

# 📌 경로 설정
input_dir = "/home/bcml1/2025_EMOTION/DEAP_EEG_10s_1soverlap/"
output_dir = "/home/bcml1/2025_EMOTION/DEAP_EEG_10s_1soverlap_labeled/"  # 저장 폴더
label_dir = "/home/bcml1/2025_EMOTION/DEAP_three_labels"  # 감정 라벨 데이터

# 📌 실행
add_emotion_labels_to_filenames(input_dir, label_dir, output_dir)