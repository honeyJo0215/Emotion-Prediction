# 0: neutral, 1: sad, 2: fear, 3: happy
# (SEEDIV데이터셋과 동일한 라벨링을 적용하기)

# 4~6은 neutral
# 우선, 두 척도의 중앙 영역(4~6점 사이)을 중립(neutral) 상태로 설정합니다.
# valence와 arousal 모두 낮은 경우에는 슬픔(sad)으로 판단합니다.
# valence는 낮지만 arousal이 높은 경우는 공포(fear)로 분류합니다.
# 반대로, 두 척도 모두 높은 경우는 행복(happy)으로 분류할 수 있습니다.
# 이와 같이, 전체 valence–arousal 평면을 중심 영역과 네 개의 사분면으로 나누어 0: neutral, 1: sad, 2: fear, 3: happy의 4개 클래스로 분류할 수 있음



# 데이터 읽어오는 경로: '/home/bcml1/2025_EMOTION/DEAP_eeg_npy_files/'
# 데이터의 shape: (40, 32, T(8064)) (1초는 128Hz임)
import numpy as np
import os
import glob

# ==============================
# 1. 감정 라벨 정의 함수
"""
0:중립
1:슬픔
2:공포
3:행복
4:제외(원래는 bored)
"""
# ==============================
def label_emotion(valence, arousal):
    if 4 <= valence <= 6 and 4 <= arousal <= 6:
        return 0  # neutral
    elif valence < 4 and arousal < 4:
        return 1  # sad
    elif valence < 4 and arousal > 6:
        return 2  # fear
    elif valence > 6 and arousal > 6:
        return 3  # happy
    else:
        return 4  # default to remove

# ==============================
# 2. 경로 설정
# ==============================
label_dir = '/home/bcml1/2025_EMOTION/DEAP_npy_files+label'
signal_dir = '/home/bcml1/2025_EMOTION/DEAP_eeg_new'
save_dir   = '/home/bcml1/2025_EMOTION/DEAP_eeg_new_label'

os.makedirs(save_dir, exist_ok=True)

# ==============================
# 3. 파일 목록 순회
# ==============================
signal_files = sorted(glob.glob(os.path.join(signal_dir, 'folder1_subject*_sample_*.npy')))

for path in signal_files:
    # 파일명 파싱: subject 번호와 sample 번호 추출
    filename = os.path.basename(path)
    parts = filename.split('_')
    subject_num = int(parts[1].replace('subject', ''))  # subject1 → 1
    sample_num = int(parts[3].replace('.npy', ''))      # sample_03.npy → 3

    # 라벨 파일 로드
    label_path = os.path.join(label_dir, f"s{subject_num:02}_labels.npy")
    labels = np.load(label_path)  # shape: (40, 4)
    val, aro = labels[sample_num - 1, 0], labels[sample_num - 1, 1]

    # 라벨 계산
    label = label_emotion(val, aro)

    # 신호 데이터 로드
    signal_data = np.load(path)

    # 새로운 파일 이름 생성
    new_filename = f"folder1_subject{subject_num}_sample_{sample_num:02}_label{label}.npy"
    save_path = os.path.join(save_dir, new_filename)

    # 저장
    np.save(save_path, signal_data)

    print(f"✅ Saved: {new_filename} (valence={val:.2f}, arousal={aro:.2f} → label={label})")

print("\n🎉 모든 파일 저장 완료!")

