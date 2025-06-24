import numpy as np
import matplotlib.pyplot as plt
import os

# 데이터 경로 설정
base_dir = "/home/bcml1/2025_EMOTION/DEAP_npy_files+label"
subject_files = [f"s{str(i).zfill(2)}_labels.npy" for i in range(1, 23)]

# 모든 subject의 valence 값 저장 리스트
valence_data = {}

# 각 subject 파일에서 첫 번째 열(Valence 값, 40개 샘플) 추출
for file_name in subject_files:
    file_path = os.path.join(base_dir, file_name)
    if os.path.exists(file_path):
        labels_data = np.load(file_path, allow_pickle=True)
        valence_data[file_name.split("_")[0]] = labels_data[:, 0]  # 첫 번째 열(Valence) 값 저장
    else:
        print(f"Warning: {file_path} not found!")

# 박스플롯(Boxplot)으로 시각화
plt.figure(figsize=(12, 6))
plt.boxplot(valence_data.values(), labels=valence_data.keys(), patch_artist=True)
plt.xlabel("Subjects (s01 to s22)")
plt.ylabel("Valence Score (0-10)")
plt.title("Valence Distribution Across Subjects (Boxplot)")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 박스플롯 저장
plt.savefig("valence_boxplot.png")
print("Saved: valence_boxplot.png")

# 히스토그램 시각화
plt.figure(figsize=(12, 6))
for subject, values in valence_data.items():
    plt.hist(values, bins=10, alpha=0.5, label=subject)

plt.xlabel("Valence Score (0-10)")
plt.ylabel("Frequency")
plt.title("Valence Distribution Across Subjects (Histogram)")
plt.legend(loc="upper right")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 히스토그램 저장
plt.savefig("valence_histogram.png")
print("Saved: valence_histogram.png")

plt.show()
