import numpy as np
import matplotlib.pyplot as plt
import os

# 데이터 경로 설정
base_dir = "/home/bcml1/2025_EMOTION/DEAP_npy_files+label"
subject_files = [f"s{str(i).zfill(2)}_labels.npy" for i in range(1, 23)]

# 모든 subject의 valence 값 저장 리스트
valence_data = {}

# 각 subject 파일에서 첫 번째 열(Valence 값, 40개 샘플) 추출
all_valence_values = []
for file_name in subject_files:
    file_path = os.path.join(base_dir, file_name)
    if os.path.exists(file_path):
        labels_data = np.load(file_path, allow_pickle=True)
        valence_data[file_name.split("_")[0]] = labels_data[:, 0]  # 첫 번째 열(Valence) 값 저장
        all_valence_values.extend(labels_data[:, 0])  # 전체 valence 값 저장
    else:
        print(f"Warning: {file_path} not found!")

# 전체 데이터를 기반으로 3등분하는 구간 설정
all_valence_values = np.array(all_valence_values)
lower_bound = np.percentile(all_valence_values, 33)  # 1/3 지점
upper_bound = np.percentile(all_valence_values, 66)  # 2/3 지점

print(f"Automatically calculated ranges:\nNegative: <= {lower_bound:.2f}, Neutral: {lower_bound:.2f} ~ {upper_bound:.2f}, Positive: >= {upper_bound:.2f}")

# 각 subject에 대해 "Negative", "Neutral", "Positive" 비율 확인
category_counts = {"Negative": 0, "Neutral": 0, "Positive": 0}
categorized_data = {}

for subject, values in valence_data.items():
    negative = values <= lower_bound
    neutral = (values > lower_bound) & (values <= upper_bound)
    positive = values > upper_bound

    categorized_data[subject] = {
        "Negative": values[negative],
        "Neutral": values[neutral],
        "Positive": values[positive]
    }

    category_counts["Negative"] += len(values[negative])
    category_counts["Neutral"] += len(values[neutral])
    category_counts["Positive"] += len(values[positive])

# 시각화
plt.figure(figsize=(12, 6))

# Stacked Histogram으로 3가지 범주를 시각화
plt.hist(all_valence_values, bins=20, color="gray", alpha=0.2, label="All Valence")
plt.axvline(lower_bound, color="blue", linestyle="dashed", label="Negative/Neutral Boundary")
plt.axvline(upper_bound, color="red", linestyle="dashed", label="Neutral/Positive Boundary")

plt.xlabel("Valence Score (0-10)")
plt.ylabel("Frequency")
plt.title("Valence Score Categorization")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 그래프 저장
plt.savefig("valence_categorization.png")
print("Saved: valence_categorization.png")

# 카테고리별 비율 출력
print(f"\nFinal Categorization Counts:\n{category_counts}")
