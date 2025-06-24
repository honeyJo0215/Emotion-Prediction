# import scipy.io as sio
# import numpy as np
# import os

# # 🔹 미리 설정된 입력 폴더 및 출력 폴더 경로
# input_folder = "/home/bcml1/2025_EMOTION/SEED_IV/eye_feature_smooth/3"
# output_folder = "/home/bcml1/2025_EMOTION/SEED_IV/eye_feature_smooth/3_npy"

# def convert_mat_to_npy(input_folder, output_folder):
#     """폴더 내 모든 .mat 파일을 .npy로 변환"""
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     for file_name in os.listdir(input_folder):
#         if file_name.endswith('.mat'):
#             mat_path = os.path.join(input_folder, file_name)
#             npy_path = os.path.join(output_folder, file_name.replace('.mat', '.npy'))

#             # .mat 파일 읽기
#             mat_data = sio.loadmat(mat_path)
#             # 메타데이터 제거
#             data = {key: value for key, value in mat_data.items() if not key.startswith('__')}
#             # .npy 파일 저장
#             np.save(npy_path, data)
#             print(f"{mat_path} → {npy_path} 변환 완료.")

# if __name__ == '__main__':
#     convert_mat_to_npy(input_folder, output_folder)
import scipy.io as sio
import numpy as np
import os

mat_path = "/home/bcml1/2025_EMOTION/SEED_IV/eye_feature_smooth/1/1_20160518.mat"
mat_data = sio.loadmat(mat_path)

print(f"파일명: {os.path.basename(mat_path)}")
print("=" * 60)
for key, value in mat_data.items():
    if key.startswith("__"):
        continue
    print(f"키: {key}")
    print(f"  타입: {type(value)}")
    if hasattr(value, "shape"):
        print(f"  Shape: {value.shape}")
        # 데이터가 numpy array이면 일부 샘플 출력 (첫 5개 값)
        if value.ndim >= 2:
            print("  데이터 예시 (일부):")
            # 2D 이상의 배열일 경우 5x5 미리보기
            preview = value[:5, :5] if value.ndim == 2 else value[:5, :5, ...]
            print(preview)
    else:
        print(f"  값: {value}")
    print("-" * 60)
