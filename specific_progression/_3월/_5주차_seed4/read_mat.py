import scipy.io as sio
import numpy as np
import os

def read_mat_file(file_path):
    """ .mat 파일의 모든 데이터를 읽고 출력하는 함수 """
    try:
        # .mat 파일 로드
        mat_data = sio.loadmat(file_path)

        print(f"\n📂 파일명: {os.path.basename(file_path)}")
        print("=" * 60)

        # 각 변수 출력
        for key, value in mat_data.items():
            if key.startswith("__"):  # MATLAB 메타데이터 제거
                continue

            print(f"🔹 변수명: {key}")
            print(f"   데이터 타입: {type(value)}")

            # numpy 배열인 경우
            if isinstance(value, np.ndarray):
                print(f"   Shape: {value.shape}")

                # 스칼라 값인 경우
                if value.size == 1:
                    print(f"   값: {value.item()}")
                # 1D 배열일 경우 처음 5개 데이터 출력
                elif value.ndim == 1:
                    print(f"   데이터 예시: {value[:5]} ...")
                # 2D 이상의 배열이면 일부만 출력
                else:
                    print(f"   데이터 예시 (5x5 일부):\n{value[:5, :5]} ...")

            else:
                print(f"   값: {value}")

            print("-" * 60)
    
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

# 실행 예시
file_path = "/home/bcml1/2025_EMOTION/SEED_IV/eeg_feature_smooth/1/1_20160518.mat"  # MAT 파일 경로 입력
read_mat_file(file_path)
