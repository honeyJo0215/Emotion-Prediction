import os
import numpy as np

# npy 파일들이 저장된 폴더
input_dir = '/home/bcml1/2025_EMOTION/SEED_IV/eeg_raw_data/3_npy_sample'

# input_dir 내의 모든 npy 파일을 처리합니다.
for file_name in os.listdir(input_dir):
    if file_name.endswith('.npy'):
        file_path = os.path.join(input_dir, file_name)
        try:
            # allow_pickle=True로 npy 파일을 불러옵니다.
            loaded = np.load(file_path, allow_pickle=True)
            # np.load로 불러온 객체가 dict를 감싸고 있을 수 있으므로 .item()로 추출합니다.
            try:
                obj = loaded.item()
            except Exception:
                obj = loaded

            # 딕셔너리 형태이고 'data' 키가 존재하면, 해당 값(배열)만 추출
            if isinstance(obj, dict) and 'data' in obj:
                data_array = obj['data']
                # 기존 파일 경로에 덮어쓰듯이 배열 형태로 저장합니다.
                np.save(file_path, data_array)
                print(f"Processed {file_name}")
            else:
                print(f"Skipping {file_name}: dict 형태가 아니거나 'data' 키가 없습니다.")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
