import os
import glob
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout, concatenate
from tensorflow.keras.optimizers import Adam

# def load_seediv_data(base_dirs, de_keys=["de_movingAve", "de_LDS"], psd_keys=["psd_movingAve", "psd_LDS"]):
def load_seediv_data(base_dirs, de_keys=["de_movingAve"], psd_keys=["psd_movingAve"]):
    """
    SEED-IV 데이터셋의 numpy 파일들을 로드합니다.
    각 파일은 원래 (62, T, 5) shape이며, 주파수 밴드는 다음 순서로 구성됩니다:
       1) delta (1~4 Hz), 2) theta (4~8 Hz), 3) alpha (8~14 Hz),
       4) beta (14~31 Hz), 5) gamma (31~50 Hz).
    여기서는 첫 번째 밴드인 delta를 제거하여 (62, T, 4) shape로 변환합니다.
    
    파일명 형식: X_XXXXXXXX(피실험자이름)_sample_XX_Key_label_X.npy
    (Key는 "de_movingAve", "de_LDS", "psd_movingAve", "psd_LDS" 중 하나)
    
    DE feature와 PSD 데이터를 각각 로드하여, 두 모달리티 모두 존재하는 trial만 선택합니다.
    
    Parameters:
      base_dirs (list): 실험별 데이터가 위치한 디렉토리 경로 리스트.
      de_keys (list): DE feature 관련 파일의 key.
      psd_keys (list): PSD feature 관련 파일의 key.
      
    Returns:
      de_list (list): 각 trial의 DE 데이터 (shape: (62, T, 4))
      psd_list (list): 각 trial의 PSD 데이터 (shape: (62, T, 4))
      label_list (list): 각 trial의 라벨 (정수형)
    """
    data_de = {}
    data_psd = {}
    labels = {}
    
    for base_dir in base_dirs:
        file_list = glob.glob(os.path.join(base_dir, "*.npy"))
        for file in file_list:
            filename = os.path.basename(file)
            # 파일명 형식: X_XXXXXXXX(피실험자이름)_sample_XX_Key_label_X.npy
            parts = filename.replace('.npy','').split('_')
            if len(parts) < 7:
                continue  # 형식이 맞지 않으면 건너뜁니다.
            subject = parts[1]
            trial = parts[3]
            key_name = parts[4]
            label_str = parts[6]
            uid = f"{subject}_{trial}"
            try:
                label_val = int(label_str)
            except:
                continue
            
            # DE feature 파일인 경우
            if key_name in de_keys:
                arr = np.load(file)  # 원래 shape: (62, T, 5)
                arr = arr[..., 1:]   # delta (index 0) 제거 -> (62, T, 4)
                data_de[uid] = arr
                labels[uid] = label_val
            
            # PSD feature 파일인 경우
            if key_name in psd_keys:
                arr = np.load(file)
                arr = arr[..., 1:]   # delta 제거 -> (62, T, 4)
                data_psd[uid] = arr
                
    # DE와 PSD 데이터가 모두 존재하는 trial만 선택
    common_ids = set(data_de.keys()).intersection(set(data_psd.keys()))
    de_list = []
    psd_list = []
    label_list = []
    for uid in sorted(common_ids):
        de_list.append(data_de[uid])
        psd_list.append(data_psd[uid])
        label_list.append(labels[uid])
    return de_list, psd_list, label_list

def create_seediv_cnn_model(input_shape=(62, None, 4), num_classes=4):
    """
    SEED-IV 데이터셋에 맞춰, DE와 PSD 데이터를 각각 입력받아 CNN을 통해 특징을 추출하고,
    두 branch를 결합하여 num_classes개의 감정을 분류하는 모델을 생성하는 함수입니다.
    
    Parameters:
      input_shape: tuple, (채널 수, 가변 시간 길이, 밴드 수). 기본값은 (62, None, 4)
      num_classes: int, 분류할 클래스 수. 기본값은 4.
      
    Returns:
      model: 컴파일된 Keras Model 객체.
    """
    # DE feature branch
    input_de = Input(shape=input_shape, name='input_de')
    x_de = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_first')(input_de)
    x_de = BatchNormalization(axis=1)(x_de)
    x_de = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(x_de)
    x_de = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_first')(x_de)
    x_de = BatchNormalization(axis=1)(x_de)
    x_de = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(x_de)
    x_de = GlobalAveragePooling2D(data_format='channels_first')(x_de)
    
    # PSD feature branch
    input_psd = Input(shape=input_shape, name='input_psd')
    x_psd = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_first')(input_psd)
    x_psd = BatchNormalization(axis=1)(x_psd)
    x_psd = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(x_psd)
    x_psd = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_first')(x_psd)
    x_psd = BatchNormalization(axis=1)(x_psd)
    x_psd = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(x_psd)
    x_psd = GlobalAveragePooling2D(data_format='channels_first')(x_psd)
    
    # 두 branch의 특징 결합
    merged = concatenate([x_de, x_psd], name='merged_features')
    fc = Dense(128, activation='relu')(merged)
    fc = Dropout(0.5)(fc)
    output = Dense(num_classes, activation='softmax', name='output')(fc)
    
    model = Model(inputs=[input_de, input_psd], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 모델 생성 예시:
# 사용 예시:
base_dirs = [
    "/home/bcml1/2025_EMOTION/SEED_IV/eeg_feature_smooth/1_npy_sample",
    "/home/bcml1/2025_EMOTION/SEED_IV/eeg_feature_smooth/2_npy_sample",
    "/home/bcml1/2025_EMOTION/SEED_IV/eeg_feature_smooth/3_npy_sample"
]
de_data, psd_data, labels = load_seediv_data(base_dirs)
model = create_seediv_cnn_model()
model.summary()
