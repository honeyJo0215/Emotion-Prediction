import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from keras.layers import (Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization,
                          Flatten, Dense, Reshape, Average)
from keras.models import Sequential, Model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf

# -----------------------------------------------------------------------------
# GPU 메모리 제한 (필요 시)
# -----------------------------------------------------------------------------
def limit_gpu_memory(memory_limit_mib=8000):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit_mib)]
            )
            print(f"GPU memory limited to {memory_limit_mib} MiB.")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU available, using CPU.")

limit_gpu_memory(5000)

# -----------------------------------------------------------------------------
# [1] 상수 및 입력 형상 정의
# -----------------------------------------------------------------------------
ppg_input_shape = (1280, 5)   # PPG branch 입력: (1280, 5)
eeg_input_shape = (8, 9, 8)    # EEG branch 입력: (8, 9, 8)
num_classes = 4              # 4 감정 클래스

# -----------------------------------------------------------------------------
# [2] PPG Branch: EMCNN Feature Extractor (논문 구현 그대로)
# -----------------------------------------------------------------------------
def emcnn_branch(input_layer):
    x = tf.keras.layers.Conv1D(8, kernel_size=7, strides=1, padding='same',
                               activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(input_layer)
    x = tf.keras.layers.DepthwiseConv1D(kernel_size=7, strides=4, padding='same',
                                        depth_multiplier=1, depthwise_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv1D(16, kernel_size=1, strides=1, padding='same',
                               activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)

    x = tf.keras.layers.DepthwiseConv1D(kernel_size=7, strides=2, padding='same',
                                        depth_multiplier=1, depthwise_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv1D(32, kernel_size=1, strides=1, padding='same',
                               activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)

    x = tf.keras.layers.DepthwiseConv1D(kernel_size=7, strides=4, padding='same',
                                        depth_multiplier=1, depthwise_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv1D(64, kernel_size=1, strides=1, padding='same',
                               activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)

    x = tf.keras.layers.DepthwiseConv1D(kernel_size=7, strides=2, padding='same',
                                        depth_multiplier=1, depthwise_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv1D(128, kernel_size=1, strides=1, padding='same',
                               activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    return x

def build_ppg_branch(input_shape=ppg_input_shape):
    inputs = Input(shape=input_shape, name="PPG_Input")
    branch_outputs = []
    # 각 5 채널에 대해 개별 branch 적용
    for i in range(input_shape[1]):
        channel_i = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[:, :, i], axis=-1))(inputs)
        feat_i = emcnn_branch(channel_i)
        branch_outputs.append(feat_i)
    merged = tf.keras.layers.concatenate(branch_outputs, name="PPG_Features")
    features = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001),
                     name="PPG_Feature_Vector")(merged)
    model = Model(inputs=inputs, outputs=features, name="PPG_Branch")
    return model

# -----------------------------------------------------------------------------
# [3] EEG Branch: CNN Feature Extractor (MT_CNN 기반)
# -----------------------------------------------------------------------------
def create_base_network(input_dim, dropout_rate):
    seq = Sequential(name="EEG_Base")
    seq.add(Conv2D(64, 5, activation='relu', padding='same', input_shape=input_dim, name='conv1'))
    seq.add(BatchNormalization())
    seq.add(Dropout(dropout_rate))
    seq.add(Conv2D(128, 4, activation='relu', padding='same', name='conv2'))
    seq.add(BatchNormalization())
    seq.add(Dropout(dropout_rate))
    seq.add(Conv2D(256, 4, activation='relu', padding='same', name='conv3'))
    seq.add(BatchNormalization())
    seq.add(Dropout(dropout_rate))
    seq.add(Conv2D(64, 1, activation='relu', padding='same', name='conv4'))
    seq.add(MaxPooling2D(2, 2, name='pool1'))
    seq.add(BatchNormalization())
    seq.add(Dropout(dropout_rate))
    seq.add(Flatten(name='flatten'))
    seq.add(Dense(512, activation='relu', name='dense1'))
    seq.add(Reshape((1, 512), name='reshape'))
    seq.add(BatchNormalization())
    seq.add(Dropout(dropout_rate))
    return seq

def build_eeg_branch(img_size=eeg_input_shape, dropout_rate=0.2, number_of_inputs=1):
    base_network = create_base_network(img_size, dropout_rate)
    inputs = [Input(shape=img_size, name=f"EEG_Input_{i}") for i in range(number_of_inputs)]
    if number_of_inputs == 1:
        x = base_network(inputs[0])
    else:
        x = Average(name="EEG_Avg")([base_network(inp) for inp in inputs])
    x = Flatten(name="EEG_Flatten")(x)
    features = Dense(128, activation='relu', name="EEG_Feature_Vector")(x)
    model = Model(inputs=inputs, outputs=features, name="EEG_Branch")
    return model

# -----------------------------------------------------------------------------
# [4] 멀티모달 모델: PPG와 EEG branch 특징 결합
# -----------------------------------------------------------------------------
def build_multimodal_model(ppg_input_shape=ppg_input_shape,
                           eeg_input_shape=eeg_input_shape, dropout_rate=0.2):
    ppg_model = build_ppg_branch(ppg_input_shape)
    eeg_model = build_eeg_branch(eeg_input_shape, dropout_rate=dropout_rate, number_of_inputs=1)
    combined_features = tf.keras.layers.concatenate([ppg_model.output, eeg_model.output],
                                                    name="Fused_Features")
    x = Dense(128, activation='relu', name="fusion_dense1")(combined_features)
    x = Dropout(0.5, name="fusion_dropout")(x)
    x = Dense(64, activation='relu', name="fusion_dense2")(x)
    output = Dense(num_classes, activation='softmax', name="final_output")(x)
    model = Model(inputs=[ppg_model.input, eeg_model.input], outputs=output, name="Multimodal_Model")
    return model

# -----------------------------------------------------------------------------
# [5] 실제 데이터 로드 함수
# -----------------------------------------------------------------------------
# PPG 데이터 로드 (논문 PPG 코드 참조)
def load_ppg_data_for_subject(ppg_data_dir, subject_id):
    file_paths = [os.path.join(ppg_data_dir, f) for f in os.listdir(ppg_data_dir)
                  if f.startswith(subject_id)]
    X_list = []
    y_list = []
    for file_path in file_paths:
        data = np.load(file_path)  # 원래 shape: (51, 5, 1280)
        data = np.transpose(data, (0, 2, 1))  # -> (51, 1280, 5)
        label = int(file_path.split('_')[-1].split('.')[0])
        X_list.append(data)
        y_list.append(np.full((data.shape[0],), label))
    if len(X_list) == 0:
        raise ValueError(f"{subject_id}에 유효한 PPG 데이터가 없습니다.")
    X = np.concatenate(X_list, axis=0)  # (num_samples, 1280, 5)
    y = np.concatenate(y_list, axis=0)
    return X, to_categorical(y, num_classes=num_classes)

# EEG 데이터 로드 (논문 EEG 로드 코드 참조)
def load_eeg_data_for_subject(subject_id, eeg_data_dir, label_dir):
    pattern = os.path.join(eeg_data_dir, f"{subject_id}_sample_*_segment_*_2D_DE.npy")
    file_list = glob.glob(pattern)
    if len(file_list) == 0:
        raise ValueError(f"{subject_id}에 해당하는 EEG 데이터 파일이 없습니다.")
    label_file = os.path.join(label_dir, f"{subject_id}_emotion_labels.npy")
    try:
        label_data = np.load(label_file, allow_pickle=True)
    except Exception as e:
        raise ValueError(f"{subject_id} 라벨 파일 로드 실패: {e}")
    
    X_list = []
    y_list = []
    regex = re.compile(r"(s\d{2})_sample_(\d{2})_segment_(\d{3})_label_.*_2D_DE\.npy")
    for file in file_list:
        basename = os.path.basename(file)
        m = regex.match(basename)
        if not m:
            print(f"패턴에 맞지 않는 파일명: {basename}")
            continue
        subj, sample_str, segment = m.group(1), m.group(2), m.group(3)
        sample_idx = int(sample_str)
        if sample_idx >= label_data.shape[0]:
            print(f"sample index {sample_idx}가 라벨 파일 범위를 초과함: {basename}")
            continue
        try:
            data = np.load(file)
        except Exception as e:
            print(f"파일 로드 에러 {file}: {e}")
            continue
        X_list.append(data)  # 원래 shape: (51, 5, 1280)
        label_value = label_data[sample_idx]
        if hasattr(label_value, 'shape') and label_value.shape != ():
            label_value = int(label_value[0])
        else:
            label_value = int(label_value)
        y_list.append(label_value)
    if len(X_list) == 0:
        raise ValueError(f"{subject_id}에 유효한 EEG 데이터가 없습니다.")
    X = np.concatenate(X_list, axis=0)
    # EEG 데이터는 본 예제에서 이미 전처리되어 (num_samples, 8, 9, 8) shape로 저장되어 있다고 가정합니다.
    return X, to_categorical(np.array(y_list), num_classes=num_classes)

# -----------------------------------------------------------------------------
# [6] 학습 및 평가 루프: subject별 멀티모달 학습 (Intra-Subject)
# -----------------------------------------------------------------------------
def train_multimodal_model(subject_id, X_ppg, X_eeg, y, epochs=300, batch_size=128):
    print(f"Subject {subject_id} 멀티모달 학습 시작")
    X_ppg_train, X_ppg_test, X_eeg_train, X_eeg_test, y_train, y_test = train_test_split(
        X_ppg, X_eeg, y, test_size=0.3, stratify=np.argmax(y, axis=1), random_state=42)
    
    model = build_multimodal_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.99),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit([X_ppg_train, X_eeg_train], y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=([X_ppg_test, X_eeg_test], y_test), verbose=1)
    
    result_dir = os.path.join("/home/bcml1/sigenv/_4주차_eeg/multimodal_results", subject_id)
    os.makedirs(result_dir, exist_ok=True)
    model.save(os.path.join(result_dir, f"{subject_id}_multimodal_model.keras"))
    np.save(os.path.join(result_dir, f"{subject_id}_multimodal_history.npy"), history.history)
    
    y_pred = np.argmax(model.predict([X_ppg_test, X_eeg_test]), axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    plt.figure()
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title(f"{subject_id} Confusion Matrix")
    plt.savefig(os.path.join(result_dir, f"{subject_id}_confusion_matrix.png"))
    plt.close()
    
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.savefig(os.path.join(result_dir, f"{subject_id}_loss.png"))
    plt.close()
    
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.savefig(os.path.join(result_dir, f"{subject_id}_accuracy.png"))
    plt.close()
    
    with open(os.path.join(result_dir, f"{subject_id}_evaluation.txt"), 'w') as f:
        f.write(classification_report(y_true, y_pred))
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(y_true, y_pred)))
    print(f"Subject {subject_id} 학습 완료")

# -----------------------------------------------------------------------------
# [7] Main 실행: 각 subject에 대해 EEG와 PPG 데이터를 로드 후 멀티모달 학습 수행
# -----------------------------------------------------------------------------
def main():
    eeg_data_dir = "/home/bcml1/2025_EMOTION/DEAP_EEG_2D_DE"
    ppg_data_dir = "/home/bcml1/2025_EMOTION/DEAP_PPG_10s_1soverlap_4label"
    label_dir = "/home/bcml1/2025_EMOTION/DEAP_four_labels"
    results_root = "/home/bcml1/sigenv/_4주차_eeg/multimodal_results"
    os.makedirs(results_root, exist_ok=True)
    
    subjects = [f"s{str(i).zfill(2)}" for i in range(1, 23)]
    for subject in subjects:
        print(f"====== {subject} 멀티모달 학습 시작 ======")
        try:
            X_eeg, y_eeg = load_eeg_data_for_subject(subject, eeg_data_dir, label_dir)
            X_ppg, y_ppg = load_ppg_data_for_subject(ppg_data_dir, subject)
            # 두 modality의 레이블이 동일하다고 가정 (여기서는 EEG 레이블 사용)
            y = y_eeg
        except ValueError as e:
            print(e)
            continue
        
        subject_results_dir = os.path.join(results_root, subject)
        os.makedirs(subject_results_dir, exist_ok=True)
        train_multimodal_model(subject, X_ppg, X_eeg, y, epochs=150, batch_size=32)
        print(f"====== {subject} 멀티모달 학습 완료 ======")

if __name__ == "__main__":
    main()
