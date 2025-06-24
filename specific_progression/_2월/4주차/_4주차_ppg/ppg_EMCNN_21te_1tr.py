import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, DepthwiseConv1D, Dense, GlobalAveragePooling1D, concatenate, BatchNormalization, Activation, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# GPU 메모리 제한 (필요 시)
# =============================================================================
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

limit_gpu_memory(8000)

# =============================================================================
# 모델 및 EMCNN 구성 (원본 코드와 동일)
# =============================================================================
input_shape = (5, 1280)  # 입력 데이터 형태
num_classes = 4         # 감정 클래스 수

def emcnn_branch(input_layer):
    x = Conv1D(8, kernel_size=7, strides=1, padding='same',
               activation='relu', kernel_regularizer=l2(0.001))(input_layer)
    x = DepthwiseConv1D(kernel_size=7, strides=4, padding='same', depth_multiplier=1,
                        depthwise_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(16, kernel_size=1, strides=1, padding='same',
               activation='relu', kernel_regularizer=l2(0.001))(x)

    x = DepthwiseConv1D(kernel_size=7, strides=2, padding='same', depth_multiplier=1,
                        depthwise_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(32, kernel_size=1, strides=1, padding='same',
               activation='relu', kernel_regularizer=l2(0.001))(x)

    x = DepthwiseConv1D(kernel_size=7, strides=4, padding='same', depth_multiplier=1,
                        depthwise_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(64, kernel_size=1, strides=1, padding='same',
               activation='relu', kernel_regularizer=l2(0.001))(x)

    x = DepthwiseConv1D(kernel_size=7, strides=2, padding='same', depth_multiplier=1,
                        depthwise_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(128, kernel_size=1, strides=1, padding='same',
               activation='relu', kernel_regularizer=l2(0.001))(x)

    x = GlobalAveragePooling1D()(x)
    return x

def build_emcnn():
    inputs = Input(shape=(5, 1280))
    # 각 branch에 대해 Lambda 레이어로 차원 확장 후 CNN 적용
    branch1 = emcnn_branch(Lambda(lambda x: tf.expand_dims(x[:, 0, :], axis=-1))(inputs))
    branch2 = emcnn_branch(Lambda(lambda x: tf.expand_dims(x[:, 1, :], axis=-1))(inputs))
    branch3 = emcnn_branch(Lambda(lambda x: tf.expand_dims(x[:, 2, :], axis=-1))(inputs))
    branch4 = emcnn_branch(Lambda(lambda x: tf.expand_dims(x[:, 3, :], axis=-1))(inputs))
    branch5 = emcnn_branch(Lambda(lambda x: tf.expand_dims(x[:, 4, :], axis=-1))(inputs))
    
    merged = concatenate([branch1, branch2, branch3, branch4, branch5])
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(merged)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# =============================================================================
# 데이터 로드 함수 (파일명에 subject id가 포함되어 있다고 가정)
# =============================================================================
def load_data(data_dir):
    # .npy 파일 리스트
    file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
    data_dict = {}  # subject id를 key로 하는 딕셔너리

    for file_path in file_paths:
        basename = os.path.basename(file_path)
        # 예: "data_s01_label0.npy"와 같이 파일명에 sXX가 포함되어 있다고 가정
        subject_id = None
        for part in basename.split('_'):
            if part.startswith('s') and len(part) == 3:
                subject_id = part
                break
        if subject_id is None:
            continue
        
        data = np.load(file_path)
        # 파일명 마지막 부분에 label 정보가 있다고 가정 (예: "label0")
        try:
            label_str = basename.split('_')[-1].split('.')[0]
            label = int(label_str)
        except:
            continue

        if subject_id not in data_dict:
            data_dict[subject_id] = {'X': [], 'y': []}
        data_dict[subject_id]['X'].append(data)
        data_dict[subject_id]['y'].append(np.full((data.shape[0],), label))
    
    # 각 subject별로 데이터를 합치기
    for subj in data_dict:
        data_dict[subj]['X'] = np.concatenate(data_dict[subj]['X'], axis=0)
        data_dict[subj]['y'] = np.concatenate(data_dict[subj]['y'], axis=0)
    return data_dict

# =============================================================================
# 학습 곡선(accuracy, loss) 플롯 저장 함수
# =============================================================================
def plot_training_curves(history, save_path):
    plt.figure(figsize=(12, 5))
    # Accuracy curve
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    # Loss curve
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# =============================================================================
# Leave-One-Subject-Out 방식으로 학습 및 평가 (상위 폴더 "All_Test_Results" 안에 저장)
# =============================================================================
def train_subject_leave_one_out(data_dict, epochs=300, batch_size=128):
    # 상위 결과 폴더 생성
    parent_dir = "/home/bcml1/sigenv/_4주차_ppg/21te_1tr_result"
    os.makedirs(parent_dir, exist_ok=True)

    # subject id는 예: s01, s02, ... s22
    subject_ids = sorted(data_dict.keys())
    for test_subj in subject_ids:
        print(f"Subject {test_subj}을(를) 테스트 데이터로 사용하여 학습합니다.")
        # 학습 데이터: test_subj를 제외한 모든 subject의 데이터 합치기
        X_train_list = []
        y_train_list = []
        for subj in subject_ids:
            if subj == test_subj:
                continue
            X_train_list.append(data_dict[subj]['X'])
            y_train_list.append(data_dict[subj]['y'])
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        X_test = data_dict[test_subj]['X']
        y_test = data_dict[test_subj]['y']
        
        # 결과 저장 폴더 생성 (예: All_Test_Results/test_s01)
        result_dir = os.path.join(parent_dir, f"test_{test_subj}")
        os.makedirs(result_dir, exist_ok=True)
        
        model = build_emcnn()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.99),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
        
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(X_test, y_test), verbose=1)
        
        # 모델 저장
        model.save(os.path.join(result_dir, f"{test_subj}_model.keras"))
        # 학습 히스토리 저장
        np.save(os.path.join(result_dir, f"{test_subj}_history.npy"), history.history)
        
        # 테스트 데이터에 대한 평가 수행
        y_pred = np.argmax(model.predict(X_test), axis=1)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        # 평가 결과 파일 저장
        eval_path = os.path.join(result_dir, f"{test_subj}_evaluation.txt")
        with open(eval_path, 'w') as f:
            f.write(report)
            f.write("\nConfusion Matrix:\n")
            f.write(str(cm))
        
        # 혼동행렬 그림 저장
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{test_subj} Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.savefig(os.path.join(result_dir, f"{test_subj}_confusion_matrix.png"))
        plt.close()
        
        # 학습 곡선 (accuracy & loss) 플롯 저장
        plot_training_curves(history, os.path.join(result_dir, f"{test_subj}_training_curves.png"))

if __name__ == "__main__":
    # 데이터 디렉토리 (파일명에 sXX가 포함되어 있다고 가정)
    data_dir = '/home/bcml1/2025_EMOTION/DEAP_PPG_10s_1soverlap_4label/'
    data_dict = load_data(data_dir)
    train_subject_leave_one_out(data_dict, epochs=300, batch_size=128)
