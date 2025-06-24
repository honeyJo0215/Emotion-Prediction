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

# === EMCNN 입력 데이터 형태 ===
# 이제 각 sample(세그먼트)의 shape은 (1280, 5): 1280 샘플의 시간축, 5개의 변환 채널
input_shape = (1280, 5)
num_classes = 4  # 감정 클래스 (HVHA, HVLA, LVLA, LVHA)

# === MobileNet 기반 Feature Extraction CNN ===
def emcnn_branch(input_layer):
    """ EMCNN의 각 branch에 해당하는 CNN 블록 (MobileNet 기반) """
    x = Conv1D(8, kernel_size=7, strides=1, padding='same', activation='relu',
               kernel_regularizer=l2(0.001))(input_layer)
    
    # Depthwise Separable Convolution 적용
    x = DepthwiseConv1D(kernel_size=7, strides=4, padding='same', depth_multiplier=1,
                        depthwise_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(16, kernel_size=1, strides=1, padding='same', activation='relu',
               kernel_regularizer=l2(0.001))(x)

    x = DepthwiseConv1D(kernel_size=7, strides=2, padding='same', depth_multiplier=1,
                        depthwise_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(32, kernel_size=1, strides=1, padding='same', activation='relu',
               kernel_regularizer=l2(0.001))(x)

    x = DepthwiseConv1D(kernel_size=7, strides=4, padding='same', depth_multiplier=1,
                        depthwise_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(64, kernel_size=1, strides=1, padding='same', activation='relu',
               kernel_regularizer=l2(0.001))(x)

    x = DepthwiseConv1D(kernel_size=7, strides=2, padding='same', depth_multiplier=1,
                        depthwise_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(128, kernel_size=1, strides=1, padding='same', activation='relu',
               kernel_regularizer=l2(0.001))(x)

    # Global Average Pooling 적용
    x = GlobalAveragePooling1D()(x)
    return x

# =============================================================================
# EMCNN 전체 모델 (입력 shape: (1280, 5))
# =============================================================================
def build_emcnn():
    inputs = Input(shape=input_shape)  # 각 sample의 shape: (1280, 5)

    # Lambda Layer를 사용하여 각 채널(branch)를 추출 (axis=2에서 채널 추출)
    branch1 = emcnn_branch(Lambda(lambda x: tf.expand_dims(x[:, :, 0], axis=-1))(inputs))
    branch2 = emcnn_branch(Lambda(lambda x: tf.expand_dims(x[:, :, 1], axis=-1))(inputs))
    branch3 = emcnn_branch(Lambda(lambda x: tf.expand_dims(x[:, :, 2], axis=-1))(inputs))
    branch4 = emcnn_branch(Lambda(lambda x: tf.expand_dims(x[:, :, 3], axis=-1))(inputs))
    branch5 = emcnn_branch(Lambda(lambda x: tf.expand_dims(x[:, :, 4], axis=-1))(inputs))
    
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
        
        data = np.load(file_path)  # 원래 shape: (51, 5, 1280)
        # 축 변경: (51, 5, 1280) -> (51, 1280, 5)
        data = np.transpose(data, (0, 2, 1))
        try:
            label_str = basename.split('_')[-1].split('.')[0]
            label = int(label_str)
        except:
            continue

        if subject_id not in data_dict:
            data_dict[subject_id] = {'X': [], 'y': []}
        data_dict[subject_id]['X'].append(data)
        data_dict[subject_id]['y'].append(np.full((data.shape[0],), label))
    
    for subj in data_dict:
        data_dict[subj]['X'] = np.concatenate(data_dict[subj]['X'], axis=0)
        data_dict[subj]['y'] = np.concatenate(data_dict[subj]['y'], axis=0)
    return data_dict

# =============================================================================
# 학습 곡선(accuracy, loss) 플롯 저장 함수
# =============================================================================
def plot_training_curves(history, save_path):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
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
# Leave-One-Subject-Out 방식으로 학습 및 평가 (결과는 상위 폴더 "All_Test_Results"에 저장)
# =============================================================================
def train_multimodal_model_leave_one_subject_out(test_subject_id, train_eeg_X, train_ppg_X, train_y,
                                                 test_eeg_X, test_ppg_X, test_y,
                                                 num_classes=4, epochs=150, batch_size=32,
                                                 result_dir_base="/home/bcml1/sigenv/_4주차_eeg+ppg/result_inter_subject"):
    result_dir = os.path.join(result_dir_base, f"test_{test_subject_id}")
    os.makedirs(result_dir, exist_ok=True)
    
    # train/validation 분할 (80:20, stratify 적용)
    eeg_X_train, eeg_X_val, ppg_X_train, ppg_X_val, y_train, y_val = train_test_split(
        train_eeg_X, train_ppg_X, train_y, test_size=0.2, random_state=42, stratify=np.argmax(train_y, axis=1)
    )
    
    # 두 모달리티 모두 증강하여 클래스 균형 맞추기
    eeg_X_train, ppg_X_train, y_train = balance_multimodal_dataset(
        eeg_X_train, ppg_X_train, y_train, noise_std_eeg=0.01, noise_std_ppg=0.01
    )
    eeg_X_val, ppg_X_val, y_val = balance_multimodal_dataset(
        eeg_X_val, ppg_X_val, y_val, noise_std_eeg=0.01, noise_std_ppg=0.01
    )
    test_eeg_X, ppg_X_test, test_y = balance_multimodal_dataset(
        test_eeg_X, test_ppg_X, test_y, noise_std_eeg=0.01, noise_std_ppg=0.01
    )
    
    # 모델 생성 및 컴파일
    eeg_input_shape = train_eeg_X.shape[1:]
    ppg_input_shape = train_ppg_X.shape[1:]
    model = create_multimodal_model(eeg_input_shape, ppg_input_shape, dropout_rate=0.2, num_classes=num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    checkpoint_path = os.path.join(result_dir, "best_model.keras")
    checkpoint_cb = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
    earlystop_cb = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
    
    history = model.fit(
        [eeg_X_train, ppg_X_train], y_train,
        validation_data=([eeg_X_val, ppg_X_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_cb, earlystop_cb],
        verbose=1
    )
    
    save_training_history(history, result_dir, test_subject_id, fold_no=0)  # fold_no 0 표기
    
    model.load_weights(checkpoint_path)
    y_pred_prob = model.predict([test_eeg_X, ppg_X_test])
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_test_labels = np.argmax(test_y, axis=1)
    
    # 전체 테스트에 대한 overall report 생성
    overall_report = classification_report(y_test_labels, y_pred, digits=4)
    overall_cm = confusion_matrix(y_test_labels, y_pred)
    with open(os.path.join(result_dir, "overall_classification_report.txt"), "w") as f:
        f.write(overall_report)
    with open(os.path.join(result_dir, "overall_confusion_matrix.txt"), "w") as f:
        f.write(np.array2string(overall_cm))
    
    # ---- 각 trial(샘플) 별 report 생성 ----
    # 여기서는 test 데이터가 총 40 trial로 구성되어 있다고 가정합니다.
    num_trials = 40
    total_test_samples = len(y_test_labels)
    if total_test_samples % num_trials != 0:
        print("주의: 테스트 샘플 수가 40으로 나누어 떨어지지 않습니다.")
    samples_per_trial = total_test_samples // num_trials
    
    trial_reports = {}
    for trial in range(num_trials):
        start_idx = trial * samples_per_trial
        end_idx = (trial + 1) * samples_per_trial
        trial_true = y_test_labels[start_idx:end_idx]
        trial_pred = y_pred[start_idx:end_idx]
        trial_report = classification_report(trial_true, trial_pred, digits=4)
        trial_cm = confusion_matrix(trial_true, trial_pred)
        trial_reports[trial + 1] = {"report": trial_report, "cm": trial_cm}
        # 저장
        trial_report_path = os.path.join(result_dir, f"trial_{trial+1:02d}_classification_report.txt")
        with open(trial_report_path, "w") as f:
            f.write(trial_report)
        trial_cm_path = os.path.join(result_dir, f"trial_{trial+1:02d}_confusion_matrix.txt")
        with open(trial_cm_path, "w") as f:
            f.write(np.array2string(trial_cm))
    
    print(f"Test subject {test_subject_id} overall evaluation:")
    print(overall_report)
    for trial in range(1, num_trials + 1):
        print(f"Trial {trial:02d} evaluation:")
        print(trial_reports[trial]["report"])
    
    return model, history, overall_report, trial_reports

# def train_subject_leave_one_out(data_dict, epochs=300, batch_size=128):
#     parent_dir = "/home/bcml1/sigenv/_4주차_ppg/21te_1tr_result_nonorm"
#     os.makedirs(parent_dir, exist_ok=True)
    
#     subject_ids = sorted(data_dict.keys())
#     for test_subj in subject_ids:
#         print(f"LOSO - Test Subject: {test_subj}")
#         X_train_list = []
#         y_train_list = []
#         for subj in subject_ids:
#             if subj == test_subj:
#                 continue
#             X_train_list.append(data_dict[subj]['X'])
#             y_train_list.append(data_dict[subj]['y'])
#         X_train = np.concatenate(X_train_list, axis=0)
#         y_train = np.concatenate(y_train_list, axis=0)
#         X_test = data_dict[test_subj]['X']
#         y_test = data_dict[test_subj]['y']
        
#         result_dir = os.path.join(parent_dir, f"test_{test_subj}")
#         os.makedirs(result_dir, exist_ok=True)
        
#         model = build_emcnn()
#         model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.99),
#                       loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#                       metrics=['accuracy'])
        
#         history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
#                             validation_data=(X_test, y_test), verbose=1)
        
#         model.save(os.path.join(result_dir, f"{test_subj}_model.keras"))
#         np.save(os.path.join(result_dir, f"{test_subj}_history.npy"), history.history)
        
#         y_pred = np.argmax(model.predict(X_test), axis=1)
#         report = classification_report(y_test, y_pred)
#         cm = confusion_matrix(y_test, y_pred)
        
#         eval_path = os.path.join(result_dir, f"{test_subj}_evaluation.txt")
#         with open(eval_path, 'w') as f:
#             f.write(report)
#             f.write("\nConfusion Matrix:\n")
#             f.write(str(cm))
        
#         plt.figure(figsize=(6, 5))
#         sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
#         plt.title(f"{test_subj} Confusion Matrix")
#         plt.xlabel("Predicted Label")
#         plt.ylabel("True Label")
#         plt.savefig(os.path.join(result_dir, f"{test_subj}_confusion_matrix.png"))
#         plt.close()
        
#         plot_training_curves(history, os.path.join(result_dir, f"{test_subj}_training_curves.png"))

if __name__ == "__main__":
    # data_dir = '/home/bcml1/2025_EMOTION/DEAP_PPG_10s_1soverlap_4label/'
    data_dir = '/home/bcml1/2025_EMOTION/DEAP_PPG_1s_nonorm'
    data_dict = load_data(data_dir)
    train_multimodal_model_leave_one_subject_out(data_dict, epochs=300, batch_size=128)
