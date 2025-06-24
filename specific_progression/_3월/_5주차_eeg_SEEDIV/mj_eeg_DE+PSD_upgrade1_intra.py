import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

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

limit_gpu_memory(10000)

# ----------------------------------------------------------------------------- 
# Helper 함수: trial 데이터를 시간 슬라이스로 분리 및 라벨 복제
# -----------------------------------------------------------------------------
def split_trial_to_time_slices(trial):
    """
    단일 trial 데이터의 shape: (62, T, 4)
    각 trial의 시간 슬라이스를 (62, 1, 4)로 분리합니다.
    반환 shape: (T, 62, 1, 4)
    """
    channels, T, bands = trial.shape
    slices = []
    for t in range(T):
        # trial[:, t:t+1, :]의 shape는 (62, 1, 4)
        slices.append(trial[:, t:t+1, :])
    return np.array(slices)

def split_trials_list_to_time_slices(trials_list):
    """
    trials_list: list of trial 배열, 각각 shape: (62, T_i, 4) (T_i가 trial마다 다를 수 있음)
    각 trial에 대해 시간 슬라이스 분리 후, 모두 concatenate하여 
    최종 배열 shape: (sum_i(T_i), 62, 1, 4)
    """
    slices_list = []
    for trial in trials_list:
        slices = split_trial_to_time_slices(trial)
        slices_list.append(slices)
    if slices_list:
        return np.concatenate(slices_list, axis=0)
    else:
        return None

def replicate_labels_for_trials(trials_list, labels):
    """
    trials_list: list of trial 배열, 각각 shape: (62, T_i, 4)
    labels: list/array of trial별 one-hot 라벨 (shape: (num_trials, num_classes))
    각 trial에 대해 T_i번 복제하여, 최종 shape: (sum_i(T_i), num_classes)로 만듭니다.
    """
    replicated = []
    for trial, label in zip(trials_list, labels):
        T = trial.shape[1]  # trial의 시간 길이
        replicated.append(np.repeat(label[np.newaxis, :], T, axis=0))
    if replicated:
        return np.concatenate(replicated, axis=0)
    else:
        return None

# ----------------------------------------------------------------------------- 
# 데이터 로드 함수
# -----------------------------------------------------------------------------
def load_seediv_data(base_dirs, de_keys=["de_movingAve"], psd_keys=["psd_movingAve"]):
    """
    파일명 예:
      1_20160518_sample_01_de_movingAve_label_1.npy
      1_20160518_sample_01_psd_movingAve_label_1.npy
    파일명을 '_'로 split하면 8개의 파트로 구성됨:
      subject = parts[0], trial = parts[3],
      key_name = parts[4] + "_" + parts[5],
      label_str = parts[7]
    각 파일은 (62, T, 5) shape이며, delta 밴드를 제거하여 (62, T, 4)로 변환합니다.
    DE와 PSD 데이터가 모두 있는 trial만 선택합니다.
    
    Returns:
      de_list: list of DE trial 배열 (각 배열 shape: (62, T, 4))
      psd_list: list of PSD trial 배열 (각 배열 shape: (62, T, 4))
      label_list: list of 라벨 (정수)
      subject_list: list of subject id (문자열)
    """
    data_de = {}
    data_psd = {}
    labels = {}
    
    for base_dir in base_dirs:
        file_list = glob.glob(os.path.join(base_dir, "*.npy"))
        for file in file_list:
            filename = os.path.basename(file)
            parts = filename.replace('.npy','').split('_')
            if len(parts) < 8:
                continue
            subject = parts[0]
            trial = parts[3]
            key_name = parts[4] + "_" + parts[5]
            label_str = parts[7]
            try:
                label_val = int(label_str)
            except:
                continue
            arr = np.load(file)  # 원래 shape: (62, T, 5)
            arr = arr[..., 1:]   # delta 제거 → (62, T, 4)
            
            if key_name in de_keys:
                data_de[(subject, trial)] = (arr, label_val)
            elif key_name in psd_keys:
                data_psd[(subject, trial)] = (arr, label_val)
    
    common_ids = set(data_de.keys()).intersection(set(data_psd.keys()))
    de_list = []
    psd_list = []
    label_list = []
    subject_list = []
    for sid in sorted(common_ids):
        subj, trial = sid
        arr_de, label_de = data_de[sid]
        arr_psd, label_psd = data_psd[sid]
        if label_de != label_psd:
            continue
        de_list.append(arr_de)
        psd_list.append(arr_psd)
        label_list.append(label_de)
        subject_list.append(subj)
    
    return de_list, psd_list, label_list, subject_list

# ----------------------------------------------------------------------------- 
# CNN 모델 생성 함수 (입력 shape: (62, 1, 4))
# -----------------------------------------------------------------------------
def create_seediv_cnn_model(input_shape=(62, 1, 4), num_classes=4):
    """
    DE와 PSD 데이터를 각각 입력받아 CNN을 통해 특징을 추출하고,
    두 branch를 결합하여 num_classes개의 감정을 분류하는 모델 생성 함수.
    
    Parameters:
      input_shape: tuple, (채널 수, 1, 밴드 수). 기본값 (62, 1, 4)
      num_classes: 분류할 클래스 수 (기본값 4)
      
    Returns:
      model: 컴파일된 Keras Model 객체.
    """
    # DE branch
    input_de = Input(shape=input_shape, name='input_de')
    x_de = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_first')(input_de)
    x_de = BatchNormalization(axis=1)(x_de)
    x_de = MaxPooling2D(pool_size=(1, 2), data_format='channels_first')(x_de)
    x_de = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_first')(x_de)
    x_de = BatchNormalization(axis=1)(x_de)
    x_de = MaxPooling2D(pool_size=(1, 2), data_format='channels_first')(x_de)
    x_de = GlobalAveragePooling2D(data_format='channels_first')(x_de)
    
    # PSD branch
    input_psd = Input(shape=input_shape, name='input_psd')
    x_psd = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_first')(input_psd)
    x_psd = BatchNormalization(axis=1)(x_psd)
    x_psd = MaxPooling2D(pool_size=(1, 2), data_format='channels_first')(x_psd)
    x_psd = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', data_format='channels_first')(x_psd)
    x_psd = BatchNormalization(axis=1)(x_psd)
    x_psd = MaxPooling2D(pool_size=(1, 2), data_format='channels_first')(x_psd)
    x_psd = GlobalAveragePooling2D(data_format='channels_first')(x_psd)
    
    merged = concatenate([x_de, x_psd], name='merged_features')
    fc = Dense(128, activation='relu')(merged)
    fc = Dropout(0.5)(fc)
    output = Dense(num_classes, activation='softmax', name='output')(fc)
    
    model = Model(inputs=[input_de, input_psd], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ----------------------------------------------------------------------------- 
# Intra-subject cross-validation with trial-level split 후, 시간 슬라이스 단위로 분리하여 학습
# ----------------------------------------------------------------------------- 
# -----------------------------------------------------------------------------
# LOSO (Inter-subject) Cross-Validation 함수
# -----------------------------------------------------------------------------
def loso_cv_training(base_dirs, result_dir, epochs=150, batch_size=16):
    """
    SEED‑IV 데이터셋을 LOSO 방식(각 subject를 한 번씩 테스트셋으로 사용)으로 학습하는 함수입니다.
    나머지 subject들의 trial을 합쳐 training/validation 셋으로 사용하며, pre‑training과 fine‑tuning을 적용합니다.
    또한, 각 subject별 confusion matrix, classification report, training curves, 모델 파일을 저장합니다.
    """
    os.makedirs(result_dir, exist_ok=True)
    overall_folder = os.path.join(result_dir, "overall")
    os.makedirs(overall_folder, exist_ok=True)
    
    # 데이터 로드
    de_trials, psd_trials, label_list, subject_list = load_seediv_data(base_dirs)
    
    # subject별로 데이터 그룹화
    subject_data = {}
    for de, psd, label, subj in zip(de_trials, psd_trials, label_list, subject_list):
        if subj not in subject_data:
            subject_data[subj] = {"de": [], "psd": [], "labels": []}
        subject_data[subj]["de"].append(de)
        subject_data[subj]["psd"].append(psd)
        subject_data[subj]["labels"].append(label)
    
    overall_acc = {}
    overall_reports = {}
    
    # LOSO: 각 subject를 테스트 셋으로 하고, 나머지를 training/validation으로 사용
    for test_subj in subject_data.keys():
        print(f"\n========== LOSO: Test subject: {test_subj} ==========")
        # 나머지 subject들의 데이터를 training으로 사용
        train_de, train_psd, train_labels = [], [], []
        for subj, data in subject_data.items():
            if subj != test_subj:
                train_de.extend(data["de"])
                train_psd.extend(data["psd"])
                train_labels.extend(data["labels"])
        
        # 테스트 셋: 해당 subject의 모든 trial
        test_de = subject_data[test_subj]["de"]
        test_psd = subject_data[test_subj]["psd"]
        test_labels = subject_data[test_subj]["labels"]
        
        # trial 단위 one-hot 인코딩
        y_train_trials = to_categorical(np.array(train_labels), num_classes=4)
        y_test_trials = to_categorical(np.array(test_labels), num_classes=4)
        
        # training subject들의 trial 중 10%를 validation 셋으로 분할
        train_indices, val_indices = train_test_split(
            np.arange(len(train_de)), test_size=0.1, random_state=42, stratify=np.array(train_labels)
        )
        train_de_trials = [train_de[i] for i in train_indices]
        train_psd_trials = [train_psd[i] for i in train_indices]
        val_de_trials = [train_de[i] for i in val_indices]
        val_psd_trials = [train_psd[i] for i in val_indices]
        y_train_split = y_train_trials[train_indices]
        y_val_split = y_train_trials[val_indices]
        
        # 시간 슬라이스 변환 및 trial별 라벨 복제
        X_de_train = split_trials_list_to_time_slices(train_de_trials)
        X_psd_train = split_trials_list_to_time_slices(train_psd_trials)
        y_train = replicate_labels_for_trials(train_de_trials, y_train_split)
        
        X_de_val = split_trials_list_to_time_slices(val_de_trials)
        X_psd_val = split_trials_list_to_time_slices(val_psd_trials)
        y_val = replicate_labels_for_trials(val_de_trials, y_val_split)
        
        X_de_test = split_trials_list_to_time_slices(test_de)
        X_psd_test = split_trials_list_to_time_slices(test_psd)
        y_test = replicate_labels_for_trials(test_de, y_test_trials)
        
        # Dataset 생성
        train_dataset = tf.data.Dataset.from_tensor_slices(((X_de_train, X_psd_train), y_train)).shuffle(1000).batch(batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices(((X_de_val, X_psd_val), y_val)).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices(((X_de_test, X_psd_test), y_test)).batch(batch_size)
        
        # 모델 생성
        model = create_seediv_cnn_model(input_shape=(62, 1, 4), num_classes=4)
        model.summary()
        
        # Early Stopping 설정
        callbacks = [EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)]
        # callbacks = [EarlyStopping(monitor='val_accuracy', patience=40, min_delta=0.0001, restore_best_weights=True)] -> 위의 것보다 성능이 낮게 나옴
        # --- 1. Pre-training Phase ---
        pretrain_epochs = int(epochs * 0.6)
        finetune_epochs = epochs - pretrain_epochs
        
        print(f"Subject {test_subj}: Starting pre-training phase for {pretrain_epochs} epochs...")
        history_pretrain = model.fit(train_dataset, epochs=pretrain_epochs, validation_data=val_dataset, callbacks=callbacks, verbose=1)
        
        # --- 2. Fine-tuning Phase ---
        # feature extractor 계층 동결, 분류기(Dense) 계층은 학습
        for layer in model.layers:
            if isinstance(layer, (tf.keras.layers.Conv2D,
                                  tf.keras.layers.BatchNormalization,
                                  tf.keras.layers.MaxPooling2D,
                                  tf.keras.layers.GlobalAveragePooling2D)):
                layer.trainable = False
            else:
                layer.trainable = True
        
        model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"Subject {test_subj}: Starting fine-tuning phase for {finetune_epochs} epochs...")
        history_finetune = model.fit(train_dataset, epochs=finetune_epochs, validation_data=val_dataset, callbacks=callbacks, verbose=1)
        
        # Training curves 저장 (Pre-training + Fine-tuning)
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        total_loss = history_pretrain.history['loss'] + history_finetune.history['loss']
        total_val_loss = history_pretrain.history['val_loss'] + history_finetune.history['val_loss']
        plt.plot(total_loss, label='Train Loss')
        plt.plot(total_val_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curve')
        plt.subplot(1,2,2)
        total_acc = history_pretrain.history['accuracy'] + history_finetune.history['accuracy']
        total_val_acc = history_pretrain.history['val_accuracy'] + history_finetune.history['val_accuracy']
        plt.plot(total_acc, label='Train Accuracy')
        plt.plot(total_val_acc, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy Curve')
        train_curve_path = os.path.join(result_dir, f"s{test_subj.zfill(2)}", "training_curves.png")
        plt.savefig(train_curve_path)
        plt.close()
        print(f"Subject {test_subj}: Training curves saved to {train_curve_path}")
        
        # 평가
        test_loss, test_acc = model.evaluate(test_dataset, verbose=0)
        print(f"Subject {test_subj}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        overall_acc[test_subj] = test_acc
        
        # 예측 결과, confusion matrix, classification report 생성
        y_pred_prob = model.predict(test_dataset)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(np.concatenate([y for _, y in test_dataset], axis=0), axis=1)
        
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, digits=2)
        overall_reports[test_subj] = report
        
        # Confusion Matrix와 Classification Report 저장
        subj_folder = os.path.join(result_dir, f"s{test_subj.zfill(2)}")
        os.makedirs(subj_folder, exist_ok=True)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        cm_path = os.path.join(subj_folder, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        print(f"Subject {test_subj}: Confusion matrix saved to {cm_path}")
        report_path = os.path.join(subj_folder, "classification.txt")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Subject {test_subj}: Classification report saved to {report_path}")
        
        # 모델 저장
        model_save_path = os.path.join(subj_folder, "model_eeg_cnn.keras")
        model.save(model_save_path)
        print(f"Subject {test_subj}: Model saved to {model_save_path}")
    
    # 전체 결과 집계 및 저장
    overall_avg_acc = np.mean(list(overall_acc.values()))
    overall_report_path = os.path.join(overall_folder, "overall_classification.txt")
    with open(overall_report_path, "w") as f:
        f.write("Overall LOSO Test Accuracy: {:.4f}\n\n".format(overall_avg_acc))
        for subj in sorted(overall_reports.keys()):
            f.write(f"Subject {subj}:\n")
            f.write(overall_reports[subj])
            f.write("\n\n")
    print(f"Overall results saved to {overall_report_path}")

# -----------------------------------------------------------------------------
# 사용 예시:
# -----------------------------------------------------------------------------
base_dirs = [
    "/home/bcml1/2025_EMOTION/SEED_IV/eeg_feature_smooth/1_npy_sample",
    "/home/bcml1/2025_EMOTION/SEED_IV/eeg_feature_smooth/2_npy_sample",
    "/home/bcml1/2025_EMOTION/SEED_IV/eeg_feature_smooth/3_npy_sample"
]
RESULT_DIR = "/home/bcml1/sigenv/_5주차_eeg_SEEDIV/DE+PSD_up1_inter"
loso_cv_training(base_dirs, RESULT_DIR, epochs=150, batch_size=16)