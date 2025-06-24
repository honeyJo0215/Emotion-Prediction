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
from tensorflow.keras.callbacks import EarlyStopping, Callback

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
    channels, T, bands = trial.shape
    slices = [trial[:, t:t+1, :] for t in range(T)]
    return np.array(slices)

def split_trials_list_to_time_slices(trials_list):
    slices_list = [split_trial_to_time_slices(trial) for trial in trials_list]
    return np.concatenate(slices_list, axis=0) if slices_list else None

def replicate_labels_for_trials(trials_list, labels):
    replicated = [np.repeat(label[np.newaxis, :], trial.shape[1], axis=0) for trial, label in zip(trials_list, labels)]
    return np.concatenate(replicated, axis=0) if replicated else None

# -----------------------------------------------------------------------------
# 데이터 로드 함수
# -----------------------------------------------------------------------------
def load_seediv_data(base_dirs, de_keys=["de_movingAve"], psd_keys=["psd_movingAve"]):
    data_de = {}
    data_psd = {}
    for base_dir in base_dirs:
        file_list = glob.glob(os.path.join(base_dir, "*.npy"))
        for file in file_list:
            filename = os.path.basename(file)
            parts = filename.replace('.npy','').split('_')
            if len(parts) < 8:
                continue
            subject, trial = parts[0], parts[3]
            key_name = parts[4] + "_" + parts[5]
            try:
                label_val = int(parts[7])
            except:
                continue
            arr = np.load(file)[..., 1:]  # delta 제거
            if key_name in de_keys:
                data_de[(subject, trial)] = (arr, label_val)
            elif key_name in psd_keys:
                data_psd[(subject, trial)] = (arr, label_val)
    common_ids = set(data_de.keys()).intersection(set(data_psd.keys()))
    de_list, psd_list, label_list, subject_list = [], [], [], []
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
# CNN 모델 생성 함수
# -----------------------------------------------------------------------------
def create_seediv_cnn_model(input_shape=(62, 1, 4), num_classes=4):
    # DE branch
    input_de = Input(shape=input_shape, name='input_de')
    x_de = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(input_de)
    x_de = BatchNormalization(axis=1)(x_de)
    x_de = MaxPooling2D((1, 2), data_format='channels_first')(x_de)
    x_de = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(x_de)
    x_de = BatchNormalization(axis=1)(x_de)
    x_de = MaxPooling2D((1, 2), data_format='channels_first')(x_de)
    x_de = GlobalAveragePooling2D(data_format='channels_first')(x_de)
    
    # PSD branch
    input_psd = Input(shape=input_shape, name='input_psd')
    x_psd = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(input_psd)
    x_psd = BatchNormalization(axis=1)(x_psd)
    x_psd = MaxPooling2D((1, 2), data_format='channels_first')(x_psd)
    x_psd = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(x_psd)
    x_psd = BatchNormalization(axis=1)(x_psd)
    x_psd = MaxPooling2D((1, 2), data_format='channels_first')(x_psd)
    x_psd = GlobalAveragePooling2D(data_format='channels_first')(x_psd)
    
    merged = concatenate([x_de, x_psd], name='merged_features')
    fc = Dense(128, activation='relu')(merged)
    fc = Dropout(0.3)(fc)
    output = Dense(num_classes, activation='softmax', name='output')(fc)
    
    model = Model(inputs=[input_de, input_psd], outputs=output)
    # model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -----------------------------------------------------------------------------
# Custom Callback for Gradual Unfreezing
# -----------------------------------------------------------------------------
class GradualUnfreeze(Callback):
    def __init__(self, unfreeze_epoch, layers_to_unfreeze, new_learning_rate=1e-3): #1e-5였음
        """
        unfreeze_epoch: fine-tuning 단계에서 unfreeze를 시작할 epoch 번호 (예: 5)
        layers_to_unfreeze: unfreeze할 layer 클래스 목록 (예: [tf.keras.layers.Conv2D])
        new_learning_rate: unfreeze 후 적용할 새로운 학습률
        """
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch
        self.layers_to_unfreeze = layers_to_unfreeze
        self.new_learning_rate = new_learning_rate

    def on_epoch_begin(self, epoch, logs=None):
        if epoch + 1 == self.unfreeze_epoch:
            print(f"\nUnfreezing layers at epoch {epoch+1}...")
            for layer in self.model.layers:
                if any(isinstance(layer, lt) for lt in self.layers_to_unfreeze):
                    layer.trainable = True
                    print(f"Layer {layer.name} unfreezed.")
            # 학습률만 변경 (모델을 다시 compile 하지 않음)
            # tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.new_learning_rate)
            self.model.optimizer.learning_rate.assign(self.new_learning_rate)
            print(f"Learning rate set to {self.new_learning_rate}")

# -----------------------------------------------------------------------------
# LOSO 방식 (inter-subject) cross-validation 학습 함수
# -----------------------------------------------------------------------------
def inter_subject_cv_training(base_dirs, result_dir, epochs=150, batch_size=16, target_subjects=None):
    os.makedirs(result_dir, exist_ok=True)
    overall_folder = os.path.join(result_dir, "overall")
    os.makedirs(overall_folder, exist_ok=True)

    # 데이터 로드
    de_trials, psd_trials, label_list, subject_list = load_seediv_data(base_dirs)

    # subject별 그룹화
    subject_data = {}
    for de, psd, label, subj in zip(de_trials, psd_trials, label_list, subject_list):
        if subj not in subject_data:
            subject_data[subj] = {"de": [], "psd": [], "labels": []}
        subject_data[subj]["de"].append(de)
        subject_data[subj]["psd"].append(psd)
        subject_data[subj]["labels"].append(label)

    overall_acc = {}
    overall_reports = {}
    
    # 사용할 subjects 리스트 설정
    subjects = sorted(subject_data.keys(), key=lambda x: int(x) if x.isdigit() else x)

    if target_subjects is not None:
        subjects = [subj for subj in subjects if subj in target_subjects]
        
    for test_subj in subjects:
        print(f"\n========== LOSO: Test subject: {test_subj} ==========")
        # 테스트 셋
        X_de_test_trials = subject_data[test_subj]["de"]
        X_psd_test_trials = subject_data[test_subj]["psd"]
        y_test_trials = np.array(subject_data[test_subj]["labels"])
        y_cat_test_trials = to_categorical(y_test_trials, num_classes=4)

        # 나머지 subject들의 데이터를 training으로 사용
        X_de_train_trials = []
        X_psd_train_trials = []
        y_train_list = []
        for subj in subjects:
            if subj == test_subj:
                continue
            X_de_train_trials.extend(subject_data[subj]["de"])
            X_psd_train_trials.extend(subject_data[subj]["psd"])
            y_train_list.extend(subject_data[subj]["labels"])
        y_train_list = np.array(y_train_list)
        y_cat_train_trials = to_categorical(y_train_list, num_classes=4)

        # Train/Validation split (80/20)
        num_train_trials = len(X_de_train_trials)
        indices = np.arange(num_train_trials)
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y_train_list)
        X_de_train_split = [X_de_train_trials[i] for i in train_idx]
        X_de_val_split = [X_de_train_trials[i] for i in val_idx]
        X_psd_train_split = [X_psd_train_trials[i] for i in train_idx]
        X_psd_val_split = [X_psd_train_trials[i] for i in val_idx]
        y_train_split = y_cat_train_trials[train_idx]
        y_val_split = y_cat_train_trials[val_idx]

        # 시간 슬라이스 및 라벨 복제
        X_de_train = split_trials_list_to_time_slices(X_de_train_split)
        X_psd_train = split_trials_list_to_time_slices(X_psd_train_split)
        y_train = replicate_labels_for_trials(X_de_train_split, y_train_split)

        X_de_val = split_trials_list_to_time_slices(X_de_val_split)
        X_psd_val = split_trials_list_to_time_slices(X_psd_val_split)
        y_val = replicate_labels_for_trials(X_de_val_split, y_val_split)

        X_de_test = split_trials_list_to_time_slices(X_de_test_trials)
        X_psd_test = split_trials_list_to_time_slices(X_psd_test_trials)
        y_test = replicate_labels_for_trials(X_de_test_trials, y_cat_test_trials)

        # Dataset 생성
        train_dataset = tf.data.Dataset.from_tensor_slices(((X_de_train, X_psd_train), y_train)).shuffle(1000).batch(batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices(((X_de_val, X_psd_val), y_val)).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices(((X_de_test, X_psd_test), y_test)).batch(batch_size)

        # 모델 생성
        model = create_seediv_cnn_model(input_shape=(62, 1, 4), num_classes=4)
        model.summary()

        # ----- 1. Pre-training Phase ----- 
        pretrain_epochs = int(epochs * 0.6)
        finetune_epochs = epochs - pretrain_epochs
        pretrain_callbacks = [EarlyStopping(monitor='val_accuracy', patience=20, min_delta=0.001, restore_best_weights=True)]
        print(f"Test subject {test_subj}: Starting pre-training phase for {pretrain_epochs} epochs...")
        history_pretrain = model.fit(train_dataset, epochs=pretrain_epochs, validation_data=val_dataset, callbacks=pretrain_callbacks, verbose=1)

        # ----- 2. Fine-tuning Phase -----
        # 처음에는 분류기(Dense)만 학습하도록 feature extractor 동결
        for layer in model.layers:
            if isinstance(layer, (tf.keras.layers.Conv2D,
                                  tf.keras.layers.BatchNormalization,
                                  tf.keras.layers.MaxPooling2D,
                                  tf.keras.layers.GlobalAveragePooling2D)):
                layer.trainable = False
            else:
                layer.trainable = True

        # Fine-tuning 단계에서는 EarlyStopping 없이 모든 epoch 진행
        finetune_callbacks = []  # 또는 EarlyStopping(patience=100, restore_best_weights=False)
        # Layer-specific Learning Rate: 낮은 학습률 적용 (이미 학습된 feature extractor는 학습률 낮게)
        model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
        #원래 1e-4이었음
        print(f"Test subject {test_subj}: Starting fine-tuning phase for {finetune_epochs} epochs...")
        # Gradual Unfreeze Callback: 예를 들어, fine-tuning 시작 5번째 epoch에 Conv2D 레이어 일부를 unfreeze
        gradual_unfreeze_cb = GradualUnfreeze(unfreeze_epoch=5, layers_to_unfreeze=[tf.keras.layers.Conv2D])
        history_finetune = model.fit(train_dataset, epochs=finetune_epochs, validation_data=val_dataset,
                                     callbacks=finetune_callbacks + [gradual_unfreeze_cb], verbose=1)

        # Pre-training 단계 학습 곡선 저장
        subj_folder = os.path.join(result_dir, f"s{test_subj.zfill(2)}")
        os.makedirs(subj_folder, exist_ok=True)
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(history_pretrain.history['loss'], label='Pre-train Train Loss')
        plt.plot(history_pretrain.history['val_loss'], label='Pre-train Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Pre-training Loss Curve')
        plt.subplot(1,2,2)
        plt.plot(history_pretrain.history['accuracy'], label='Pre-train Train Acc')
        plt.plot(history_pretrain.history['val_accuracy'], label='Pre-train Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Pre-training Accuracy Curve')
        pretrain_curve_path = os.path.join(subj_folder, "pretrain_training_curves.png")
        plt.savefig(pretrain_curve_path)
        plt.close()
        print(f"Test subject {test_subj}: Pre-training curves saved to {pretrain_curve_path}")

        # Fine-tuning 단계 학습 곡선 저장
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(history_finetune.history['loss'], label='Fine-tune Train Loss')
        plt.plot(history_finetune.history['val_loss'], label='Fine-tune Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Fine-tuning Loss Curve')
        plt.subplot(1,2,2)
        plt.plot(history_finetune.history['accuracy'], label='Fine-tune Train Acc')
        plt.plot(history_finetune.history['val_accuracy'], label='Fine-tune Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Fine-tuning Accuracy Curve')
        finetune_curve_path = os.path.join(subj_folder, "finetune_training_curves.png")
        plt.savefig(finetune_curve_path)
        plt.close()
        print(f"Test subject {test_subj}: Fine-tuning curves saved to {finetune_curve_path}")

        # 평가 및 결과 저장
        test_loss, test_acc = model.evaluate(test_dataset, verbose=0)
        print(f"Test subject {test_subj}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        overall_acc[test_subj] = test_acc

        y_pred_prob = model.predict(test_dataset)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(np.concatenate([y for _, y in test_dataset], axis=0), axis=1)

        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, digits=2)
        overall_reports[test_subj] = report

        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        cm_path = os.path.join(subj_folder, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        print(f"Test subject {test_subj}: Confusion matrix saved to {cm_path}")

        report_path = os.path.join(subj_folder, "classification.txt")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Test subject {test_subj}: Classification report saved to {report_path}")

        model_save_path = os.path.join(subj_folder, "model_eeg_cnn.keras")
        model.save(model_save_path)
        print(f"Test subject {test_subj}: Model saved to {model_save_path}")

    overall_avg_acc = np.mean(list(overall_acc.values()))
    overall_report_path = os.path.join(overall_folder, "overall_classification.txt")
    with open(overall_report_path, "w") as f:
        f.write("Overall LOSO Test Accuracy: {:.4f}\n\n".format(overall_avg_acc))
        for subj in sorted(overall_reports.keys(), key=lambda x: int(x) if x.isdigit() else x):
            f.write(f"Test Subject {subj}:\n")
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
# TARGET_SUBJECTS = ["1", "3", "7"]  # 예: Subject 1, 3, 7만 대상으로 LOSO 수행
TARGET_SUBJECTS = [str(i) for i in range(1, 16)]  # Subject 1~15를 대상으로 LOSO 수행
RESULT_DIR = "/home/bcml1/sigenv/_5주차_eeg_SEEDIV/DE+PSD_up2_inter_lr_1e-3"
inter_subject_cv_training(base_dirs, RESULT_DIR, epochs=150, batch_size=16, target_subjects=TARGET_SUBJECTS)