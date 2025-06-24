import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, TimeDistributed, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# GPU 메모리 제한
def limit_gpu_memory(memory_limit_mib=14000):
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
limit_gpu_memory(2000)

# ===== 파일 단위 데이터 로딩 =====
# 데이터 정규화 예시: 각 윈도우마다 표준화 (평균 0, 표준편차 1)
def normalize_windows(windows):
    # windows shape: (n_windows, window_size, num_features)
    mean = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True) + 1e-8  # division by zero 방지
    return (windows - mean) / std

# load_file_data() 함수 내에서
def load_file_data(file, window_size=10, stride=1):
    try:
        data = np.load(file)  # shape: (T,4)
    except Exception as e:
        print(f"Error loading file {file}: {e}")
        return None, None, None

    T = data.shape[0]
    if T < window_size:
        return None, None, None
    windows = []
    for start in range(0, T - window_size + 1, stride):
        window = data[start:start+window_size, :]  # shape: (10,4)
        windows.append(window)
    windows = np.array(windows)
    # 입력 윈도우 정규화 적용
    windows = normalize_windows(windows)
    
    base = os.path.basename(file)
    parts = base.split('_')
    try:
        subject_id = parts[0]  # 피험자 번호
        # 레이블이 0,1,2,3인 경우, 별도의 보정 없이 그대로 사용
        label = int(parts[-1].split('.')[0])
    except Exception as e:
        print(f"Error parsing file {file}: {e}")
        return None, None, None
    labels = np.full((windows.shape[0],), label)
    return subject_id, windows, labels

def load_all_subjects(data_base, window_size=10, stride=1):
    """
    data_base 폴더 내의 모든 실험 폴더를 순회하여, 각 파일에서 피험자(subject) 정보를 추출.
    동일 피험자에 해당하는 데이터들을 그룹화하여 dict {subject_id: (X, y)}로 반환.
    """
    data_dict = {}
    # data_base 내의 모든 실험 폴더를 검색
    exp_folders = [os.path.join(data_base, d) for d in os.listdir(data_base) if os.path.isdir(os.path.join(data_base, d))]
    for exp_folder in exp_folders:
        files = sorted(glob.glob(os.path.join(exp_folder, '*.npy')))
        for file in files:
            subject_id, X, y = load_file_data(file, window_size, stride)
            if X is None:
                continue
            if subject_id not in data_dict:
                data_dict[subject_id] = ([], [])
            data_dict[subject_id][0].append(X)
            data_dict[subject_id][1].append(y)
    # 피험자별로 모든 윈도우와 라벨을 합침
    for subject_id in data_dict:
        X_list, y_list = data_dict[subject_id]
        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)
        data_dict[subject_id] = (X_all, y_all)
        print(f"Subject {subject_id}: loaded {X_all.shape[0]} windows.")
    return data_dict

def get_LOSO_split(data_dict, test_subject):
    """
    LOSO 방식: test_subject에 해당하는 데이터는 test set, 나머지는 training set으로 구성.
    """
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    for subj, (X, y) in data_dict.items():
        if subj == test_subject:
            X_test_list.append(X)
            y_test_list.append(y)
        else:
            X_train_list.append(X)
            y_train_list.append(y)
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)
    return X_train, y_train, X_test, y_test

# ===== 모델 구성: CNN + Transformer =====
def build_model(input_shape=(10,4), num_classes=2, transformer_heads=2, transformer_ff_dim=32, dropout_rate=0.1):
    """
    - 입력: (window_size=10, feature_dim=4)
    - 입력 정규화를 위해 TimeDistributed BatchNormalization 추가
    - TimeDistributed Dense layer를 통해 간단한 CNN 역할 수행
    - Transformer block: MultiHeadAttention + FeedForward network + LayerNormalization (epsilon=1e-3)
    - GlobalAveragePooling1D 후 최종 Dense (softmax)로 분류
    """
    inputs = Input(shape=input_shape)  # (10,4)
    # 입력 정규화
    x = TimeDistributed(BatchNormalization())(inputs)
    # TimeDistributed Dense layer (각 초별 feature embedding)
    x = TimeDistributed(Dense(16, activation='relu'))(x)  # (10,16)
    
    # Transformer: MultiHeadAttention + Dropout
    attn_output = MultiHeadAttention(num_heads=transformer_heads, key_dim=16)(x, x)
    attn_output = Dropout(dropout_rate)(attn_output)
    x = LayerNormalization(epsilon=1e-3)(x + attn_output)
    
    # Transformer: FeedForward Network + Dropout
    ffn = TimeDistributed(Dense(transformer_ff_dim, activation='relu'))(x)
    ffn = TimeDistributed(Dense(16))(ffn)
    ffn = Dropout(dropout_rate)(ffn)
    x = LayerNormalization(epsilon=1e-3)(x + ffn)
    
    # Global pooling over time
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # 낮은 학습률과 gradient clipping 적용
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ===== 학습 및 평가 함수 =====
def plot_training_history(history, save_path, title="Loss Curve"):
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved training curve: {save_path}")

def plot_confusion_matrix(cm, num_classes, save_path, title="Confusion Matrix"):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved confusion matrix: {save_path}")

def plot_roc_curve(fpr, tpr, roc_auc, save_path, title="ROC Curve"):
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved ROC curve: {save_path}")

# ===== LOSO 학습 및 Fine tuning =====
def run_loso_experiment(data_base, num_classes=2, window_size=10, stride=1):
    """
    data_base: 실험 폴더들이 존재하는 상위 디렉토리.
             각 실험 폴더 내에 피험자 데이터(np.npy 파일)가 있으며,
             파일명에서 맨 앞 숫자가 피험자(subject) 번호임.
    num_classes: 감정 분류 클래스 수 (예, 2 또는 다중 클래스)
    LOSO 방식으로 pretrain 후, test subject에 대해 fine tuning 및 평가 진행
    """
    data_dict = load_all_subjects(data_base, window_size, stride)
    # 피험자(subject) id들을 정렬 (숫자 순으로)
    subject_ids = sorted(data_dict.keys(), key=lambda x: int(x))
    
    # 각 subject를 test subject로 두고 반복 (LOSO)
    for test_subj in subject_ids:
        print(f"\n===== LOSO: Test Subject {test_subj} =====")
        X_train, y_train, X_test_full, y_test_full = get_LOSO_split(data_dict, test_subject=test_subj)
        print(f"Train set: {X_train.shape[0]} samples, Test set (all from subject {test_subj}): {X_test_full.shape[0]} samples")
        
        # --- Pretrain phase (training on other subjects) ---
        model = build_model(input_shape=(window_size, 4), num_classes=num_classes)
        pretrain_ckpt = f"pretrain_subject_{test_subj}.keras"
        checkpoint = ModelCheckpoint(pretrain_ckpt, monitor='val_accuracy', save_best_only=True, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        
        history = model.fit(X_train, y_train, validation_split=0.1,
                            epochs=50, batch_size=64, callbacks=[checkpoint, early_stop])
        
        # 저장: pretrain training curve
        plot_training_history(history, f"pretrain_loss_curve_subject_{test_subj}.png",
                              title=f"Pretrain Loss Curve (Test Subject {test_subj})")
        
        # 로드: best pretrain model
        model = tf.keras.models.load_model(pretrain_ckpt)
        
        # --- Fine tuning phase on test subject ---
        # test subject 내 데이터를 fine tuning용으로 train / test split (예: 80/20)
        X_ft_train, X_ft_test, y_ft_train, y_ft_test = train_test_split(
            X_test_full, y_test_full, test_size=0.2, random_state=42, stratify=y_test_full)
        print(f"Fine tuning: Train {X_ft_train.shape[0]} samples, Test {X_ft_test.shape[0]} samples")
        
        # 낮은 학습률과 gradient clipping을 적용하여 fine tuning (learning_rate=1e-5, clipnorm=1.0)
        ft_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0)
        model.compile(optimizer=ft_optimizer,
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        ft_history = model.fit(X_ft_train, y_ft_train, validation_split=0.1,
                               epochs=20, batch_size=64, callbacks=[early_stop])
        
        # 저장: fine tuning training curve
        plot_training_history(ft_history, f"finetune_loss_curve_subject_{test_subj}.png",
                              title=f"Fine Tuning Loss Curve (Test Subject {test_subj})")
        
        # --- Evaluation on fine tuning test set ---
        y_pred_probs = model.predict(X_ft_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # ROC curve (이진 분류일 경우, 다중 클래스인 경우 별도 구현 필요)
        if num_classes == 2:
            fpr, tpr, _ = roc_curve(y_ft_test, y_pred_probs[:, 1])
            roc_auc = auc(fpr, tpr)
            plot_roc_curve(fpr, tpr, roc_auc, f"roc_curve_subject_{test_subj}.png",
                           title=f"ROC Curve (Test Subject {test_subj})")
        else:
            print("다중 클래스 ROC curve 계산은 별도로 구현해야 합니다.")
        
        # Confusion Matrix & Classification Report
        cm = confusion_matrix(y_ft_test, y_pred)
        plot_confusion_matrix(cm, num_classes, f"confusion_matrix_subject_{test_subj}.png",
                              title=f"Confusion Matrix (Test Subject {test_subj})")
        report = classification_report(y_ft_test, y_pred)
        with open(f"classification_report_subject_{test_subj}.txt", "w") as f:
            f.write(report)
        print(f"Classification Report for Test Subject {test_subj}:\n", report)

# ===== 메인 실행 =====
if __name__ == '__main__':
    # data_base 폴더 경로: 실험 폴더들이 모여있는 상위 디렉토리
    # 예: /home/bcml1/2025_EMOTION/SEED_IV/eeg_CSP
    data_base = '/home/bcml1/2025_EMOTION/SEED_IV/eeg_CSP'
    # 감정 클래스 수 (예: 2, 3, 4 등 상황에 맞게 설정)
    num_classes = 4  # 필요에 따라 변경
    # window size: 10초, stride: 1초 (overlap)
    window_size = 10
    stride = 1

    run_loso_experiment(data_base, num_classes, window_size, stride)
