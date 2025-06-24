import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.layers import Flatten, Dense, Reshape, Average
from keras.models import Sequential, Model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

# -----------------------------------------------------------------------------
# 모델 구성
# -----------------------------------------------------------------------------
def create_base_network(input_dim, dropout_rate):
    seq = Sequential()
    seq.add(Conv2D(64, 5, activation='relu', padding='same', name='conv1', input_shape=input_dim))
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
    seq.add(Flatten(name='fla1'))
    seq.add(Dense(512, activation='relu', name='dense1'))
    seq.add(Reshape((1, 512), name='reshape'))
    seq.add(BatchNormalization())
    seq.add(Dropout(dropout_rate))
    return seq

def create_MT_CNN(img_size=(8, 9, 8), dropout_rate=0.2, number_of_inputs=1):
    """
    수정된 모델: 한 개의 출력 레이어로 4 클래스 분류 (0:Excited, 1:Relaxed, 2:Stressed, 3:Bored)
    """
    base_network = create_base_network(img_size, dropout_rate)
    inputs = [Input(shape=img_size) for _ in range(number_of_inputs)]
    if number_of_inputs == 1:
        x = base_network(inputs[0])
    else:
        x = Average()([base_network(inp) for inp in inputs])
    x = Flatten(name='flat')(x)
    out = Dense(4, activation='softmax', name='out')(x)
    model = Model(inputs, out)
    return model

# -----------------------------------------------------------------------------
# 평균 classification report 생성 함수
# -----------------------------------------------------------------------------
def create_average_report(fold_metrics):
    """
    fold_metrics: 각 fold별 metric 딕셔너리 리스트  
      각 딕셔너리는 {'precision': array, 'recall': array, 'f1': array, 'support': array, 'accuracy': float}  
    각 클래스(0,1,2,3)에 대해 fold별로 계산된 precision, recall, f1 점수의 평균을 계산하고,  
    macro, weighted 평균도 함께 포함하는 문자열 보고서를 반환함.
    """
    # fold별 배열을 np.array로 변환 후 평균 계산
    avg_precision = np.mean([m['precision'] for m in fold_metrics], axis=0)
    avg_recall = np.mean([m['recall'] for m in fold_metrics], axis=0)
    avg_f1 = np.mean([m['f1'] for m in fold_metrics], axis=0)
    # support는 fold별 합산 (전체 테스트 샘플 수)
    total_support = np.sum([m['support'] for m in fold_metrics], axis=0)
    avg_accuracy = np.mean([m['accuracy'] for m in fold_metrics])
    
    # macro average
    macro_precision = np.mean(avg_precision)
    macro_recall = np.mean(avg_recall)
    macro_f1 = np.mean(avg_f1)
    # weighted average
    weighted_precision = np.average(avg_precision, weights=total_support)
    weighted_recall = np.average(avg_recall, weights=total_support)
    weighted_f1 = np.average(avg_f1, weights=total_support)
    
    # 클래스 이름 (0: Excited, 1: Relaxed, 2: Stressed, 3: Bored)
    target_names = ['Excited', 'Relaxed', 'Stressed', 'Bored']
    
    report = "Average Classification Report over 5 folds:\n\n"
    report += f"{'Class':<10}{'Precision':>10}{'Recall':>10}{'F1-score':>10}{'Support':>10}\n"
    for i in range(4):
        report += f"{target_names[i]:<10}{avg_precision[i]:10.4f}{avg_recall[i]:10.4f}{avg_f1[i]:10.4f}{int(total_support[i]):10d}\n"
    report += "\n"
    report += f"{'Accuracy':<10}{avg_accuracy:10.4f}\n"
    report += f"{'Macro avg':<10}{macro_precision:10.4f}{macro_recall:10.4f}{macro_f1:10.4f}{np.sum(total_support):10d}\n"
    report += f"{'Weighted avg':<10}{weighted_precision:10.4f}{weighted_recall:10.4f}{weighted_f1:10.4f}{np.sum(total_support):10d}\n"
    
    return report

# -----------------------------------------------------------------------------
# 데이터 로딩 함수 (subject별 EEG 파일과 emotion label 파일 로드)
# -----------------------------------------------------------------------------
def load_data_for_subject(subject_id, data_dir, label_dir):
    """
    EEG 파일명: sXX_sample_YY_segment_ZZZ_label_..._2D_DE.npy
    각 파일에서 sample_YY 부분의 숫자(0~39)를 추출하여,
    subject별 emotion label 파일 (sXX_emotion_labels.npy; 40개의 정수 0,1,2,3)에서 해당 인덱스의 라벨을 가져옴.
    """
    pattern = os.path.join(data_dir, f"{subject_id}_sample_*_segment_*_2D_DE.npy")
    file_list = glob.glob(pattern)
    if len(file_list) == 0:
        raise ValueError(f"{subject_id}에 해당하는 EEG 데이터 파일이 없습니다.")
    
    # subject별 라벨 파일 로드
    label_file = os.path.join(label_dir, f"{subject_id}_emotion_labels.npy")
    try:
        label_data = np.load(label_file, allow_pickle=True)
    except Exception as e:
        raise ValueError(f"{subject_id} 라벨 파일 로드 실패: {e}")
    
    X_list = []
    y_list = []
    
    # 파일명에서 subject, sample, segment 정보를 추출 (정규표현식 이용)
    regex = re.compile(r"(s\d{2})_sample_(\d{2})_segment_(\d{3})_label_.*_2D_DE\.npy")
    
    for file in file_list:
        basename = os.path.basename(file)
        m = regex.match(basename)
        if not m:
            print(f"패턴에 맞지 않는 파일명: {basename}")
            continue
        
        subj, sample_str, segment = m.group(1), m.group(2), m.group(3)
        sample_idx = int(sample_str)  # sample index (0~39)
        if sample_idx >= label_data.shape[0]:
            print(f"sample index {sample_idx}가 라벨 파일 범위를 초과함: {basename}")
            continue
        
        try:
            data = np.load(file)
        except Exception as e:
            print(f"파일 로드 에러 {file}: {e}")
            continue
        X_list.append(data)
        
        # 라벨 파일에서 해당 sample index의 라벨 (0,1,2,3)을 가져옴
        label_value = label_data[sample_idx]
        # label_value가 배열인 경우 첫번째 값을 사용
        if hasattr(label_value, 'shape') and label_value.shape != ():
            label_value = int(label_value[0])
        else:
            label_value = int(label_value)
        y_list.append(label_value)
    
    if len(X_list) == 0:
        raise ValueError(f"{subject_id}에 유효한 EEG 데이터가 없습니다.")
    
    X = np.array(X_list)
    # 4개의 클래스로 one-hot encoding
    y = to_categorical(np.array(y_list), num_classes=4)
    
    return X, y

# -----------------------------------------------------------------------------
# 메인 학습 및 평가 루프 (subject별 intra-subject 학습)
# -----------------------------------------------------------------------------
def main():
    data_dir = "/home/bcml1/2025_EMOTION/DEAP_EEG_2D_DE"
    label_dir = "/home/bcml1/2025_EMOTION/DEAP_four_labels"
    results_dir = "/home/bcml1/sigenv/_4주차_eeg/eeg_1"
    # 결과 저장할 최상위 폴더 (필요시 생성)
    results_dir = os.path.join(results_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    subjects = [f"s{str(i).zfill(2)}" for i in range(1, 23)]  # s01 ~ s22
    
    for subject in subjects:
        print(f"====== {subject} 학습 시작 ======")
        try:
            X, y = load_data_for_subject(subject, data_dir, label_dir)
        except ValueError as e:
            print(e)
            continue
        
        subject_results_dir = os.path.join(results_dir, subject)
        os.makedirs(subject_results_dir, exist_ok=True)
        
        # 각 fold별 metric 저장 리스트 (평균 report 생성에 사용)
        subject_fold_metrics = []
        
        # 5-fold cross validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_no = 1
        
        for train_val_idx, test_idx in kf.split(X):
            print(f"Subject {subject} - Fold {fold_no} 시작")
            
            X_train_val, X_test = X[train_val_idx], X[test_idx]
            y_train_val, y_test = y[train_val_idx], y[test_idx]
            
            # train/validation 분할 (80:20)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=0.2, random_state=42
            )
            
            # 입력 shape은 EEG 데이터의 shape에 맞게 (예: (8, 9, 8))
            input_shape = X.shape[1:]
            model = create_MT_CNN(img_size=input_shape, dropout_rate=0.2, number_of_inputs=1)
            model.compile(optimizer='adam', 
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            
            fold_results_dir = os.path.join(subject_results_dir, f"fold_{fold_no}")
            os.makedirs(fold_results_dir, exist_ok=True)
            checkpoint_path = os.path.join(fold_results_dir, "best_model.keras")
            checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
            earlystop = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=150,
                batch_size=32,
                callbacks=[checkpoint, earlystop],
                verbose=1
            )
            
            # 학습 과정 (loss, accuracy) plot 저장
            plt.figure(figsize=(12,5))
            plt.subplot(1,2,1)
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Val Loss')
            plt.title(f"{subject} - Fold {fold_no} Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            
            plt.subplot(1,2,2)
            plt.plot(history.history['accuracy'], label='Train Acc')
            plt.plot(history.history['val_accuracy'], label='Val Acc')
            plt.title(f"{subject} - Fold {fold_no} Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.tight_layout()
            loss_acc_plot_path = os.path.join(fold_results_dir, "training_history.png")
            plt.savefig(loss_acc_plot_path)
            plt.close()
            
            # 테스트 평가
            model.load_weights(checkpoint_path)
            test_eval = model.evaluate(X_test, y_test, verbose=0)
            y_pred = model.predict(X_test)
            y_test_labels = np.argmax(y_test, axis=1)
            y_pred_labels = np.argmax(y_pred, axis=1)
            
            report = classification_report(y_test_labels, y_pred_labels, digits=4)
            cm = confusion_matrix(y_test_labels, y_pred_labels)
            
            report_path = os.path.join(fold_results_dir, "classification_report.txt")
            with open(report_path, "w") as f:
                f.write(report)
            
            cm_path = os.path.join(fold_results_dir, "confusion_matrix.txt")
            with open(cm_path, "w") as f:
                f.write(np.array2string(cm))
            
            # --- 추가 부분: fold별 metric 계산 후 저장 ---
            # 각 fold마다 precision, recall, f1, support 계산 (클래스 0,1,2,3)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_test_labels, y_pred_labels, labels=[0, 1, 2, 3], zero_division=0)
            accuracy = np.sum(y_test_labels == y_pred_labels) / len(y_test_labels)
            subject_fold_metrics.append({
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support,
                'accuracy': accuracy
            })

            print(f"Subject {subject} - Fold {fold_no} 완료")
            fold_no += 1
        
        # --- 추가 부분: subject별 5-fold 평균 classification report 생성 및 저장 ---
        avg_report = create_average_report(subject_fold_metrics)
        avg_report_path = os.path.join(subject_results_dir, "average_classification_report.txt")
        with open(avg_report_path, "w") as f:
            f.write(avg_report)

        print(f"====== {subject} 학습 완료 ======")

if __name__ == "__main__":
    main()
