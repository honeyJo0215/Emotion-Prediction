import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 결과 저장 경로 설정
RESULT_DIR = "/home/bcml1/sigenv/_4월/_diffusion/CSP_CNN_Transformer"
os.makedirs(RESULT_DIR, exist_ok=True)

# =============================================================================
# GPU 메모리 제한 (필요시 사용)
def limit_gpu_memory(memory_limit_mib=10000):
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

# =============================================================================
# Positional Embedding layer: 입력 sequence에 trainable한 위치 임베딩을 더합니다.
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.pos_emb = self.add_weight(shape=(1, self.sequence_length, self.embed_dim),
                                       name="pos_emb")
        super().build(input_shape)

    def call(self, x):
        return x + self.pos_emb


# =============================================================================
# Transformer Encoder Block
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    
    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# =============================================================================
# Frequency branch: 입력 shape (200,8,4)
# CNN으로 feature 추출 후, 2D feature map을 sequence로 reshape하여 Transformer block 적용
def build_freq_branch_with_transformer(input_shape=(200,8,4), num_heads=4, ff_dim=128, transformer_layers=1):
    inp = layers.Input(shape=input_shape)
    # CNN block
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(inp)
    x = layers.MaxPooling2D((2,2))(x)  # (200,8) -> (100,4)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    # x shape: (100,4,64)
    new_height = input_shape[0] // 2  # 100
    new_width = input_shape[1] // 2   # 4
    seq_len = new_height * new_width  # 400 tokens
    x = layers.Reshape((seq_len, 64))(x)
    # Positional embedding
    x = PositionalEmbedding(sequence_length=seq_len, embed_dim=64)(x)
    # Transformer blocks
    for _ in range(transformer_layers):
        x = TransformerBlock(embed_dim=64, num_heads=num_heads, ff_dim=ff_dim)(x)
    # Sequence pooling: GlobalAveragePooling1D
    x = layers.GlobalAveragePooling1D()(x)
    branch = models.Model(inputs=inp, outputs=x, name="FreqBranchTransformer")
    return branch

# =============================================================================
# Channel branch: 입력 shape (200,4,8)
# CNN으로 feature 추출 후, 2D feature map을 sequence로 reshape하여 Transformer block 적용
def build_chan_branch_with_transformer(input_shape=(200,4,8), num_heads=4, ff_dim=128, transformer_layers=1):
    inp = layers.Input(shape=input_shape)
    # CNN block
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(inp)
    x = layers.MaxPooling2D((2,2))(x)  # (200,4) -> (100,2)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    # x shape: (100,2,64)
    new_height = input_shape[0] // 2  # 100
    new_width = input_shape[1] // 2   # 2
    seq_len = new_height * new_width  # 200 tokens
    x = layers.Reshape((seq_len, 64))(x)
    # Positional embedding
    x = PositionalEmbedding(sequence_length=seq_len, embed_dim=64)(x)
    # Transformer blocks
    for _ in range(transformer_layers):
        x = TransformerBlock(embed_dim=64, num_heads=num_heads, ff_dim=ff_dim)(x)
    # Sequence pooling: GlobalAveragePooling1D
    x = layers.GlobalAveragePooling1D()(x)
    branch = models.Model(inputs=inp, outputs=x, name="ChanBranchTransformer")
    return branch

# =============================================================================
# 전체 모델 구성: 주파수와 채널 branch를 결합하는 모델 (CNN+Transformer)
def build_separated_model_with_transformer(num_classes=4, num_heads=4, ff_dim=128, transformer_layers=1):
    # 입력 shape: (4,200,8)
    input_csp = layers.Input(shape=(4,200,8), name="CSP_Input")
    
    # Frequency branch: Permute to (200,8,4)
    # (batch,4,200,8) -> (batch,200,8,4)
    freq_input = layers.Permute((2,3,1), name="Freq_Permute")(input_csp)
    freq_branch = build_freq_branch_with_transformer(input_shape=(200,8,4),
                                                     num_heads=num_heads,
                                                     ff_dim=ff_dim,
                                                     transformer_layers=transformer_layers)
    freq_feat = freq_branch(freq_input)
    
    # Channel branch: Permute to (200,4,8)
    # (batch,4,200,8) -> (batch,200,4,8)
    chan_input = layers.Permute((2,1,3), name="Chan_Permute")(input_csp)
    chan_branch = build_chan_branch_with_transformer(input_shape=(200,4,8),
                                                     num_heads=num_heads,
                                                     ff_dim=ff_dim,
                                                     transformer_layers=transformer_layers)
    chan_feat = chan_branch(chan_input)
    
    # 두 branch의 특징 결합
    combined = layers.Concatenate()([freq_feat, chan_feat])
    x = layers.Dense(128, activation='relu')(combined)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=input_csp, outputs=output, name="Separated_CNN_Transformer")
    return model

# =============================================================================
# CSP feature 로드 및 1초 단위 분할 함수
def load_csp_features(csp_feature_dir="/home/bcml1/2025_EMOTION/SEED_IV/eeg_EM-CSP"):
    """
    지정된 디렉토리 내의 모든 npy 파일을 로드합니다.
    파일명 형식: folder{folder_num}_subject{subject}_sample{sample}_label{label}.npy
    각 파일은 (4, T, 8) 형태이며, 1초(200 샘플) 단위로 분할하여 (4,200,8) 샘플 생성.
    반환: X, Y, subjects, file_ids
    """
    X_list, Y_list, subjects_list, files_list = [], [], [], []
    for file_name in os.listdir(csp_feature_dir):
        if not file_name.endswith(".npy"):
            continue
        file_path = os.path.join(csp_feature_dir, file_name)
        try:
            data = np.load(file_path)  # 예상 shape: (4, T, 8)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
        if data.ndim != 3 or data.shape[0] != 4 or data.shape[2] != 8:
            print(f"Unexpected shape in {file_path}: {data.shape}")
            continue
        T = data.shape[1]
        n_windows = T // 200  # 1초 = 200 샘플
        if n_windows < 1:
            continue
        for i in range(n_windows):
            window = data[:, i*200:(i+1)*200, :]  # (4,200,8)
            X_list.append(window)
            # subject 추출: 파일명 내 "subject" 뒤의 숫자 (예: folder3_subject11_sample23_label1.npy)
            subject_match = re.search(r'subject(\d+)', file_name, re.IGNORECASE)
            subject = subject_match.group(1) if subject_match else 'unknown'
            # label 추출: "label" 뒤의 숫자
            label_match = re.search(r'label(\d+)', file_name, re.IGNORECASE)
            label = int(label_match.group(1)) if label_match else -1
            Y_list.append(label)
            files_list.append(f"{file_name}_win{i}")
            subjects_list.append(subject)
    X = np.array(X_list)
    Y = np.array(Y_list)
    subjects = np.array(subjects_list)
    file_ids = np.array(files_list)
    
    valid_mask = Y >= 0
    X = X[valid_mask]
    Y = Y[valid_mask]
    subjects = subjects[valid_mask]
    file_ids = file_ids[valid_mask]
    
    print(f"Loaded {X.shape[0]} samples, each sample shape: {X.shape[1:]}")
    print(f"Unique labels found: {np.unique(Y)}")
    print(f"Unique subjects found: {np.unique(subjects)}")
    return X, Y, subjects, file_ids

# =============================================================================
# Intra-subject 학습 및 평가
if __name__ == "__main__":
    # CSP feature 데이터 로드: 각 샘플 shape = (4,200,8)
    CSP_FEATURE_DIR = "/home/bcml1/2025_EMOTION/SEED_IV/eeg_EM-CSP"
    X, Y, subjects, file_ids = load_csp_features(CSP_FEATURE_DIR)
    
    # Intra-subject 방식: subject별로 분할 후 파일 단위로 train/validation/test split
    unique_subjects = np.unique(subjects)
    for subj in unique_subjects:
        if not (4 <= int(subj) < 10):
            continue
        print(f"\n========== Intra-subject: Subject = {subj} ==========")
        subj_mask = (subjects == subj)
        X_subj = X[subj_mask]
        Y_subj = Y[subj_mask]
        F_subj = file_ids[subj_mask]
        
        unique_files = np.unique(F_subj)
        if len(unique_files) < 2:
            print(f"Subject {subj}의 파일 수가 충분하지 않아 스킵합니다.")
            continue
        
        train_files, test_files = train_test_split(unique_files, test_size=0.3, random_state=42)
        train_files_final, val_files = train_test_split(train_files, test_size=0.25, random_state=42)
        
        train_mask = np.isin(F_subj, train_files_final)
        val_mask = np.isin(F_subj, val_files)
        test_mask = np.isin(F_subj, test_files)
        
        X_train = X_subj[train_mask]
        Y_train = Y_subj[train_mask]
        X_val = X_subj[val_mask]
        Y_val = Y_subj[val_mask]
        X_test = X_subj[test_mask]
        Y_test = Y_subj[test_mask]
        
        print(f"Subject {subj}: Train {X_train.shape[0]} samples, Val {X_val.shape[0]} samples, Test {X_test.shape[0]} samples.")
        
        subj_folder = os.path.join(RESULT_DIR, f"s{subj}")
        os.makedirs(subj_folder, exist_ok=True)
        
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(1000).batch(64)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(64)
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)
        
        # 모델 생성: 입력 shape = (4,200,8)
        model = build_separated_model_with_transformer(num_classes=4,
                                                       num_heads=4,
                                                       ff_dim=128,
                                                       transformer_layers=1)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=1e-4, decay_steps=100000, decay_rate=0.9, staircase=True),
            clipnorm=1.0
        )
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=2000, mode='max', restore_best_weights=True)
        model.compile(optimizer=optimizer,
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        model.summary()
        
        history = model.fit(train_dataset, epochs=2000, validation_data=val_dataset, callbacks=[early_stopping])
        
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curve')
        
        plt.subplot(1,2,2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy Curve')
        train_curve_path = os.path.join(subj_folder, "training_curves.png")
        plt.savefig(train_curve_path)
        plt.close()
        print(f"Subject {subj}: Training curves saved to {train_curve_path}")
        
        test_loss, test_acc = model.evaluate(test_dataset)
        print(f"Subject {subj}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        
        y_pred_prob = model.predict(test_dataset)
        y_pred = np.argmax(y_pred_prob, axis=1)
        cm = confusion_matrix(Y_test, y_pred)
        report = classification_report(Y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(report)
        
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        cm_path = os.path.join(subj_folder, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()
        print(f"Subject {subj}: Confusion matrix saved to {cm_path}")
        
        report_path = os.path.join(subj_folder, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Subject {subj}: Classification report saved to {report_path}")
        
        model_save_path = os.path.join(subj_folder, "model_eeg_cnn_transformer.keras")
        model.save(model_save_path)
        print(f"Subject {subj}: Model saved to {model_save_path}")
