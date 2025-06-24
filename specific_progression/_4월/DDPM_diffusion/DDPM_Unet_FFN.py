import os
import re
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 결과 저장 경로 설정
RESULT_DIR = "/home/bcml1/sigenv/_4월/DDPM_diffusion/DDPM_Emotion_FullUNet"
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
# (4,200,8) CSP Feature 로드, 파일명에서 label(0~3) 추출
def load_csp_features(csp_feature_dir="/home/bcml1/2025_EMOTION/DEAP_eeg_new_label_CSP"):
    X_list, Y_list, subjects_list = [], [], []
    for file_name in os.listdir(csp_feature_dir):
        if not file_name.endswith(".npy"):
            continue
        file_path = os.path.join(csp_feature_dir, file_name)
        try:
            data = np.load(file_path)  # shape: (4, T, 8)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
        if data.shape[0] != 4 or data.shape[2] != 8:
            continue
        
        # label, subject 파싱
        label_match = re.search(r'label(\d+)', file_name, re.IGNORECASE)
        if not label_match:
            continue
        label = int(label_match.group(1))
        if label not in [0,1,2,3]:
            continue
        subj_match = re.search(r'subject(\d+)', file_name, re.IGNORECASE)
        subject = subj_match.group(1) if subj_match else 'unknown'
        
        T = data.shape[1]
        n_windows = T // 200
        for i in range(n_windows):
            window = data[:, i*200:(i+1)*200, :]  # (4,200,8)
            X_list.append(window)
            Y_list.append(label)
            subjects_list.append(subject)
    X = np.array(X_list)
    Y = np.array(Y_list)
    subjects = np.array(subjects_list)
    print(f"Total loaded samples: {X.shape[0]}, shape: {X.shape[1:]}")
    return X, Y, subjects

# =============================================================================
# UnetDown Block
def unet_down(filters, kernel_size=3, apply_batchnorm=False):
    block = tf.keras.Sequential()
    block.add(layers.Conv2D(filters, kernel_size, strides=1, padding='same', activation='relu'))
    block.add(layers.MaxPooling2D((2,2)))
    if apply_batchnorm:
        block.add(layers.BatchNormalization())
    return block

# =============================================================================
# UnetUp Block
def unet_up(filters, kernel_size=3, apply_batchnorm=False):
    block = tf.keras.Sequential()
    block.add(layers.UpSampling2D((2,2)))
    block.add(layers.Conv2D(filters, kernel_size, strides=1, padding='same', activation='relu'))
    if apply_batchnorm:
        block.add(layers.BatchNormalization())
    return block

# =============================================================================
# Encoder: 3 * UnetDown + AveragePooling (논문 그림 참고)
def build_encoder():
    inp = layers.Input(shape=(200,8,4))   # (H=200, W=8, C=4)
    # Down1
    d1 = unet_down(32)(inp)  # (100,4,32)
    # Down2
    d2 = unet_down(64)(d1)   # (50,2,64)
    # Down3
    d3 = unet_down(128)(d2)  # (25,1,128)
    # AveragePooling
    x = layers.GlobalAveragePooling2D()(d3)  # (batch,128)
    model = models.Model(inputs=inp, outputs=[d1,d2,d3,x], name="Encoder")
    return model

# =============================================================================
# Decoder: 3 * UnetUp (Skip connection from encoder)
def build_decoder():
    # d1=(100,4,32), d2=(50,2,64), d3=(25,1,128), x=(128)
    d1_in = layers.Input(shape=(100,4,32))
    d2_in = layers.Input(shape=(50,2,64))
    d3_in = layers.Input(shape=(25,1,128))
    # latent_from_ddpm: (25,1,128) 복원된 최종 latent
    latent_in = layers.Input(shape=(25,1,128))

    # Up1
    up1 = unet_up(128)(latent_in)  # (50,2,128)
    d3_up = layers.UpSampling2D(size=(2,2))(d3_in)  # d3_in: (None,25,1,128) → (None,50,2,128)
    cat1 = layers.Concatenate()([up1, d3_up])
 # skip connection
    # Up2
    up2 = unet_up(64)(cat1)       # (100,4,64)
    d2_up = layers.UpSampling2D(size=(2,2))(d2_in)  # d3_in: (None,25,1,128) → (None,50,2,128)
    cat2 = layers.Concatenate()([up2, d2_up])
    #cat2 = layers.Concatenate()([up2, d2_in])
    # Up3
    up3 = unet_up(32)(cat2)       # (200,8,32)
    d1_up = layers.UpSampling2D(size=(2,2))(d1_in)  # d3_in: (None,25,1,128) → (None,50,2,128)
    cat3 = layers.Concatenate()([up3, d1_up])
    
    # 최종 출력: (batch,200,8,4)
    out = layers.Conv2D(4, (3,3), padding='same', activation=None)(cat3)
    
    model = models.Model(inputs=[d1_in,d2_in,d3_in,latent_in], outputs=out, name="Decoder")
    return model

# =============================================================================
# DDPM module: noisy latent -> predicted noise (UNet-like)
def build_ddpm_block():
    inp = layers.Input(shape=(25,1,128))  # shape of d3 in the figure
    x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(inp)
    x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
    out = layers.Conv2D(128, (3,3), padding='same', activation=None)(x)
    model = models.Model(inputs=inp, outputs=out, name="DDPM_Block")
    return model

# =============================================================================
# 전체 모델: encoder–DDPM–decoder–FFN
class DiffusionClassificationModel(tf.keras.Model):
    def __init__(self, encoder, ddpm_block, decoder, num_classes=4, alpha=0.9, lambda_diff=1.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.ddpm_block = ddpm_block
        self.decoder = decoder
        self.num_classes = num_classes
        
        self.alpha = alpha
        self.lambda_diff = lambda_diff
        self.sqrt_alpha = math.sqrt(alpha)
        self.sqrt_one_minus_alpha = math.sqrt(1 - alpha)
        
        # FFN (논문 그림의 최종 분류기 부분)
        self.ffn = tf.keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ], name="FFN_classifier")
    
    def call(self, x, training=False):
        """
        x shape: (batch,4,200,8).
        encoder outputs: d1,d2,d3 (skip), x_enc (128-dim)
        """
        # 1) Encoder
        d1,d2,d3,x_enc = self.encoder(tf.transpose(x, [0,2,3,1]))  # (batch,200,8,4)->(batch, H=200,W=8,C=4)
        
        # 2) Diffusion (latent space) : x_t = sqrt(a)*latent_x0 + sqrt(1-a)*noise
        epsilon = tf.random.normal(tf.shape(d3))  # d3 shape: (batch,25,1,128)
        latent_t = self.sqrt_alpha * d3 + self.sqrt_one_minus_alpha * epsilon
        
        # 3) DDPM block: 예측 노이즈
        epsilon_hat = self.ddpm_block(latent_t)   # (batch,25,1,128)
        
        # 4) target noise: (latent_t - sqrt(a)*d3)/sqrt(1-a)
        #   => 여기서 d3는 사실상 원본 latent_x0
        epsilon_target = (latent_t - self.sqrt_alpha*d3) / self.sqrt_one_minus_alpha
        
        # 5) 복원된 latent: latent0_hat
        latent0_hat = (latent_t - self.sqrt_one_minus_alpha*epsilon_hat)/self.sqrt_alpha
        
        # 6) Decoder (skip connection: d1,d2,d3 + latent0_hat)
        x_dec = self.decoder([d1,d2,d3, latent0_hat])  # shape: (batch,200,8,4)
        # (batch,4,200,8)로 원복
        x_dec_perm = tf.transpose(x_dec, [0,3,1,2])
        
        # 7) 분류 (FFN): decoder 출력의 마지막 layer (d3 pooling)을 사용할 수도 있으나
        #    여기서는 간단히 x_enc(128-dim) 사용 + decoder의 global feature를 결합
        #    - 논문 그림에서는 decoder 출력을 1D pooling 후 FFN에 넣지만, 여기서는 x_enc만 사용 예시
        class_logits = self.ffn(x_enc, training=training)  # (batch,num_classes)
        
        return class_logits, x_dec_perm, epsilon_hat, epsilon_target
    
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            class_logits, x_dec, epsilon_hat, epsilon_target = self(x, training=True)
            # 분류 손실
            loss_cls = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, class_logits))
            # 복원 손실
            loss_recon = tf.reduce_mean(tf.keras.losses.mse(x, x_dec))
            # DDPM 손실
            loss_diff = tf.reduce_mean(tf.keras.losses.mse(epsilon_target, epsilon_hat))
            loss = loss_cls + loss_recon + self.lambda_diff*loss_diff
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        acc = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(y, class_logits))
        return {"loss": loss, "loss_cls": loss_cls, "loss_recon": loss_recon, "loss_diff": loss_diff, "accuracy": acc}
    
    def test_step(self, data):
        x, y = data
        class_logits, x_dec, epsilon_hat, epsilon_target = self(x, training=False)
        loss_cls = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, class_logits))
        loss_recon = tf.reduce_mean(tf.keras.losses.mse(x, x_dec))
        loss_diff = tf.reduce_mean(tf.keras.losses.mse(epsilon_target, epsilon_hat))
        loss = loss_cls + loss_recon + self.lambda_diff*loss_diff
        acc = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(y, class_logits))
        return {"loss": loss, "loss_cls": loss_cls, "loss_recon": loss_recon, "loss_diff": loss_diff, "accuracy": acc}

# =============================================================================
# 실행 스크립트
if __name__=="__main__":
    X, Y, subjects = load_csp_features()
    unique_subjects = np.unique(subjects)
    
    for subj in unique_subjects:
        # 예시: subject 12~32만 학습
        if not (12 <= int(subj) < 33):
            continue
        print(f"\n===== Subject {subj} =====")
        mask = (subjects == subj)
        X_subj = X[mask]
        Y_subj = Y[mask]
        # 라벨이 일부일 경우 재매핑
        unique_labels_subj = np.unique(Y_subj)
        label_map = {old:new for new,old in enumerate(unique_labels_subj)}
        Y_subj = np.array([label_map[y] for y in Y_subj])
        num_classes = len(unique_labels_subj)
        
        # train/val/test split
        X_train, X_temp, Y_train, Y_temp = train_test_split(X_subj, Y_subj, test_size=0.3, random_state=42, stratify=Y_subj)
        X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42, stratify=Y_temp)
        
        print(f"Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}, num_classes={num_classes}")
        
        # 데이터셋
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(500).batch(16)
        val_ds   = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(16)
        test_ds  = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(16)
        
        # 모델 빌드
        enc = build_encoder()
        ddpm_block = build_ddpm_block()
        dec = build_decoder()
        model = DiffusionClassificationModel(enc, ddpm_block, dec, num_classes=num_classes, alpha=0.9, lambda_diff=1.0)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer=optimizer)
        model.build(input_shape=(None,4,200,8))
        model.summary()
        
        # 학습
        early_stop = EarlyStopping(monitor='val_accuracy', patience=200, mode='max', restore_best_weights=True)
        history = model.fit(train_ds, epochs=500, validation_data=val_ds, callbacks=[early_stop])
        
        # 학습 곡선 시각화
        subj_folder = os.path.join(RESULT_DIR, f"s{subj}")
        os.makedirs(subj_folder, exist_ok=True)
        
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.legend(); plt.title("Loss Curve"); plt.xlabel("Epoch"); plt.ylabel("Loss")
        
        plt.subplot(1,2,2)
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.legend(); plt.title("Accuracy Curve"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
        
        plt.tight_layout()
        curve_path = os.path.join(subj_folder, "training_curves.png")
        plt.savefig(curve_path)
        plt.close()
        
        # 평가
        test_metrics = model.evaluate(test_ds, return_dict=True)
        print(f"Test Loss={test_metrics['loss']:.4f}, Test Acc={test_metrics['accuracy']:.4f}")
        
        # 예측
        y_pred_prob = model.predict(test_ds)[0]
        y_pred = np.argmax(y_pred_prob, axis=1)
        cm = confusion_matrix(Y_test, y_pred)
        report = classification_report(Y_test, y_pred)
        print("Confusion Matrix:"); print(cm)
        print("Classification Report:"); print(report)
        
        # 혼동행렬 시각화
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Pred"); plt.ylabel("True")
        plt.title("Confusion Matrix")
        cm_path = os.path.join(subj_folder, "cm.png")
        plt.savefig(cm_path)
        plt.close()
        
        # Classification report 저장
        with open(os.path.join(subj_folder,"cls_report.txt"), "w") as f:
            f.write(report)
        
        # 모델 저장
        model_save_path = os.path.join(subj_folder, "model_ddpm_cls.keras")
        model.save(model_save_path)
        print(f"Subject {subj} saved model to {model_save_path}")
