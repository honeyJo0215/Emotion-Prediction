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

# Result directory
RESULT_DIR = "/home/bcml1/sigenv/_4월/DDPM_diffusion/DDPM_emotion1"
os.makedirs(RESULT_DIR, exist_ok=True)

# gpu limit
# def limit_gpu_memory(memory_limit_mib=10000):
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     if gpus:
#         try:
#             tf.config.experimental.set_virtual_device_configuration(
#                 gpus[0],
#                 [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit_mib)]
#             )
#             print(f"GPU memory limited to {memory_limit_mib} MiB.")
#         except RuntimeError as e:
#             print(e)
#     else:
#         print("No GPU available, using CPU.")

# limit_gpu_memory(10000)

# load features
def load_csp_features(csp_feature_dir="/home/bcml1/2025_EMOTION/DEAP_eeg_new_label_CSP"):
    X_list, Y_list, subjects_list, files_list = [], [], [], []
    for file_name in os.listdir(csp_feature_dir):
        if not file_name.endswith(".npy"):
            continue
        file_path = os.path.join(csp_feature_dir, file_name)
        try:
            data = np.load(file_path)  # shape: (4, T, 8)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
        if data.ndim != 3 or data.shape[0] != 4 or data.shape[2] != 8:
            print(f"Unexpected shape in {file_path}: {data.shape}")
            continue

        label_match = re.search(r'label(\d+)', file_name, re.IGNORECASE)
        if label_match is None:
            print(f"No label found in {file_name}, skipping file.")
            continue
        label = int(label_match.group(1))
        if label not in [0, 1, 2, 3]:
            print(f"Label {label} in file {file_name} is not in [0,1,2,3], skipping file.")
            continue

        # subject
        subject_match = re.search(r'subject(\d+)', file_name, re.IGNORECASE)
        subject = subject_match.group(1) if subject_match else 'unknown'
        T = data.shape[1]
        n_windows = T // 200
        if n_windows < 1:
            continue
        for i in range(n_windows):
            window = data[:, i*200:(i+1)*200, :]  # (4,200,8)
            X_list.append(window)
            Y_list.append(label)
            files_list.append(f"{file_name}_win{i}")
            subjects_list.append(subject)
    X = np.array(X_list)
    Y = np.array(Y_list)
    subjects = np.array(subjects_list)
    file_ids = np.array(files_list)
    print(f"Loaded {X.shape[0]} samples, each shape: {X.shape[1:]}")
    print(f"Unique labels found: {np.unique(Y)}")
    print(f"Unique subjects found: {np.unique(subjects)}")
    return X, Y, subjects, file_ids

# model

def build_encoder():
    inp = layers.Input(shape=(200,8,4))
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inp)
    x = layers.MaxPooling2D((2,2))(x)   # (100,4,32)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)   # (50,2,64)
    model = models.Model(inputs=inp, outputs=x, name="Encoder")
    return model

def build_ddpm_block():
    inp = layers.Input(shape=(50,2,64))
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(inp)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    out = layers.Conv2D(64, (3,3), activation=None, padding='same')(x)
    model = models.Model(inputs=inp, outputs=out, name="DDPM_Block")
    return model

def build_decoder():
    inp = layers.Input(shape=(50,2,64))
    x = layers.UpSampling2D(size=(2,2))(inp)  # (100,4,64)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D(size=(2,2))(x)       # (200,8,32)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    out = layers.Conv2D(4, (3,3), activation=None, padding='same')(x)
    model = models.Model(inputs=inp, outputs=out, name="Decoder")
    return model

# CNN branch
def build_freq_branch_with_diffusion(input_shape=(200,8,4), noise_std=0.05):
    inp = layers.Input(shape=input_shape)
    pre_processed = layers.Lambda(lambda x: tf.random.normal(tf.shape(x), mean=0.0, stddev=noise_std) + x)(inp)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(pre_processed)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    freq_feat = x
    noisy_freq_feat = layers.Lambda(lambda x: tf.random.normal(tf.shape(x), mean=0.0, stddev=noise_std) + x)(freq_feat)

    diffu_freq = layers.Dense(64, activation='relu')(noisy_freq_feat)
    freq_res = layers.Add()([freq_feat, diffu_freq])
    branch = models.Model(inputs=inp, outputs=freq_res, name="FreqBranchDiff")
    return branch

def build_chan_branch_with_diffusion(input_shape=(200,4,8), noise_std=0.05):
    inp = layers.Input(shape=input_shape)
    pre_processed = layers.Lambda(lambda x: tf.random.normal(tf.shape(x), mean=0.0, stddev=noise_std) + x)(inp)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(pre_processed)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    chan_feat = x
    noisy_chan_feat = layers.Lambda(lambda x: tf.random.normal(tf.shape(x), mean=0.0, stddev=noise_std) + x)(chan_feat)
    diffu_chan = layers.Dense(64, activation='relu')(noisy_chan_feat)
    chan_res = layers.Add()([chan_feat, diffu_chan])
    branch = models.Model(inputs=inp, outputs=chan_res, name="ChanBranchDiff")
    return branch

# combined model
class EmotionDDPMModel(tf.keras.Model):
    def __init__(self, encoder, ddpm_block, decoder, num_classes,
                 noise_std=0.02, alpha=0.9, lambda_diff=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.lambda_diff = lambda_diff
        self.sqrt_alpha = math.sqrt(alpha)
        self.sqrt_one_minus_alpha = math.sqrt(1 - alpha)
        # DDPM 
        self.encoder = encoder
        self.ddpm_block = ddpm_block
        self.decoder = decoder
        
        self.freq_branch = build_freq_branch_with_diffusion(input_shape=(200,8,4), noise_std=noise_std)
        self.chan_branch = build_chan_branch_with_diffusion(input_shape=(200,4,8), noise_std=noise_std)
        self.classifier_dense1 = layers.Dense(128, activation='relu')
        self.dropout1 = layers.Dropout(0.3)
        self.classifier_dense2 = layers.Dense(64, activation='relu')
        self.dropout2 = layers.Dropout(0.3)
        self.classifier_out = layers.Dense(num_classes, activation='softmax')
    
    def call(self, x, training=False):
        # x: (batch,4,200,8)
        # DDPM 
        # (4,200,8) → (200,8,4)
        x_perm = tf.transpose(x, perm=[0,2,3,1])
        # original latent
        latent_x0 = self.encoder(x_perm)           # (batch,50,2,64)
        # noise: ε ~ N(0,I)
        epsilon = tf.random.normal(tf.shape(x_perm))
        # noisy x: x_t = √α*x0 + √(1-α)*ε
        x_t = self.sqrt_alpha * x_perm + self.sqrt_one_minus_alpha * epsilon
        latent_t = self.encoder(x_t)               # (batch,50,2,64)
        # DDPM block
        epsilon_hat = self.ddpm_block(latent_t)      # (batch,50,2,64)
        # target noise in latent space: ε_target = (latent_t - √α*latent_x0)/√(1-α)
        epsilon_target = (latent_t - self.sqrt_alpha * latent_x0) / self.sqrt_one_minus_alpha
        # recoverd latent: latent0_hat = (latent_t - √(1-α)*ε̂)/√α
        latent0_hat = (latent_t - self.sqrt_one_minus_alpha * epsilon_hat) / self.sqrt_alpha
        # recover: latent0_hat → x0_hat (feature space: (batch,200,8,4))
        x0_hat = self.decoder(latent0_hat)
        # (batch,200,8,4) → (batch,4,200,8)
        restored = tf.transpose(x0_hat, perm=[0,3,1,2])
        
        # Frequency branch: (batch,4,200,8) → (batch,200,8,4)
        freq_input = tf.transpose(restored, perm=[0,2,3,1])
        freq_feat = self.freq_branch(freq_input)
        # Channel branch: (batch,4,200,8) → (batch,200,4,8)
        chan_input = tf.transpose(restored, perm=[0,2,1,3])
        chan_feat = self.chan_branch(chan_input)
        combined = layers.Concatenate()([freq_feat, chan_feat])
        x_cls = self.classifier_dense1(combined)
        x_cls = self.dropout1(x_cls, training=training)
        x_cls = self.classifier_dense2(x_cls)
        x_cls = self.dropout2(x_cls, training=training)
        class_output = self.classifier_out(x_cls)
        return class_output, restored, epsilon_hat, epsilon_target

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            class_output, restored, epsilon_hat, epsilon_target = self(x, training=True)
            loss_cls = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, class_output))
            loss_recon = tf.reduce_mean(tf.keras.losses.mse(x, restored))
            loss_diff = tf.reduce_mean(tf.keras.losses.mse(epsilon_target, epsilon_hat))
            loss = loss_cls + loss_recon + self.lambda_diff * loss_diff
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # accuracy 
        acc = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(y, class_output))
        return {"loss": loss, "loss_cls": loss_cls, "loss_recon": loss_recon, "loss_diff": loss_diff, "accuracy": acc}

    def test_step(self, data):
        x, y = data
        class_output, restored, epsilon_hat, epsilon_target = self(x, training=False)
        loss_cls = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, class_output))
        loss_recon = tf.reduce_mean(tf.keras.losses.mse(x, restored))
        loss_diff = tf.reduce_mean(tf.keras.losses.mse(epsilon_target, epsilon_hat))
        loss = loss_cls + loss_recon + self.lambda_diff * loss_diff
        acc = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(y, class_output))
        return {"loss": loss, "loss_cls": loss_cls, "loss_recon": loss_recon, "loss_diff": loss_diff, "accuracy": acc}

# Main
if __name__ == "__main__":
    # CSP
    CSP_FEATURE_DIR = "/home/bcml1/2025_EMOTION/DEAP_eeg_new_label_CSP"
    X, Y, subjects, file_ids = load_csp_features(CSP_FEATURE_DIR)
    
    unique_subjects = np.unique(subjects)
    for subj in unique_subjects:
        
        if not (12 <= int(subj) < 33):
            continue
        print(f"\n========== Intra-subject Emotion Classification: Subject = {subj} ==========")
        subj_mask = (subjects == subj)
        X_subj = X[subj_mask]
        Y_subj = Y[subj_mask]
        
        unique_labels_subj = np.sort(np.unique(Y_subj))
        label_map = {old: new for new, old in enumerate(unique_labels_subj)}
        Y_subj = np.array([label_map[y] for y in Y_subj])
        num_classes = len(unique_labels_subj)
        print(f"Using labels: {unique_labels_subj} remapped to 0 ~ {num_classes-1}")
        
        X_train, X_temp, Y_train, Y_temp = train_test_split(X_subj, Y_subj, test_size=0.3, random_state=42, stratify=Y_subj)
        X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42, stratify=Y_temp)
        print(f"Subject {subj}: Train {X_train.shape[0]} samples, Val {X_val.shape[0]} samples, Test {X_test.shape[0]} samples.")
        print("Train label distribution:", np.unique(Y_train, return_counts=True))
        print("Val label distribution:", np.unique(Y_val, return_counts=True))
        print("Test label distribution:", np.unique(Y_test, return_counts=True))
        
        unique_labels_train, counts_train = np.unique(Y_train, return_counts=True)
        max_count_train = counts_train.max()
        print(f"Balancing training classes to {max_count_train} samples per label using diffusion noise augmentation.")
        augmented_train_X = []
        augmented_train_Y = []
        for label in unique_labels_train:
            label_mask = (Y_train == label)
            current_count = np.sum(label_mask)
            num_to_augment = max_count_train - current_count
            if num_to_augment > 0:
                indices = np.where(label_mask)[0]
                for i in range(num_to_augment):
                    random_index = np.random.choice(indices)
                    sample = X_train[random_index]
                    sample_tensor = tf.convert_to_tensor(sample, dtype=tf.float32)
                    augmented_sample = sample_tensor + tf.random.normal(tf.shape(sample_tensor), mean=0.0, stddev=0.05)
                    augmented_train_X.append(augmented_sample.numpy())
                    augmented_train_Y.append(label)
        if augmented_train_X:
            X_train = np.concatenate([X_train, np.array(augmented_train_X)], axis=0)
            Y_train = np.concatenate([Y_train, np.array(augmented_train_Y)], axis=0)
            print(f"After augmentation (train): {X_train.shape[0]} samples.")
        
        subj_folder = os.path.join(RESULT_DIR, f"s{subj}")
        os.makedirs(subj_folder, exist_ok=True)
        
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(1000).batch(16)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(16)
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(16)
        
        encoder = build_encoder()
        ddpm_block = build_ddpm_block()
        decoder = build_decoder()
        model = EmotionDDPMModel(encoder, ddpm_block, decoder, num_classes=num_classes,
                                  noise_std=0.02, alpha=0.9, lambda_diff=1.0)
        optimizer = tf.keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-4, decay_steps=100000, decay_rate=0.9, staircase=True), clipnorm=1.0)
        model.compile(optimizer=optimizer)
        model.build(input_shape=(None,4,200,8))
        model.summary()
        
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=2000, mode='max', restore_best_weights=True)
        history = model.fit(train_dataset, epochs=500, validation_data=val_dataset, callbacks=[early_stopping])
        
        # result
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curve')
        plt.subplot(1,2,2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy Curve')
        train_curve_path = os.path.join(subj_folder, "training_curves.png")
        plt.savefig(train_curve_path)
        plt.close()
        print(f"Subject {subj}: Training curves saved to {train_curve_path}")
        
        # test
        test_metrics = model.evaluate(test_dataset, return_dict=True)
        print(f"Subject {subj}: Test Loss: {test_metrics['loss']:.4f}, Test Accuracy: {test_metrics['accuracy']:.4f}")

        # Confusion Matrix, Classification Report
        y_pred_prob = model.predict(test_dataset)[0]
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
        
        # model save
        model_save_path = os.path.join(subj_folder, "model_eeg_emotion_ddpm.keras")
        model.save(model_save_path)
        print(f"Subject {subj}: Model saved to {model_save_path}")
