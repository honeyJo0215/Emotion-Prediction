# CNN 기반의 Dual-Stream Feature Extractor + Cross-Modal Transformer + rPPG 모달 추가
# Stratify 적용, Focal Loss 구현, 오버샘플링 적용
# EEG, Eye, rPPG 데이터가 trial 및 segment 단위로 정렬되어 동일하게 로드됨

import os
import re
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.layers import (
    Conv2D, Dense, Flatten, Dropout, AveragePooling2D, DepthwiseConv2D, LayerNormalization,
    BatchNormalization, MultiHeadAttention, Concatenate, Input, TimeDistributed, Add, Softmax, Lambda,
    MaxPooling2D, GlobalAveragePooling2D, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from collections import Counter

# GPU limit
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

# label mapping
EMOTION_MAPPING = {"Negative": 0, "Positive": 1, "Neutral": 2}

# hyper parameter
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-3  

# data path
EEG_DATA_PATH = "/home/bcml1/2025_EMOTION/DEAP_EEG_10s_1soverlap_labeled"
EYE_CROP_PATH = "/home/bcml1/2025_EMOTION/eye_crop"
RPPG_DATA_PATH = "/home/bcml1/yigeon06/PPG_SEG"  
SAVE_PATH = "/home/bcml1/sigenv/_2주차_eye+eeg+rppg/final_result1"
os.makedirs(SAVE_PATH, exist_ok=True)
SUBJECTS = [f"s{str(i).zfill(2)}" for i in range(1, 23)]

# feature extractor
def create_eeg_feature_extractor():
    eeg_input = Input(shape=(32,1280,1), name="EEG_Input")
    x = Conv2D(64, kernel_size=(4,16), strides=(2,8), padding='valid', activation='relu')(eeg_input)
    x = BatchNormalization()(x)
    x = DepthwiseConv2D(kernel_size=(6,6), strides=(3,3), padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)
    x = Conv2D(128, kernel_size=(1,1), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    out = Dense(128, activation='relu')(x)
    out = Dropout(0.3)(out)
    return Model(inputs=eeg_input, outputs=out, name="EEGFeatureExtractor")

def create_eye_feature_extractor():
    eye_input = Input(shape=(100,8,64,3), name="Eye_Input")
    x = TimeDistributed(Conv2D(32, kernel_size=(3,3), activation="relu", padding="same"))(eye_input)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2,2)))(x)
    x = TimeDistributed(Conv2D(64, kernel_size=(3,3), activation="relu", padding="same"))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2,2)))(x)
    x = TimeDistributed(Conv2D(128, kernel_size=(3,3), activation="relu", padding="same"))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    x = GlobalAveragePooling1D()(x)
    out = Dense(128, activation="relu")(x)
    out = Dropout(0.3)(out)
    return Model(inputs=eye_input, outputs=out, name="EyeFeatureExtractor")

def build_segment_cnn():
    seg_input = Input(shape=(100,13,3))
    y = Conv2D(32, kernel_size=(3,5), strides=(1,2), padding='same', activation='relu')(seg_input)
    y = BatchNormalization()(y)
    
    # noise filtering
    y = DepthwiseConv2D(kernel_size=(3,3), padding='same', activation='relu')(y)
    y = BatchNormalization()(y)

    y = AveragePooling2D(pool_size=(2,2), strides=(2,2))(y)
    
    # high feature
    y = Conv2D(64, kernel_size=(1,1), padding='valid', activation='relu')(y)
    y = BatchNormalization()(y)
    y = Flatten()(y)

    y = Dense(128, activation='relu')(y)
    y = LayerNormalization()(y)

    y = Dropout(0.3)(y)

    return Model(inputs=seg_input, outputs=y, name="SegmentCNN")

def create_rppg_feature_extractor():
    rppg_input = Input(shape=(100,13,3), name="RPPG_Input")
    segment_model = build_segment_cnn() 
    x = segment_model(rppg_input)  # (128,)
    return Model(inputs=rppg_input, outputs=x, name="RPPGFeatureExtractor")

# inter modality fusion
def create_inter_modality_fusion(feat1, feat2, num_heads=4, d_model=128, dropout_rate=0.1):
    q = Dense(d_model)(feat1)
    kv = Dense(d_model)(feat2)
    q = Lambda(lambda x: tf.expand_dims(x, axis=1))(q)
    kv = Lambda(lambda x: tf.expand_dims(x, axis=1))(kv)
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(query=q, key=kv, value=kv)
    attn = Dropout(dropout_rate)(attn)
    attn = Add()([q, attn])
    attn = LayerNormalization(epsilon=1e-6)(attn)
    attn = Lambda(lambda x: tf.squeeze(x, axis=1))(attn)
    return attn

# intra modality encoder
def create_intra_encoding(feature, num_heads=4, d_model=128, dropout_rate=0.1):
    x = Lambda(lambda x: tf.expand_dims(x, axis=1))(feature)
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(query=x, key=x, value=x)
    attn = Dropout(dropout_rate)(attn)
    x = Add()([x, attn])
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Lambda(lambda x: tf.squeeze(x, axis=1))(x)
    return x

# model build
def build_full_multimodal_model(num_classes=3):
    
    eeg_input = Input(shape=(32,1280,1), name="EEG_Input")
    eye_input = Input(shape=(100,8,64,3), name="Eye_Input")
    rppg_input = Input(shape=(100,13,3), name="RPPG_Input")

    eeg_feat = create_eeg_feature_extractor()(eeg_input)   # (batch,128)
    eye_feat = create_eye_feature_extractor()(eye_input)   # (batch,128)
    rppg_feat = create_rppg_feature_extractor()(rppg_input)  # (batch,128)
    
    # Inter-modality transformers 
    inter_eeg_rppg = create_inter_modality_fusion(eeg_feat, rppg_feat)
    inter_eeg_eye  = create_inter_modality_fusion(eeg_feat, eye_feat)
    inter_eye_rppg = create_inter_modality_fusion(eye_feat, rppg_feat)
    
    # Intra-modality encoders
    intra_eeg  = create_intra_encoding(eeg_feat)
    intra_eye  = create_intra_encoding(eye_feat)
    intra_rppg = create_intra_encoding(rppg_feat)
    
    # branches
    pred_inter_eeg_rppg = Dense(num_classes, activation="softmax", name="Inter_EEG_RPPG")(inter_eeg_rppg)
    pred_inter_eeg_eye  = Dense(num_classes, activation="softmax", name="Inter_EEG_Eye")(inter_eeg_eye)
    pred_inter_eye_rppg = Dense(num_classes, activation="softmax", name="Inter_Eye_RPPG")(inter_eye_rppg)
    pred_intra_eeg  = Dense(num_classes, activation="softmax", name="Intra_EEG")(intra_eeg)
    pred_intra_eye  = Dense(num_classes, activation="softmax", name="Intra_Eye")(intra_eye)
    pred_intra_rppg = Dense(num_classes, activation="softmax", name="Intra_RPPG")(intra_rppg)
    
    # fusion
    fused_for_weight = Concatenate()([inter_eeg_rppg, inter_eeg_eye, inter_eye_rppg,
                                      intra_eeg, intra_eye, intra_rppg])
    weight_logits = Dense(6, activation=None, name="Final_Weight_Logits")(fused_for_weight)
    final_weights = Softmax(name="Final_Weights")(weight_logits)
    
    outputs = [pred_inter_eeg_rppg, pred_inter_eeg_eye, pred_inter_eye_rppg,
               pred_intra_eeg, pred_intra_eye, pred_intra_rppg, final_weights]
    
    model = Model(inputs=[eeg_input, eye_input, rppg_input], outputs=outputs, name="Multimodal_Emotion_Classifier")
    return model

# custom model classes
class FullMultimodalEmotionClassifier(tf.keras.Model):
    def __init__(self, base_model, **kwargs):
        super(FullMultimodalEmotionClassifier, self).__init__(**kwargs)
        self.base_model = base_model

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)

    def train_step(self, data):
        x, y = data  # y: (batch,)
        y_true = y if not isinstance(y, dict) else y["Inter_EEG_RPPG"]
        with tf.GradientTape() as tape:
            preds = self(x, training=True)
            p1, p2, p3, p4, p5, p6, weights = preds
            loss1 = tf.keras.losses.sparse_categorical_crossentropy(y_true, p1)
            loss2 = tf.keras.losses.sparse_categorical_crossentropy(y_true, p2)
            loss3 = tf.keras.losses.sparse_categorical_crossentropy(y_true, p3)
            loss4 = tf.keras.losses.sparse_categorical_crossentropy(y_true, p4)
            loss5 = tf.keras.losses.sparse_categorical_crossentropy(y_true, p5)
            loss6 = tf.keras.losses.sparse_categorical_crossentropy(y_true, p6)
            losses = tf.stack([loss1, loss2, loss3, loss4, loss5, loss6], axis=-1)
            weighted_loss = tf.reduce_sum(weights * losses, axis=-1)
            loss = tf.reduce_mean(weighted_loss)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y_true, p1)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss})
        return results

    def test_step(self, data):
        x, y = data
        y_true = y if not isinstance(y, dict) else y["Inter_EEG_RPPG"]
        preds = self(x, training=False)
        p1 = preds[0]
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true, p1))
        self.compiled_metrics.update_state(y_true, p1)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss})
        return results

# oversampling
def oversample_data(train_eeg, train_eye, train_rppg, train_labels):
    unique_classes, counts = np.unique(train_labels, return_counts=True)
    max_count = np.max(counts)
    new_eeg, new_eye, new_rppg, new_labels = [], [], [], []
    for cls in unique_classes:
        indices = np.where(train_labels == cls)[0]
        rep_factor = int(np.ceil(max_count / len(indices)))
        eeg_rep = np.repeat(train_eeg[indices], rep_factor, axis=0)
        eye_rep = np.repeat(train_eye[indices], rep_factor, axis=0)
        rppg_rep = np.repeat(train_rppg[indices], rep_factor, axis=0)
        label_rep = np.repeat(train_labels[indices], rep_factor, axis=0)
        perm = np.random.permutation(eeg_rep.shape[0])
        new_eeg.append(eeg_rep[perm][:max_count])
        new_eye.append(eye_rep[perm][:max_count])
        new_rppg.append(rppg_rep[perm][:max_count])
        new_labels.append(label_rep[perm][:max_count])
    new_eeg = np.concatenate(new_eeg, axis=0)
    new_eye = np.concatenate(new_eye, axis=0)
    new_rppg = np.concatenate(new_rppg, axis=0)
    new_labels = np.concatenate(new_labels, axis=0)
    perm_all = np.random.permutation(new_eeg.shape[0])
    return new_eeg[perm_all], new_eye[perm_all], new_rppg[perm_all], new_labels[perm_all]

# load
def load_rppg_data(subject):
   
    subject_path = os.path.join(RPPG_DATA_PATH, subject)
    if not os.path.exists(subject_path):
        print(f"rPPG folder not found for {subject}")
        return None, []
    all_segments = []  
    sample_indices = [] 
    for trial in range(1, 41):
        trial_str = f"{trial:02d}"
        roi_list = []
        for roi in range(13):
            file_name = f"{subject}_trial{trial_str}_roi{roi}.npy"
            file_path = os.path.join(subject_path, file_name)
            if not os.path.exists(file_path):
                print(f"rPPG file not found: {file_path}")
                continue
            data = np.load(file_path)  # shape: (51,3,1,500)
            roi_list.append(data)
        if len(roi_list) != 13:
            print(f"Trial {trial_str}: expected 13 ROI files, got {len(roi_list)}. Skipping trial.")
            continue
        trial_data = np.concatenate(roi_list, axis=2)  # (51,3,13,500)
        indices = np.linspace(0, 499, num=100, endpoint=True, dtype=int)
        for seg in range(trial_data.shape[0]):  # 51 segments
            seg_data = trial_data[seg]  # (3,13,500)
            seg_data = seg_data[..., indices]  # (3,13,100)
            seg_data = np.transpose(seg_data, (2,1,0))  # (100,13,3)
            all_segments.append(seg_data)
            sample_indices.append(trial)
    all_segments = np.array(all_segments)  # (num_trials*51, 100, 13, 3)
    return all_segments, sample_indices

# functions
def compute_class_weights(labels, num_classes=3):
    class_counts = np.bincount(labels, minlength=num_classes)
    total_samples = np.sum(class_counts)
    class_weights = total_samples / (num_classes * class_counts)
    class_weights /= np.max(class_weights)
    return class_weights.astype(np.float32)


def focal_loss(alpha, gamma=2.0):
    def loss(y_true, y_pred):
        y_true_one_hot = tf.one_hot(y_true, depth=len(alpha))
        alpha_factor = tf.gather(alpha, tf.argmax(y_true_one_hot, axis=-1))
        loss_val = -alpha_factor * y_true_one_hot * tf.pow(1 - y_pred, gamma) * tf.math.log(y_pred + 1e-7)
        return tf.reduce_mean(loss_val)
    return loss

def downsample_eye_frame(frame):
    return cv2.resize(frame, (32, 8), interpolation=cv2.INTER_AREA)

def reshape_eye_frame(data):
    if len(data.shape) == 4 and data.shape[0] > 0:
        reshaped_data = np.mean(data, axis=0)
        return downsample_eye_frame(reshaped_data)
    elif len(data.shape) == 3 and data.shape == (32, 64, 3):
        return downsample_eye_frame(data)
    else:
        raise ValueError(f"Unexpected shape: {data.shape}, expected (N, 32, 64, 3) or (32, 64, 3)")



def load_multimodal_data(subject):
    eeg_data, eye_data, labels, sample_indices = [], [], [], []
    eeg_pattern = re.compile(rf"{subject}_sample_(\d+)_segment_(\d+)_label_(\w+).npy")
    for sample_index in range(1, 41):
        sample_number = f"{sample_index:02d}"
        print(f"\n🟢 Processing {subject} - Sample {sample_number}")
        eeg_files = [f for f in os.listdir(EEG_DATA_PATH) if eeg_pattern.match(f) and f"sample_{sample_number}" in f]
        if not eeg_files:
            print(f"🚨 No EEG file found for {subject} - Sample {sample_number}")
            continue
        for file_name in eeg_files:
            match = eeg_pattern.match(file_name)
            if not match:
                continue
            segment_index = int(match.group(2))
            emotion_label = match.group(3)
            eeg_file_path = os.path.join(EEG_DATA_PATH, file_name)
            eeg_segment = np.load(eeg_file_path)  # shape: (32,1280,1)
            eye_subject_path = os.path.join(EYE_CROP_PATH, subject)
            if not os.path.exists(eye_subject_path):
                print(f"🚨 Subject folder not found: {eye_subject_path}")
                continue
            trial_number = sample_index
            expected_start = (segment_index - 1) * 500
            frame_indices = set()
            file_mapping = {}
            for f in os.listdir(eye_subject_path):
                try:
                    if not f.startswith(subject) or not f.endswith(".npy"):
                        continue
                    match_frame = re.search(r"trial(\d+).avi_frame(\d+)", f)
                    if not match_frame:
                        print(f"Skipping invalid file name: {f} (No trial/frame pattern found)")
                        continue
                    file_trial_number = int(match_frame.group(1))
                    frame_number = int(match_frame.group(2))
                    if file_trial_number == trial_number:
                        frame_indices.add(frame_number)
                        file_mapping[frame_number] = os.path.join(eye_subject_path, f)
                except ValueError as e:
                    print(f"Error processing file {f}: {e}")
                    continue
            frame_indices = sorted(frame_indices)
            print(f" Available frame indices: {frame_indices[:10]} ... {frame_indices[-10:]}")
            window_length = 500
            if len(frame_indices) < window_length:
                print(f"Warning: Not enough frames ({len(frame_indices)}) for segment {segment_index:03d}. Skipping Eye Crop.")
                eye_data.append(None)
                eeg_data.append(eeg_segment)
                labels.append(EMOTION_MAPPING[emotion_label])
                sample_indices.append(sample_index)
                continue
            selected_frames = frame_indices[:window_length]
            if len(selected_frames) < window_length:
                while len(selected_frames) < window_length:
                    selected_frames.append(selected_frames[-1])
                    print("frame copied")
            indices = np.linspace(0, window_length - 1, num=100, endpoint=True, dtype=int)
            selected_frames = [selected_frames[i] for i in indices]
            eye_frame_files = []
            for frame in selected_frames:
                if frame in file_mapping:
                    eye_frame_files.append(file_mapping[frame])
                if len(eye_frame_files) == 100:
                    break
            eye_frame_stack = []
            for f in eye_frame_files:
                frame_data = np.load(f)
                frame_data = reshape_eye_frame(frame_data)
                if frame_data.shape[-2] == 32:
                    pad_width = [(0, 0)] * frame_data.ndim
                    pad_width[-2] = (16, 16)
                    frame_data = np.pad(frame_data, pad_width, mode='constant', constant_values=0)
                eye_frame_stack.append(frame_data)
            if len(eye_frame_stack) == 100:
                eye_data.append(np.stack(eye_frame_stack, axis=0))
                eeg_data.append(eeg_segment)
                labels.append(EMOTION_MAPPING[emotion_label])
                sample_indices.append(sample_index)
            else:
                print(f"⚠ Warning: Found only {len(eye_frame_stack)} matching frames for segment {segment_index:03d}")
    print(f"✅ Total EEG Samples Loaded: {len(eeg_data)}")
    print(f"✅ Total Eye Crop Samples Loaded: {len([e for e in eye_data if e is not None])}")
    print(f"✅ Labels Loaded: {len(labels)}")
    return (np.array(eeg_data), 
            np.array([e if e is not None else np.zeros((100,8,64,3)) for e in eye_data]), 
            np.array(labels), sample_indices)


def load_all_modalities(subject):
    eeg_data, eye_data, labels, seg_ids = load_multimodal_data(subject)

    rppg_data, rppg_seg_ids = load_rppg_data(subject)
    if len(labels) != len(rppg_data):
        print(f"Warning: EEG/Eye segments: {len(labels)}, rPPG segments: {len(rppg_data)}")
    else:
        eeg_order = sorted(range(len(seg_ids)), key=lambda i: seg_ids[i])
        rppg_order = sorted(range(len(rppg_seg_ids)), key=lambda i: rppg_seg_ids[i])
        eeg_data = eeg_data[eeg_order]
        eye_data = eye_data[eeg_order]
        labels = labels[eeg_order]
        rppg_data = rppg_data[rppg_order]
        seg_ids = [seg_ids[i] for i in eeg_order]
    eeg_data = eeg_data.astype(np.float32) 
    eye_data = eye_data.astype(np.float32) / 255.0
    rppg_data = rppg_data.astype(np.float32) / 255.0  
    return eeg_data, eye_data, rppg_data, np.array(labels), seg_ids

# oversampling
def oversample_data(train_eeg, train_eye, train_rppg, train_labels):
    unique_classes, counts = np.unique(train_labels, return_counts=True)
    max_count = np.max(counts)
    new_eeg, new_eye, new_rppg, new_labels = [], [], [], []
    for cls in unique_classes:
        indices = np.where(train_labels == cls)[0]
        rep_factor = int(np.ceil(max_count / len(indices)))
        eeg_rep = np.repeat(train_eeg[indices], rep_factor, axis=0)
        eye_rep = np.repeat(train_eye[indices], rep_factor, axis=0)
        rppg_rep = np.repeat(train_rppg[indices], rep_factor, axis=0)
        label_rep = np.repeat(train_labels[indices], rep_factor, axis=0)
        perm = np.random.permutation(eeg_rep.shape[0])
        new_eeg.append(eeg_rep[perm][:max_count])
        new_eye.append(eye_rep[perm][:max_count])
        new_rppg.append(rppg_rep[perm][:max_count])
        new_labels.append(label_rep[perm][:max_count])
    new_eeg = np.concatenate(new_eeg, axis=0)
    new_eye = np.concatenate(new_eye, axis=0)
    new_rppg = np.concatenate(new_rppg, axis=0)
    new_labels = np.concatenate(new_labels, axis=0)
    perm_all = np.random.permutation(new_eeg.shape[0])
    return new_eeg[perm_all], new_eye[perm_all], new_rppg[perm_all], new_labels[perm_all]

def compute_class_weights(labels, num_classes=3):
    class_counts = np.bincount(labels, minlength=num_classes)
    total_samples = np.sum(class_counts)
    class_weights = total_samples / (num_classes * class_counts)
    class_weights /= np.max(class_weights)
    return class_weights.astype(np.float32)

def focal_loss(alpha, gamma=2.0):
    def loss(y_true, y_pred):
        y_true_one_hot = tf.one_hot(y_true, depth=len(alpha))
        alpha_factor = tf.gather(alpha, tf.argmax(y_true_one_hot, axis=-1))
        loss_val = -alpha_factor * y_true_one_hot * tf.pow(1 - y_pred, gamma) * tf.math.log(y_pred + 1e-7)
        return tf.reduce_mean(loss_val)
    return loss

# train/test
def train_multimodal():
    for subject in SUBJECTS:
        print(f"\n=================== Training subject: {subject} ====================")
        eeg_data, eye_data, rppg_data, labels, seg_ids = load_all_modalities(subject)
        if rppg_data is None:
            continue
        # EEG, Eye, rPPG 
        train_eeg, test_eeg, train_eye, test_eye, train_labels, test_labels = train_test_split(
            eeg_data, eye_data, labels, test_size=0.2, random_state=42, stratify=labels
        )
        rppg_train, rppg_test = train_test_split(rppg_data, test_size=0.2, random_state=42)
        
        # Train/Validation split
        train_eeg, valid_eeg, train_eye, valid_eye, train_labels, valid_labels = train_test_split(
            train_eeg, train_eye, train_labels, test_size=0.2, stratify=train_labels, random_state=42
        )
        rppg_train, rppg_valid = train_test_split(rppg_train, test_size=0.2, random_state=42)
       
        n_train = min(train_eeg.shape[0], train_eye.shape[0], rppg_train.shape[0])
        train_eeg = train_eeg[:n_train]
        train_eye = train_eye[:n_train]
        train_rppg = rppg_train[:n_train]
        train_labels = train_labels[:n_train]
        
        n_valid = min(valid_eeg.shape[0], valid_eye.shape[0], rppg_valid.shape[0])
        valid_eeg = valid_eeg[:n_valid]
        valid_eye = valid_eye[:n_valid]
        valid_rppg = rppg_valid[:n_valid]
        valid_labels = valid_labels[:n_valid]
        
        n_test = min(test_eeg.shape[0], test_eye.shape[0], rppg_test.shape[0])
        test_eeg = test_eeg[:n_test]
        test_eye = test_eye[:n_test]
        test_rppg = rppg_test[:n_test]
        test_labels = test_labels[:n_test]
        
        # oversampling
        train_eeg, train_eye, train_rppg, train_labels = oversample_data(train_eeg, train_eye, train_rppg, train_labels)
        print(f"After oversampling - Train Samples: {train_eeg.shape[0]}")
        
        alpha_values = compute_class_weights(train_labels)
        
        base_model = build_full_multimodal_model(num_classes=3)
        model = FullMultimodalEmotionClassifier(base_model)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                      loss=focal_loss(alpha=alpha_values),
                      metrics=["accuracy"])
        model.summary()
        
        for epoch in range(EPOCHS):
            try:
                print(f"\nSubject {subject} - Epoch {epoch+1}/{EPOCHS}")
                model.fit(
                    [train_eeg, train_eye, train_rppg], train_labels,
                    validation_data=([valid_eeg, valid_eye, valid_rppg], valid_labels),
                    epochs=1, batch_size=BATCH_SIZE
                )
            except tf.errors.ResourceExhaustedError:
                print("OOM, skipping...")
                continue
        
        subject_save_path = os.path.join(SAVE_PATH, subject)
        os.makedirs(subject_save_path, exist_ok=True)
        weight_path = os.path.join(subject_save_path, f"{subject}_multimodal_model.weights.h5")
        model.save_weights(weight_path)
        print(f"model saved: {weight_path}")
        
        predictions = model.predict([test_eeg, test_eye, test_rppg])
        inter_pred = predictions[0]
        predicted_labels = np.argmax(inter_pred, axis=-1)
        test_report = classification_report(
            test_labels, predicted_labels,
            target_names=["Negative", "Positive", "Neutral"],
            labels=[0, 1, 2],
            zero_division=0
        )
        print(f"\nTest Report for {subject}")
        print(test_report)
        report_path = os.path.join(subject_save_path, f"{subject}_test_report.txt")
        with open(report_path, "w") as f:
            f.write(test_report)
        print(f"Report saved: {report_path}")

# MAIN
if __name__ == "__main__":
    train_multimodal()
