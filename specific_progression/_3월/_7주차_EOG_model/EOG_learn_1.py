import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Set the data path
EOG_BASE = '/home/bcml1/2025_EMOTION/DEAP_EOG'
LABEL_BASE = '/home/bcml1/2025_EMOTION/DEAP_four_labels'

NUM_SUBJECTS = 22

def load_data():
    X, y, subjects = [], [], []
    for i in range(1, NUM_SUBJECTS+1):
        subj = f's{i:02d}'
        subj_eog_dir = os.path.join(EOG_BASE, subj)
        label_path = os.path.join(LABEL_BASE, f'{subj}_emotion_labels.npy')
        # sort
        eog_files = sorted(glob.glob(os.path.join(subj_eog_dir, '*_eog.npy')))
        
        if not os.path.exists(label_path):
            print(f"라벨 파일 {label_path} 이(가) 없습니다.")
            continue
        
        labels = np.load(label_path)  # shape: (40,)
        
        if len(eog_files) != 40 or labels.shape[0] != 40:
            print(f"{subj}: trial 수가 40이 아닙니다. (영상: {len(eog_files)}, 라벨: {labels.shape[0]})")
            continue
        
        for file, label in zip(eog_files, labels):
            data = np.load(file)  # data shape: (T, 4)
            if data.shape[0] == 0:
                continue
            X.append(data)
            y.append(label)
            subjects.append(subj)
    return X, np.array(y), subjects

# Transformer Encoder Block 
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    
    def call(self, inputs, training, mask=None):
        attn_output = self.att(inputs, inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def build_model(input_length, num_channels=4, num_classes=4, embed_dim=64, num_heads=4, ff_dim=128):
    """
    Input: (input_length, 4)
    Process each channel as a separate branch, then concatenate them and pass them to the Transformer Encoder.
    """
    inputs = layers.Input(shape=(input_length, num_channels))
    
    branches = []
    for i in range(num_channels):
        x_i = layers.Lambda(lambda x, idx=i: x[:, :, idx:idx+1])(inputs)
        x_i = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x_i)
        x_i = layers.MaxPooling1D(pool_size=2)(x_i)
        branches.append(x_i)
    
    x = layers.Concatenate(axis=-1)(branches)
    x = layers.Dense(embed_dim, activation='relu')(x)
    
    # Transformer Encoder
    transformer_block = TransformerEncoder(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)
    x = transformer_block(x, training=True)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def normalize_eog(X_padded, seq_lengths, num_channels=4):
    """
    For each sample, each channel, normalised by the mean/stdev over the data before padding.
    Padding left at zero.
    """
    X_norm = np.copy(X_padded)
    for i, L in enumerate(seq_lengths):
        for ch in range(num_channels):
            valid_data = X_padded[i, :L, ch]
            mean = valid_data.mean()
            std = valid_data.std() if valid_data.std() > 0 else 1.0
            X_norm[i, :L, ch] = (valid_data - mean) / std
    return X_norm

def main():
    print("Loading data...")
    X_list, y, subjects = load_data()
    if len(X_list) == 0:
        print("No data found.")
        return
    
    seq_lengths = [x.shape[0] for x in X_list]
    max_len = max(seq_lengths)
    print(f"Maximum sequence length: {max_len}")
    
    X_padded = pad_sequences(X_list, maxlen=max_len, dtype='float32', padding='post', truncating='post')
    print(f"Input data shape: {X_padded.shape}, label shape: {y.shape}")
    
    X_norm = normalize_eog(X_padded, seq_lengths, num_channels=4)
    
    test_subject = "s01"
    train_indices = [i for i, subj in enumerate(subjects) if subj != test_subject]
    test_indices = [i for i, subj in enumerate(subjects) if subj == test_subject]
    
    X_train_val = X_norm[train_indices]
    y_train_val = y[train_indices]
    
    X_test_all = X_norm[test_indices]
    y_test_all = y[test_indices]
    
    print(f"Training/validation data: {X_train_val.shape}, {len(y_train_val)}")
    print(f"LOSO test subject ({test_subject}) data: {X_test_all.shape}, {len(y_test_all)}")

    # training/validation split 
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )
    
    X_finetune, X_test, y_finetune, y_test = train_test_split(
        X_test_all, y_test_all, test_size=0.75, random_state=42, stratify=y_test_all
    )
    print(f"파인튜닝 데이터: {X_finetune.shape}, {len(y_finetune)}")
    print(f"최종 테스트 데이터: {X_test.shape}, {len(y_test)}")
    
    model = build_model(input_length=max_len, num_channels=4, num_classes=4)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.summary()
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=50, restore_best_weights=True)
    ]
    
    history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_val, y_val), callbacks=callbacks)
    
    model.save('model_initial.keras')
    print("Saving initial model complete: model_initial.keras")
    
    print("Fine-tuning in progress...")
    model.fit(X_finetune, y_finetune, epochs=10, batch_size=8, validation_data=(X_val, y_val))
    
    test_loss, test_acc = model.evaluate(X_test, y_test)    
    print(f"Final test evaluation - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
    
    model.save('model_finetuned.keras')    
    print("Finished saving the fine-tuned model: model_finetuned.keras")

if __name__ == "__main__":
    main()
