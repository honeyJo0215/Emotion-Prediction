#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

from JBlip.eeg_1dcnn import Sequence1DEncoder
from JBlip.blip_pretrain import blip_pretrain
from JBlip.blip_retrieval import blip_retrieval
from JBlip.blip_itm import blip_itm
from JBlip.blip_nlvr import blip_nlvr
from JBlip.blip_vqa import blip_vqa

def load_data(image_path, seq_path, label_path):
    """
    Load numpy arrays for images, sequences (EEG/R-PPG), and labels.
    images: [N, H, W, 3]
    sequences: [N, L_seq, C]
    labels: [N] integer labels in [0,3]
    """
    images = np.load(image_path)
    sequences = np.load(seq_path)
    labels = np.load(label_path)
    return images, sequences, labels

def prepare_dataset(images, sequences, labels, test_size=0.2, batch_size=32):
    # stratified split
    imgs_train, imgs_test, seqs_train, seqs_test, y_train, y_test = train_test_split(
        images, sequences, labels,
        test_size=test_size,
        stratify=labels,
        random_state=42
    )
    train_ds = tf.data.Dataset.from_tensor_slices(((imgs_train, seqs_train), y_train))
    test_ds  = tf.data.Dataset.from_tensor_slices(((imgs_test, seqs_test), y_test))
    train_ds = train_ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds  = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds, test_ds

def train_classification(model, train_ds, test_ds, epochs=10, lr=1e-4):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.fit(train_ds, validation_data=test_ds, epochs=epochs)
    loss, acc = model.evaluate(test_ds)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

def main():
    # --- Paths to your data (adjust as needed) ---
    image_path = 'data/images.npy'
    seq_path   = 'data/eeg.npy'
    label_path = 'data/labels.npy'

    print("Loading data...")
    images, sequences, labels = load_data(image_path, seq_path, label_path)
    print(f"Loaded {len(labels)} samples.")

    print("Preparing datasets...")
    train_ds, test_ds = prepare_dataset(images, sequences, labels,
                                        test_size=0.2, batch_size=32)

    # Create shared 1D-sequence encoder
    seq_encoder = Sequence1DEncoder(
        in_channels=sequences.shape[-1],
        hidden_size=768,
        num_layers=4
    )

    # -------------------
    # 1) Pretrain stage (contrastive + ITM + reconstruction)
    # -------------------
    pretrain_model = blip_pretrain(seq_encoder=seq_encoder, image_size=224)
    # TODO: implement .fit() loop for pretrain_model with your data
    # e.g., pretrain_model.fit(((imgs, seqs),), epochs=...)

    # -------------------
    # 2) Retrieval stage (contrastive)
    # -------------------
    retrieval_model = blip_retrieval(seq_encoder=seq_encoder, image_size=384)
    # TODO: implement .fit() loop for retrieval_model

    # -------------------
    # 3) VQA stage (generation) - optional for emotion classification
    # -------------------
    vqa_model = blip_vqa(seq_encoder=seq_encoder, image_size=480)
    # TODO: if you have question-answer pairs, train vqa_model

    # -------------------
    # 4) Emotion classification via ITM
    # -------------------
    itm_model = blip_itm(seq_encoder=seq_encoder, image_size=384, embed_dim=256)
    # Modify classification head for 4 emotion classes
    itm_model.itm_head = layers.Dense(4)
    print("Training ITM-based emotion classifier...")
    train_classification(itm_model, train_ds, test_ds, epochs=10, lr=1e-4)

    # -------------------
    # 5) Emotion classification via NLVR
    # -------------------
    nlvr_model = blip_nlvr(seq_encoder=seq_encoder, image_size=480)
    # Adjust NLVR cls_head for 4 classes
    nlvr_model.cls_head = tf.keras.Sequential([
        layers.Dense(256, activation='relu'),
        layers.Dense(4)
    ])
    print("Training NLVR-based emotion classifier...")
    train_classification(nlvr_model, train_ds, test_ds, epochs=10, lr=1e-4)

if __name__ == '__main__':
    main()
