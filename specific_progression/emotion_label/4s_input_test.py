import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# 데이터 로더 정의
class EEGDataset:
    def __init__(self, data_dir):
        self.file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])  # (band, time, channel)
        return np.transpose(data, (1, 2, 0))  # (time, channel, band)

# 데이터 변환 정의
def add_noise(data):
    noise = np.random.normal(0, 0.1, data.shape)
    return data + noise

def flip_vertical(data):
    return np.flip(data, axis=0)

def flip_horizontal(data):
    return np.flip(data, axis=1)

transformations = [add_noise, flip_vertical, flip_horizontal]

# Self-Supervised 데이터셋 정의
class PretextTaskDataset:
    def __init__(self, dataset, transformations):
        self.dataset = dataset
        self.transformations = transformations

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        transform_idx = np.random.randint(len(self.transformations))
        transformed_data = self.transformations[transform_idx](data)
        return transformed_data, transform_idx

# Self-Supervised CNN 모델 정의
def create_self_supervised_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# 데이터 로드
def load_data(data_dir):
    dataset = EEGDataset(data_dir)
    data = [dataset[i] for i in range(len(dataset))]
    return np.array(data)

# 데이터 경로
train_data_dir = "/home/bcml1/2025_EMOTION/DEAP_EEG/overlap4s_seg_conv_ch_BPF/2D_real"
test_data_dir = "/home/bcml1/2025_EMOTION/DEAP_EEG/overlap4s_seg_conv_ch_BPF/2D_real/test"

# 데이터 준비
train_raw_data = load_data(train_data_dir)
test_raw_data = load_data(test_data_dir)

train_pretext_dataset = PretextTaskDataset(train_raw_data, transformations)
test_pretext_dataset = PretextTaskDataset(test_raw_data, transformations)

train_data, train_labels = zip(*(train_pretext_dataset[i] for i in range(len(train_pretext_dataset))))
test_data, test_labels = zip(*(test_pretext_dataset[i] for i in range(len(test_pretext_dataset))))

train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)

# 모델 정의 및 학습
input_shape = train_data.shape[1:]  # (time, channel, band)
num_classes = len(transformations)

model = create_self_supervised_cnn(input_shape, num_classes)
model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_data, train_labels, epochs=100, batch_size=4, validation_split=0.2)

# 테스트 평가
test_loss, test_accuracy = model.evaluate(test_data, test_labels)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}")

# 감정 라벨 생성 및 저장
def generate_emotion_labels(data, model, save_dir):
    predictions = model.predict(data)
    emotion_labels = np.argmax(predictions, axis=1)  # 예측된 라벨

    # negative=0, positive=1, neutral=2로 매핑
    emotion_mapping = {0: "negative", 1: "positive", 2: "neutral"}
    mapped_labels = [emotion_mapping[label] for label in emotion_labels]

    # 저장 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, 'emotion_labels.npy')
    np.save(output_path, emotion_labels)

    # 매핑된 라벨 텍스트 파일 저장
    with open(os.path.join(save_dir, 'mapped_labels.txt'), 'w') as f:
        f.writelines([f"{label}\n" for label in mapped_labels])

    print(f"Emotion labels saved at: {output_path}")
    print(f"Mapped labels: {mapped_labels[:10]} (showing first 10 labels)")

result_dir = "./labels"
generate_emotion_labels(train_data, model, result_dir)
generate_emotion_labels(test_data, model, result_dir)
