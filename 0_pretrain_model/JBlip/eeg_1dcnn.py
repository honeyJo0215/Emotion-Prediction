"""
실험용코드
본래의 Blip 과는 무관합니다.
"""


import tensorflow as tf
from tensorflow.keras import layers, Model

class Sequence1DEncoder(Model):
    def __init__(self, in_channels=1, hidden_size=768, num_layers=4, kernel_size=3):
        super().__init__()
        self.convs = []
        for i in range(num_layers):
            out_ch = hidden_size if i==num_layers-1 else hidden_size//2
            self.convs.append(layers.Conv1D(out_ch, kernel_size, padding='same', activation='relu'))
            self.convs.append(layers.BatchNormalization())
            # optionally layers.MaxPool1D() 등 추가
        self.proj = layers.Dense(hidden_size)  # BERT hidden_size 와 맞추기

    def call(self, x):
        # x: [batch, seq_len, in_channels]
        h = x
        for layer in self.convs:
            h = layer(h)
        # h: [batch, seq_len, hidden_size//2] → project to hidden_size
        h = self.proj(h)            # [batch, seq_len, hidden_size]
        return h                    # last_hidden_state 유사


class Sequence1DDecoder(Model):
    """
    1D convolutional decoder that mirrors Sequence1DEncoder,
    reconstructing raw 1D signals (e.g., EEG, rPPG) from hidden embeddings.

    Args:
        hidden_size (int): dimension of encoder embeddings (H).
        out_channels (int): number of output channels (e.g., 1 for mono EEG).
        num_layers (int): number of Conv1D layers.
        kernel_size (int): convolution kernel size.
    """
    def __init__(
        self,
        hidden_size: int = 768,
        out_channels: int = 1,
        num_layers: int = 4,
        kernel_size: int = 3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.convs = []
        # First layer: map hidden_size -> hidden_size
        self.convs.append(layers.Conv1D(
            filters=hidden_size,
            kernel_size=kernel_size,
            padding='same',
            activation='relu'
        ))
        self.convs.append(layers.BatchNormalization())
        # Middle layers: keep hidden_size
        for _ in range(num_layers - 2):
            self.convs.append(layers.Conv1D(
                filters=hidden_size,
                kernel_size=kernel_size,
                padding='same',
                activation='relu'
            ))
            self.convs.append(layers.BatchNormalization())
        # Final layer: hidden_size -> out_channels
        self.convs.append(layers.Conv1D(
            filters=out_channels,
            kernel_size=1,
            padding='same',
            activation=None
        ))

    def call(self, embeddings: tf.Tensor) -> tf.Tensor:
        """
        Args:
            embeddings: Tensor of shape (batch_size, seq_len, hidden_size)

        Returns:
            Reconstructed 1D signal of shape (batch_size, seq_len, out_channels)
        """
        x = embeddings
        for layer in self.convs:
            x = layer(x)
        return x
