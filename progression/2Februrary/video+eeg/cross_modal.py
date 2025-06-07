import tensorflow as tf
from tensorflow.keras import layers, models

class CrossModalTransformer(tf.keras.layers.Layer):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(CrossModalTransformer, self).__init__()
        
        # Multi-head attention layers for both modalities (EEG and Eye-tracking video data)
        self.eeg_to_eye_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_dim)
        self.eye_to_eeg_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_dim)
        
        # Feedforward network
        self.ffn = models.Sequential([
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dense(input_dim)
        ])
        
        # Layer normalization
        self.layer_norm = layers.LayerNormalization()

    def call(self, eeg_input, eye_input):
        # EEG -> Eye-tracking cross-modal attention
        eeg_to_eye = self.eeg_to_eye_attention(query=eeg_input, value=eye_input, key=eye_input)
        
        # Eye-tracking -> EEG cross-modal attention
        eye_to_eeg = self.eye_to_eeg_attention(query=eye_input, value=eeg_input, key=eeg_input)
        
        # Combine both modalities
        combined = tf.concat([eeg_to_eye, eye_to_eeg], axis=-1)
        
        # Feedforward processing
        combined = self.ffn(combined)
        
        # Apply Layer Normalization
        combined = self.layer_norm(combined)
        
        return combined


class MultiModalTransformer(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(MultiModalTransformer, self).__init__()
        
        # Cross-modal transformer layers
        self.cross_modal_transformer = CrossModalTransformer(input_dim, hidden_dim, num_heads)
        
        # Final linear layer for output
        self.output_layer = layers.Dense(1)  # Assuming binary classification

    def call(self, eeg_input, eye_input):
        # Pass EEG and Eye inputs through the cross-modal transformer
        combined_features = self.cross_modal_transformer(eeg_input, eye_input)
        
        # Apply the final linear layer
        output = self.output_layer(combined_features)
        
        return output

# Example usage:
input_dim = 128  # Example input dimension
hidden_dim = 256
num_heads = 8

# Create model
model = MultiModalTransformer(input_dim, hidden_dim, num_heads)

# Example inputs for EEG and Eye-tracking video data
eeg_input = tf.random.normal((32, 10, input_dim))  # (batch_size, sequence_length, input_dim)
eye_input = tf.random.normal((32, 10, input_dim))  # Eye-tracking data (e.g., extracted features from video frames)

# Forward pass
output = model(eeg_input, eye_input)
print(output)
