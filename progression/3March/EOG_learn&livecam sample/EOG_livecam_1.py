import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque

WINDOW_SIZE = 100

emotion_map = {
    0: "Happy",
    1: "Sad",
    2: "Angry",
    3: "Neutral"
}

# mediapipe Face Mesh 
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

left_eye_indices = [33, 133, 160, 159, 158, 157, 173]
right_eye_indices = [362, 263, 387, 386, 385, 384, 398]

def extract_eye_features(frame):
    """
    Calculates the centre coordinates of the eyes in one frame using the mediapipe Face Mesh.
    Returns: [left_eye_x, left_eye_y, right_eye_x, right_eye_y]
    Returns None if face/eye detection fails.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    h, w, _ = frame.shape
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        left_coords = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in left_eye_indices])
        right_coords = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in right_eye_indices])
        left_center = left_coords.mean(axis=0)
        right_center = right_coords.mean(axis=0)
        return [left_center[0], left_center[1], right_center[0], right_center[1]]
    else:
        return None

def main():
    model = tf.keras.models.load_model('model.keras')
    
    signal_buffer = deque(maxlen=WINDOW_SIZE)
    
    cap = cv2.VideoCapture(0)  
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        features = extract_eye_features(frame)
        if features is not None:
            signal_buffer.append(features)
        
        if len(signal_buffer) == WINDOW_SIZE:
            input_seq = np.expand_dims(np.array(signal_buffer, dtype='float32'), axis=0)
            pred = model.predict(input_seq)
            pred_label = np.argmax(pred, axis=1)[0]
            emotion = emotion_map.get(pred_label, "Unknown")
            cv2.putText(frame, f'Emotion: {emotion}', (30, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Gathering data...', (30, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow('Real-time Emotion Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == "__main__":
    main()
