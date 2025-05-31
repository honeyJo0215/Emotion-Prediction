import os
import re
import numpy as np
import cv2
from EMCSP_1D_CNN import EEGDataLoader, EMCSP_EEG_1DCNN_Encoder


class VideoFrameExtractor:
    """
    Extracts and preprocesses frames from video files.

    Args:
        frame_size (tuple): Desired output frame size (width, height).
    """
    def __init__(self, frame_size=(224, 224)):
        self.frame_size = frame_size

    def extract(self, video_path):
        """
        Reads all frames from a .avi video, converts to RGB, resizes, and returns as numpy array.

        Args:
            video_path (str): Path to the .avi video file.

        Returns:
            np.ndarray: Array of shape [T, H, W, 3] with dtype uint8.
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize
            frame = cv2.resize(frame, self.frame_size)
            frames.append(frame)
        cap.release()
        return np.array(frames, dtype=np.uint8)


class MultiModalDataLoader:
    """
    Loads and aligns video frame data with EEG features for multimodal emotion recognition.

    Assumes EEG .npy files named like:
      folderX_subject{subject}_sample_{trial}_label{label}.npy
    and video files at:
      {video_root}/s{subject:02d}/s{subject:02d}_trial{trial:02d}.avi

    Skips any sample with label == 4.
    """
    def __init__(
        self,
        video_root,
        eeg_root,
        fs=200,
        bands=None,
        apply_smoothing=False,
        window_len=200,
        n_components=8,
        frame_size=(224, 224)
    ):
        self.video_root = video_root
        self.eeg_root = eeg_root
        self.frame_extractor = VideoFrameExtractor(frame_size=frame_size)
        self.eeg_loader = EEGDataLoader(
            fs=fs,
            bands=bands,
            apply_smoothing=apply_smoothing,
            window_len=window_len
        )
        self.encoder = None
        self.n_components = n_components

    def _parse_eeg_filename(self, fname):
        # filename like folder1_subject1_sample_01_label3.npy
        m = re.match(r".*subject(\d+)_sample_(\d+)_label(\d+)\.npy", fname)
        if not m:
            return None
        subj = int(m.group(1))
        trial = int(m.group(2))
        label = int(m.group(3))
        return subj, trial, label

    def _gather_trials(self):
        records = []
        for fname in sorted(os.listdir(self.eeg_root)):
            if not fname.endswith('.npy'):
                continue
            parsed = self._parse_eeg_filename(fname)
            if not parsed:
                continue
            subj, trial, label = parsed
            if label == 4:
                continue
            eeg_path = os.path.join(self.eeg_root, fname)
            video_path = os.path.join(
                self.video_root,
                f"s{subj:02d}",
                f"s{subj:02d}_trial{trial:02d}.avi"
            )
            if not os.path.exists(video_path):
                continue
            records.append({
                'subject': str(subj),
                'trial': trial,
                'label': label,
                'eeg_path': eeg_path,
                'video_path': video_path
            })
        return records

    def load(self):
        """
        Loads and preprocesses all paired EEG and video data.

        Returns:
            X_img (np.ndarray): List of video frame arrays [N, T, H, W, 3]
            X_seq (np.ndarray): EEG features [N, 1, n_bands, window_len, n_components]
            y     (np.ndarray): Labels [N]
        """
        trials = self._gather_trials()
        # Prepare EEG sessions for filter computation
        sessions = []
        for tr in trials:
            data = np.load(tr['eeg_path'])
            sessions.append({
                'subject': tr['subject'],
                'session': tr['trial'],
                'data': data,
                'label': tr['label']
            })
        # Compute CSP filters
        self.encoder = EMCSP_EEG_1DCNN_Encoder(
            fs=self.eeg_loader.fs,
            bands=self.eeg_loader.bands,
            apply_smoothing=self.eeg_loader.apply_smoothing,
            window_len=self.eeg_loader.window_len,
            n_components=self.n_components
        )
        self.encoder.compute_filters_from_trials(sessions)
        # Extract EEG features per trial
        X_seq, y = self.encoder.extract_features_from_trials(sessions)
        # Add sequence dim
        X_seq = X_seq[:, np.newaxis, ...]
        # Load video frames
        X_img = []
        for tr in trials:
            frames = self.frame_extractor.extract(tr['video_path'])
            X_img.append(frames)
        return np.array(X_img), X_seq, np.array(y)
