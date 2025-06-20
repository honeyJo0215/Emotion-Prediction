import tensorflow as tf
from tensorflow.keras import layers, Model
import os
import re
import numpy as np
import scipy.io as sio
from scipy.signal import medfilt, butter, filtfilt
from sklearn.preprocessing import MinMaxScaler

class EMCSP_EEG_1DCNN_Encoder(Model):
    def __init__(
        self,
        fs=200,
        bands=None,
        n_components=8,
        hidden_dim=128,
        apply_smoothing=False,
        window_len=200,
        n_channels=8,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.fs = fs
        self.bands = bands or {
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        self.n_components = n_components
        self.hidden_dim = hidden_dim
        self.apply_smoothing = apply_smoothing
        self.window_len = window_len
        self.n_channels = n_channels
        self.n_bands = len(self.bands)
        self.filters = {}

        # CNN branches expect (time, channels)
        self.chan_branch = self.build_chan_branch_1d((self.window_len, self.n_channels))
        self.freq_branch = self.build_freq_branch_1d((self.window_len, self.n_channels))
        # projection 레이어는 build()에서 초기화
        self.proj = None

    def build(self, input_shape):
        # 채널/주파수 브랜치 build
        self.chan_branch.build((None, self.window_len, self.n_channels))
        self.freq_branch.build((None, self.window_len, self.n_channels))

        # 브랜치 출력 차원 획득
        feat_dim = self.chan_branch.output_shape[-1]  # e.g. 64

        # projection 레이어 정의 및 build
        self.proj = layers.Dense(self.hidden_dim, activation='relu')
        # combined 텐서의 마지막 축 크기는 feat_dim*2
        self.proj.build((None, None, feat_dim * 2))

        super().build(input_shape)

    def build_chan_branch_1d(self, input_shape):
        inp = layers.Input(shape=input_shape)
        x = layers.Conv1D(64, 7, activation='relu', padding='same')(inp)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.GlobalAveragePooling1D()(x)
        return Model(inp, x)

    def build_freq_branch_1d(self, input_shape):
        inp = layers.Input(shape=input_shape)
        x = layers.Conv1D(64, 7, activation='relu', padding='same')(inp)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.GlobalAveragePooling1D()(x)
        return Model(inp, x)

    def bandpass_filter(self, data, lowcut, highcut):
        nyq = 0.5 * self.fs
        b, a = butter(4, [lowcut/nyq, highcut/nyq], btype='band')
        return filtfilt(b, a, data, axis=-1)

    def compute_average_covariance(self, trials):
        cov_sum = None
        for X in trials:
            C = X @ X.T
            C /= np.trace(C)
            cov_sum = C if cov_sum is None else cov_sum + C
        return cov_sum / len(trials)

    def compute_csp_filters(self, covA, covB):
        R = covA + covB
        eigvals, eigvecs = np.linalg.eigh(R)
        idx = np.argsort(eigvals)
        P = (eigvecs[:, idx] / np.sqrt(eigvals[idx])).T
        S = P @ covA @ P.T
        w_vals, w_vecs = np.linalg.eigh(S)
        order = np.argsort(w_vals)[::-1]
        W = w_vecs[:, order].T @ P
        n2 = self.n_components // 2
        return np.vstack([W[:n2], W[-n2:]])

    def load_and_preprocess_mat_file(self, file_path):
        mat_data = sio.loadmat(file_path)
        raw_dict = {k: v for k, v in mat_data.items() if not k.startswith('__')}
        scaler = MinMaxScaler()
        proc = {}
        for key, arr in raw_dict.items():
            X = arr.astype(float)
            if np.isnan(X).any():
                mu = np.nanmean(X, axis=0)
                inds = np.where(np.isnan(X))
                X[inds] = np.take(mu, inds[1])
            X = scaler.fit_transform(X)
            if self.apply_smoothing:
                X = medfilt(X, kernel_size=3)
            m = re.search(r"(\d+)$", key)
            if m:
                sess = int(m.group(1))
                proc[sess] = X
        return proc

    def load_raw_trials(self, folder_path, session_labels):
        data = {}
        for fn in sorted(os.listdir(folder_path)):
            if not fn.endswith('.mat'): continue
            subj = fn.split('_')[0]
            proc = self.load_and_preprocess_mat_file(os.path.join(folder_path, fn))
            data.setdefault(subj, {}).update(proc)

        trials = []
        for subj, sess_dict in data.items():
            for sess, full_X in sess_dict.items():
                label = session_labels[sess - 1]
                T_full = full_X.shape[1]
                for start in range(0, T_full - self.window_len + 1, self.window_len):
                    window = full_X[:, start:start + self.window_len]
                    trials.append({
                        'subject': subj,
                        'session': sess,
                        'data': window,
                        'label': label
                    })
        return trials

    def compute_filters_from_trials(self, trials):
        cov_sum = {}
        counts  = {}
        for tr in trials:
            subj, lbl = tr['subject'], tr['label']
            n_ch = tr['data'].shape[0]
            cov_sum.setdefault(subj, {}).setdefault(lbl, {})
            counts .setdefault(subj, {}).setdefault(lbl, {})
            for band_key in self.bands:
                cov_sum[subj][lbl][band_key] = np.zeros((n_ch, n_ch), dtype=np.float32)
                counts [subj][lbl][band_key] = 0

        for tr in trials:
            subj, lbl = tr['subject'], tr['label']
            X = tr['data'].astype(np.float32)
            for band_key, (lo, hi) in self.bands.items():
                Xf = self.bandpass_filter(X, lo, hi)
                C = Xf @ Xf.T
                C /= np.trace(C)
                cov_sum[subj][lbl][band_key] += C
                counts[subj][lbl][band_key] += 1

        self.filters = {}
        for subj, lbls in cov_sum.items():
            self.filters[subj] = {}
            labels = list(lbls.keys())
            for lbl in labels:
                self.filters[subj][lbl] = {}
                for b in self.bands:
                    covA = cov_sum[subj][lbl][b] / counts[subj][lbl][b]
                    covB = sum(cov_sum[subj][o][b] for o in labels if o != lbl) \
                           / sum(counts[subj][o][b] for o in labels if o != lbl)
                    W = self.compute_csp_filters(covA, covB)
                    self.filters[subj][lbl][b] = W

    def extract_features_from_trials(self, trials):
        X_out, y_out = [], []
        for tr in trials:
            subj, lbl = tr['subject'], tr['label']
            X = tr['data']
            band_feats = []
            for b, (lo, hi) in self.bands.items():
                filt = self.bandpass_filter(X, lo, hi)
                W = self.filters[subj][lbl][b]
                Y = W @ filt
                band_feats.append(Y.T)
            feat = np.stack(band_feats, axis=0)
            X_out.append(feat)
            y_out.append(lbl)
        return np.array(X_out), np.array(y_out)

    def call(self, x):
        B = tf.shape(x)[0]
        L = tf.shape(x)[1]
        flat = tf.reshape(x, [B * L, self.n_bands, self.window_len, self.n_channels])
        chan_in = tf.reshape(flat, [B * L * self.n_bands, self.window_len, self.n_channels])
        freq_in = tf.transpose(flat, perm=[0, 2, 3, 1])
        freq_in = tf.reshape(freq_in, [B * L * self.n_bands, self.window_len, self.n_channels])

        c_feat = self.chan_branch(chan_in)
        f_feat = self.freq_branch(freq_in)
        c_seq = tf.reshape(c_feat, [B, L, self.n_bands, -1])
        f_seq = tf.reshape(f_feat, [B, L, self.n_bands, -1])
        c_seq = tf.reduce_mean(c_seq, axis=2)
        f_seq = tf.reduce_mean(f_seq, axis=2)
        combined = tf.concat([c_seq, f_seq], axis=-1)
        return self.proj(combined)  

    def compute_output_shape(self, input_shape):
        batch, seq_len = input_shape[0], input_shape[1]
        return (batch, seq_len, self.hidden_dim)
