import os
import re
import numpy as np
import scipy.io as sio
from scipy.signal import medfilt, butter, filtfilt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Model

class EMCSP_EEG_1DCNN_Encoder(Model):
    def __init__(self,
                 fs=200,
                 bands=None,
                 n_components=8,
                 hidden_dim=128,
                 apply_smoothing=False,
                 window_len=200,
                 n_channels=8,
                 ):
        super().__init__()
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

        # CNN branches
        self.chan_branch = self.build_chan_branch_1d((self.window_len, self.n_channels))
        self.freq_branch = self.build_freq_branch_1d((self.window_len, self.n_channels))
        self.proj = None

    def build(self, input_shape):
        self.chan_branch.build((None, self.window_len, self.n_channels))
        self.freq_branch.build((None, self.window_len, self.n_channels))
        feat_dim = self.chan_branch.output_shape[-1]
        self.proj = layers.Dense(self.hidden_dim, activation='relu')
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

    def bandpass_filter(self, data, lowcut, highcut):
        nyq = 0.5 * self.fs
        b, a = butter(4, [lowcut/nyq, highcut/nyq], btype='band')
        return filtfilt(b, a, data, axis=-1)

    def compute_filters_from_trials(self, trials):
        # 각 subject, label 별로 공분산 합과 개수 집계
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

        # band-pass → 공분산 계산 누적
        for tr in trials:
            subj, lbl = tr['subject'], tr['label']
            X = tr['data'].astype(np.float32)
            for band_key, (lo, hi) in self.bands.items():
                Xf = self.bandpass_filter(X, lo, hi)
                C = Xf @ Xf.T
                C /= np.trace(C)
                cov_sum[subj][lbl][band_key] += C
                counts[subj][lbl][band_key] += 1

        # CSP 필터 계산
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
                    # 원래 compute_csp_filters 메서드 사용
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
        # 1) 동적으로 shape 뽑아오기
        shape = tf.shape(x)
        B, L, nb, wl, ch = shape[0], shape[1], shape[2], shape[3], shape[4]

        # 2) flat 형태로 재배열
        #    -> (B*L, nb, wl, ch)
        flat = tf.reshape(x, [B * L, nb, wl, ch])

        # 3) 채널 브랜치 입력: (B*L*nb, wl, ch)
        chan_in = tf.reshape(flat, [B * L * nb, wl, ch])

        # 4) 주파수 브랜치 입력: (B*L*nb, wl, ch)
        freq = tf.transpose(flat, perm=[0, 2, 3, 1])  # (B*L, wl, ch, nb)
        freq_in = tf.reshape(freq, [B * L * nb, wl, ch])

        # 5) 브랜치별 특징 추출
        c_feat = self.chan_branch(chan_in)  # (B*L*nb, feat_dim)
        f_feat = self.freq_branch(freq_in)  # (B*L*nb, feat_dim)

        # 6) 다시 시퀀스 형태로 복원
        feat_dim = tf.shape(c_feat)[-1]
        c_seq = tf.reshape(c_feat, [B, L, nb, feat_dim])
        f_seq = tf.reshape(f_feat, [B, L, nb, feat_dim])

        # 7) 밴드 축 평균
        c_seq = tf.reduce_mean(c_seq, axis=2)  # (B, L, feat_dim)
        f_seq = tf.reduce_mean(f_seq, axis=2)  # (B, L, feat_dim)

        # 8) concat & projection
        combined = tf.concat([c_seq, f_seq], axis=-1)  # (B, L, feat_dim*2)
        return self.proj(combined)                     # (B, L, hidden_dim)


class EEGDataLoader:
    def __init__(self, fs=200, bands=None, apply_smoothing=False, window_len=200):
        self.fs = fs
        self.bands = bands or {
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        self.apply_smoothing = apply_smoothing
        self.window_len = window_len

    def load_and_preprocess_mat_file(self, file_path):
        mat = sio.loadmat(file_path)
        raw = {k: v for k, v in mat.items() if not k.startswith('__')}
        scaler = MinMaxScaler()
        proc = {}
        for key, arr in raw.items():
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

    def load_raw_sessions(self, folder_path, session_labels):
        """
        .mat 파일 하나당 subject/session 레벨의 데이터를 반환합니다.
        [{'subject': subj, 'session': sess, 'data': full_X, 'label': lbl}, ...]
        """
        sessions = []
        for fn in sorted(os.listdir(folder_path)):
            if not fn.endswith('.mat'): continue
            subj = fn.split('_')[0]
            proc = self.load_and_preprocess_mat_file(os.path.join(folder_path, fn))
            for sess, full_X in proc.items():
                lbl = session_labels[sess - 1]
                sessions.append({
                    'subject': subj,
                    'session': sess,
                    'data': full_X,   # shape: [n_channels, T]
                    'label': lbl
                })
        return sessions

    def split_trials(self,
                     trials,
                     mode='intra',
                     test_size=0.2,
                     random_state=42,
                     leave_out_subject=None):
        """
        mode='intra': 각 subject 별로 세션 단위 split
        mode='inter': leave‐one‐subject‐out
        """
        if mode == 'intra':
            train, test = [], []
            subjects = sorted({t['subject'] for t in trials})
            for subj in subjects:
                subj_trials = [t for t in trials if t['subject'] == subj]
                # 1) 세션 번호만 유니크하게 뽑기
                sess_ids = sorted({t['session'] for t in subj_trials})
                # 2) 세션 번호 단위로 split
                tr_sess, te_sess = train_test_split(
                    sess_ids,
                    test_size=test_size,
                    random_state=random_state
                )
                # 3) 세션 번호에 해당하는 trial 객체 모으기
                train += [t for t in subj_trials if t['session'] in tr_sess]
                test  += [t for t in subj_trials if t['session'] in te_sess]
            return train, test

        elif mode == 'inter':
            if leave_out_subject is None:
                raise ValueError("leave_out_subject를 지정해주세요.")
            train = [t for t in trials if t['subject'] != leave_out_subject]
            test  = [t for t in trials if t['subject'] == leave_out_subject]
            return train, test

        else:
            raise ValueError("mode는 'intra' 또는 'inter'만 지원합니다.")

    def segment_sessions(self, sessions):
        """
        세션 단위 데이터를 window_len 크기로 분할하여
        [{'subject','session','data':window,'label'}, ...] 형태로 반환합니다.
        """
        trials = []
        for sess in sessions:
            X_full = sess['data']
            T = X_full.shape[1]
            for start in range(0, T - self.window_len + 1, self.window_len):
                window = X_full[:, start:start+self.window_len]
                trials.append({
                    'subject': sess['subject'],
                    'session': sess['session'],
                    'data': window,
                    'label': sess['label']
                })
        return trials
