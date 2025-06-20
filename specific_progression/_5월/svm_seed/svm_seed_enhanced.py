import os, re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# gpu limit
def limit_gpu_memory(memory_limit_mib=5000):
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

limit_gpu_memory(5000)

# Classification
class BidirectionalSSMLayer(layers.Layer):
    def __init__(self, units, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.fwd = layers.Conv1D(1, kernel_size, padding='causal')
        self.bwd = layers.Conv1D(1, kernel_size, padding='causal')
    def call(self, x):
        seq = tf.expand_dims(x, -1)          # (batch, units, 1)
        fwd_out = self.fwd(seq)
        rev_in  = tf.reverse(seq, axis=[1])
        bwd_out = tf.reverse(self.bwd(rev_in), axis=[1])
        return tf.squeeze(fwd_out + bwd_out, -1)

class DiffuSSMLayer(layers.Layer):
    def __init__(self, model_dim, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.norm = layers.LayerNormalization()
        self.down = layers.Dense(hidden_dim, activation='swish')
        self.ssm  = BidirectionalSSMLayer(hidden_dim)
        self.up   = layers.Dense(model_dim)
        self.alpha = self.add_weight(name="alpha",
                                     shape=(), initializer='zeros', trainable=True)
    def call(self, x):
        h = self.norm(x)
        h = self.down(h)       # → (batch, hidden_dim)
        h = self.ssm(h)        # → (batch, hidden_dim)
        h = self.up(h)         # → (batch, model_dim)
        return x + self.alpha * h

def add_diffusion_noise(x, stddev=0.05):
    return x + tf.random.normal(tf.shape(x), mean=0.0, stddev=stddev)

# =============================================================================
# 2) Frequency / Channel Branch (임베딩 dim=64)
def build_freq_branch(input_shape=(200,8,4), noise_std=0.05, emb_dim=64):
    inp = layers.Input(shape=input_shape, name="freq_in")
    x   = layers.Lambda(lambda t: add_diffusion_noise(t, noise_std))(inp)
    x   = layers.Flatten()(x)                         
    x   = layers.Dense(emb_dim, activation='swish')(x)
    x   = DiffuSSMLayer(emb_dim, emb_dim)(x)
    feat= layers.Dense(emb_dim, name="freq_emb")(x)
    return models.Model(inp, feat, name="FreqBranch")

def build_chan_branch(input_shape=(200,4,8), noise_std=0.05, emb_dim=64):
    inp = layers.Input(shape=input_shape, name="chan_in")
    x   = layers.Lambda(lambda t: add_diffusion_noise(t, noise_std))(inp)
    x   = layers.Flatten()(x)
    x   = layers.Dense(emb_dim, activation='swish')(x)
    x   = DiffuSSMLayer(emb_dim, emb_dim)(x)
    feat= layers.Dense(emb_dim, name="chan_emb")(x)
    return models.Model(inp, feat, name="ChanBranch")

# load_csp
def load_csp_features(csp_dir):
    Xs, Ys, subs, fls = [], [], [], []
    for fn in os.listdir(csp_dir):
        if not fn.endswith(".npy"): continue
        arr = np.load(os.path.join(csp_dir, fn))
        if arr.ndim!=3 or arr.shape[0]!=4 or arr.shape[2]!=8: continue
        T = arr.shape[1]
        for i in range(T//200):
            win = arr[:, i*200:(i+1)*200, :]
            Xs.append(win)
            lm = re.search(r'label(\d+)', fn)
            sm = re.search(r'subject(\d+)', fn)
            Ys.append(int(lm.group(1)) if lm else -1)
            subs.append(sm.group(1) if sm else '0')
            fls.append(f"{fn}_win{i}")
    X = np.array(Xs); Y = np.array(Ys)
    mask = (Y>=0)
    return X[mask], Y[mask], np.array(subs)[mask], np.array(fls)[mask]

# joint classifier
def build_joint_classifier(freq_branch, chan_branch, num_classes=4):
    inp = layers.Input(shape=(4,200,8), name="CSP_Input")
    # freq
    f_perm = layers.Permute((2,3,1), name="perm_freq")(inp)
    f_emb  = freq_branch(f_perm)
    # chan
    c_perm = layers.Permute((2,1,3), name="perm_chan")(inp)
    c_emb  = chan_branch(c_perm)
    # combine & head
    comb   = layers.Concatenate(name="fusion")([f_emb, c_emb])
    x      = layers.Dense(128, activation='relu')(comb)
    x      = layers.Dropout(0.3)(x)
    out    = layers.Dense(num_classes, activation='softmax', name="out")(x)
    return models.Model(inp, out, name="JointClassifier")



if __name__ == "__main__":
    CSP_DIR    = "/home/bcml1/2025_EMOTION/DEAP_eeg_new_label_CSP"
    RESULT_DIR = "/home/bcml1/sigenv/_5월/svm_seed/svm_deap_tuned"
    os.makedirs(RESULT_DIR, exist_ok=True)

    # data load
    X, Y, subjects, files = load_csp_features(CSP_DIR)

    # Branches
    freq_branch = build_freq_branch()
    chan_branch = build_chan_branch()
    freq_branch.trainable = False
    chan_branch.trainable = False

    for subj in sorted(set(subjects)):
        if not subj.isdigit(): 
            continue
        print(f"\n=== Subject {subj} ===")
        mask   = (subjects == subj)
        Xs, Ys, Fs = X[mask], Y[mask], files[mask]
        ufs    = np.unique(Fs)
        if len(ufs) < 3:
            print("  샘플 부족, 스킵"); continue

        # train/val/test split 
        tr_f, te_f = train_test_split(ufs, test_size=0.3, random_state=42)
        tr_f, va_f = train_test_split(tr_f, test_size=0.25, random_state=42)
        idx_tr = np.isin(Fs, tr_f); idx_va = np.isin(Fs, va_f); idx_te = np.isin(Fs, te_f)
        Xtr, Ytr = Xs[idx_tr], Ys[idx_tr]
        Xva, Yva = Xs[idx_va], Ys[idx_va]
        Xte, Yte = Xs[idx_te], Ys[idx_te]
        print(f"  ▶ Train:{len(Ytr)}  Val:{len(Yva)}  Test:{len(Yte)}")

        Ftr = freq_branch.predict(np.transpose(Xtr,(0,2,3,1)), verbose=0)
        Ctr = chan_branch.predict(np.transpose(Xtr,(0,2,1,3)), verbose=0)
        Fte = freq_branch.predict(np.transpose(Xte,(0,2,3,1)), verbose=0)
        Cte = chan_branch.predict(np.transpose(Xte,(0,2,1,3)), verbose=0)

        # pipeline
        param_grid = {
            'svc__C':    [0.1, 1, 10],
            'svc__gamma': ['scale', 0.01, 0.001],
            'svc__kernel': ['rbf']
        }
        freq_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('pca',    PCA(n_components=0.95, svd_solver='full')),
            ('svc',    SVC(probability=True, class_weight='balanced'))
        ])
        chan_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('pca',    PCA(n_components=0.95, svd_solver='full')),
            ('svc',    SVC(probability=True, class_weight='balanced'))
        ])

        grid_freq = GridSearchCV(freq_pipe, param_grid, cv=3, n_jobs=-1, verbose=1)
        grid_chan = GridSearchCV(chan_pipe, param_grid, cv=3, n_jobs=-1, verbose=1)

        # learn
        print("  ▶ Tuning & Training freq-SVM")
        grid_freq.fit(Ftr, Ytr)
        print("    best params:", grid_freq.best_params_)

        print("  ▶ Tuning & Training chan-SVM")
        grid_chan.fit(Ctr, Ytr)
        print("    best params:", grid_chan.best_params_)

        best_freq = grid_freq.best_estimator_
        best_chan = grid_chan.best_estimator_

        pf = best_freq.predict_proba(Fte)
        pc = best_chan.predict_proba(Cte)
        pa = (pf + pc) / 2
        y_pred = np.argmax(pa, axis=1)

        # Confusion Matrix
        cm = confusion_matrix(Yte, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Subject {subj} CM")
        cm_path = os.path.join(RESULT_DIR, f"s{subj}", "confusion_matrix.png")
        os.makedirs(os.path.dirname(cm_path), exist_ok=True)
        plt.savefig(cm_path); plt.close()
        print("  ▶ Confusion matrix saved to", cm_path)

        # Classification Report
        report = classification_report(Yte, y_pred)
        rpt_path = os.path.join(RESULT_DIR, f"s{subj}", "classification_report.txt")
        with open(rpt_path, "w") as f:
            f.write(report)
        print("  ▶ Classification report saved to", rpt_path)

        # model save
        joblib.dump(best_freq, os.path.join(RESULT_DIR, f"s{subj}", "svm_freq_tuned.joblib"))
        joblib.dump(best_chan, os.path.join(RESULT_DIR, f"s{subj}", "svm_chan_tuned.joblib"))
        print(f"  ▶ Saved tuned SVMs for subject {subj}")
