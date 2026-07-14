"""
classify_emg_svm.py
--------------------
Support Vector Machine classifier for labeled EMG recordings.
Mirrors the structure of classify_emg_lda.py — same data loading,
feature extraction, and preprocessing pipeline; SVM replaces LDA.

DATA FORMAT EXPECTED
    Each .txt file in DATA_DIR contains one recording, one class.
    Single-channel: <timestamp_seconds>,<ADC_value>
    Multi-channel:  <timestamp_seconds>,<CH1>,<CH2>,<CH3>

USAGE
    # Train/test split (same data as LDA runs):
    python classify_emg_svm.py \
        --train-dir data/act1 data/act2 \
        --test-dir  data/act1_test data/act2_test \
        --folder-labels \
        --legacy-c-compatible \
        --legacy-wl-samples 100 --legacy-winc-samples 50 \
        --legacy-deadzone-zc 0.025 --legacy-deadzone-turn 0.015 \
        --legacy-scale-mav 2 --legacy-scale-zc 15 \
        --kernel rbf --C 1.0 --gamma scale

    # Add PCA before SVM:
        --pca --pca-n 6

    # Export linear SVM weights to STM32 header:
        --kernel linear --export-model

KERNELS
    linear  — linear decision boundary; weights can be exported to STM32
    rbf     — radial basis function (default; usually best accuracy)
    poly    — polynomial kernel
"""

import argparse
import os
import glob
import sys
import time
import numpy as np
from typing import List, Tuple

import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data'))

SAMPLE_RATE = 1000
WINDOW_MS   = 200
OVERLAP_PCT = 0.5

BP_LOW   = 20.0
BP_HIGH  = 450.0
BP_ORDER = 4

ZC_THRESHOLD = 10
CV_FOLDS     = 5

LEGACY_DEADZONE_ZC   = 0.025
LEGACY_DEADZONE_TURN = 0.015
LEGACY_SCALE_MAV     = 2.0
LEGACY_SCALE_ZC      = 15.0
LEGACY_WL_SAMPLES    = 100
LEGACY_WINC_SAMPLES  = 50
# ──────────────────────────────────────────────────────────────────────────────


# ── SIGNAL PROCESSING ─────────────────────────────────────────────────────────

def bandpass_filter(signal: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    nyq   = fs / 2.0
    low_n = max(1e-4, min(low  / nyq, 0.9999))
    high_n= max(1e-4, min(high / nyq, 0.9999))
    if low_n >= high_n:
        return signal
    b, a = butter(order, [low_n, high_n], btype='band')
    if signal.ndim == 1:
        return filtfilt(b, a, signal)
    filtered = np.zeros_like(signal)
    for ch in range(signal.shape[1]):
        filtered[:, ch] = filtfilt(b, a, signal[:, ch])
    return filtered


def extract_features(window: np.ndarray) -> np.ndarray:
    if window.ndim == 1:
        mav  = np.mean(np.abs(window))
        wl   = np.sum(np.abs(np.diff(window)))
        zc   = np.sum((window[:-1] * window[1:] < 0) &
                      (np.abs(window[1:] - window[:-1]) >= ZC_THRESHOLD))
        slope = np.diff(window)
        ssc  = np.sum((slope[:-1] * slope[1:] < 0))
        return np.array([mav, float(zc), wl, float(ssc)])
    all_feats = []
    for ch in range(window.shape[1]):
        x    = window[:, ch]
        mav  = np.mean(np.abs(x))
        wl   = np.sum(np.abs(np.diff(x)))
        zc   = np.sum((x[:-1] * x[1:] < 0) &
                      (np.abs(x[1:] - x[:-1]) >= ZC_THRESHOLD))
        slope = np.diff(x)
        ssc  = np.sum((slope[:-1] * slope[1:] < 0))
        all_feats.extend([mav, float(zc), wl, float(ssc)])
    return np.array(all_feats)


def extract_features_legacy_c(window: np.ndarray) -> np.ndarray:
    if window.ndim == 1:
        window = window[:, None]
    win_length = window.shape[0]
    n_ch = window.shape[1]
    ruler    = 1.0 / float(win_length)
    ruler_sq = ruler * ruler
    lscale   = float(win_length) / 40.0
    tscale   = (float(win_length) / 40.0) * 10.0
    w = window - np.mean(window, axis=0, keepdims=True)
    feats = []
    for ch in range(n_ch):
        x  = w[:, ch]
        mav = np.mean(np.abs(x))
        zero_count = 0.0
        turns  = 0.0
        length = 0.0
        flag2  = 1
        for i in range(1, win_length - 1):
            fst = abs(x[i - 1])
            mid = abs(x[i])
            lst = abs(x[i + 1])
            if ((x[i] >= 0 and x[i-1] >= 0) or (x[i] <= 0 and x[i-1] <= 0)):
                flag1 = flag2
            else:
                if (mid < LEGACY_DEADZONE_ZC) and (fst < LEGACY_DEADZONE_ZC):
                    flag1 = flag2
                else:
                    flag1 = -flag2
            if flag1 != flag2:
                zero_count += 1.0
            flag2 = flag1
            if ((mid > fst and mid > lst) or (mid < fst and mid < lst)):
                if ((abs(mid) - abs(fst)) > LEGACY_DEADZONE_TURN) or \
                   ((abs(mid) - abs(lst)) > LEGACY_DEADZONE_TURN):
                    turns += 1.0
            d = (fst - mid) / 20.0
            length += float(np.sqrt(d * d + ruler_sq))
        zero_count = (zero_count / LEGACY_SCALE_ZC) * 40.0 / float(win_length)
        mav    = mav / LEGACY_SCALE_MAV
        length = (length - 1.0) / lscale
        turns  = turns / tscale
        feats.extend([mav, length, zero_count, turns])
    return np.array(feats, dtype=float)


def segment_and_extract(values: np.ndarray, fs: float, window_ms: int, overlap: float) -> np.ndarray:
    win_samples  = int(fs * window_ms / 1000)
    step_samples = int(win_samples * (1.0 - overlap))
    n_ch = 1 if values.ndim == 1 else values.shape[1]
    if win_samples < 10 or len(values) < win_samples:
        return np.empty((0, 4 * n_ch))
    features, start = [], 0
    while start + win_samples <= len(values):
        features.append(extract_features(values[start : start + win_samples]))
        start += step_samples
    return np.array(features) if features else np.empty((0, 4 * n_ch))


def segment_and_extract_legacy_c(values: np.ndarray, fs: float, window_ms: int, overlap: float) -> np.ndarray:
    win_samples  = int(LEGACY_WL_SAMPLES)
    step_samples = int(LEGACY_WINC_SAMPLES)
    n_ch = 1 if values.ndim == 1 else values.shape[1]
    if win_samples < 10 or len(values) < win_samples:
        return np.empty((0, 4 * n_ch))
    feats, start = [], 0
    while start + win_samples <= len(values):
        feats.append(extract_features_legacy_c(values[start : start + win_samples]))
        start += step_samples
    return np.array(feats) if feats else np.empty((0, 4 * n_ch))


def mapstd_fit_transform(X: np.ndarray):
    mean = np.mean(X, axis=0)
    std  = np.std(X, axis=0, ddof=1)
    std  = np.where(std == 0, 1.0, std)
    return (X - mean) / std, mean, std


def mapstd_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std


# ── FEATURE NAMES ─────────────────────────────────────────────────────────────

def infer_feature_names(n: int):
    base = ['MAV', 'ZC', 'WL', 'SSC']
    if n <= len(base):
        return base[:n]
    if n % 4 == 0:
        return [f"{b}_ch{ch+1}" for ch in range(n // 4) for b in base]
    return [f"F{i+1}" for i in range(n)]


def infer_feature_names_legacy(n: int):
    base = ['MAV', 'LEN', 'ZC', 'TURNS']
    if n <= len(base):
        return base[:n]
    if n % 4 == 0:
        return [f"{b}_ch{ch+1}" for ch in range(n // 4) for b in base]
    return [f"F{i+1}" for i in range(n)]


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_file(path):
    values = []
    with open(path, 'r') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if line_idx == 0 and ('timestamp' in line.lower() or 'ch' in line.lower()):
                continue
            parts = line.split('\t') if '\t' in line else line.split(',')
            try:
                channels = [float(p.strip()) for p in parts[1:] if p.strip()]
                if channels:
                    values.append(channels)
            except ValueError:
                continue
    if not values:
        return np.empty((0, 1), dtype=float)
    return np.array(values, dtype=float)


def resolve_data_dirs(data_dirs):
    if data_dirs is None:
        return []
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]
    resolved = []
    for path in data_dirs:
        if not os.path.isabs(path):
            script_rel = os.path.abspath(os.path.join(SCRIPT_DIR, path))
            path = script_rel if os.path.isdir(script_rel) else \
                   os.path.abspath(os.path.join(SCRIPT_DIR, '..', path))
        resolved.append(path)
    return resolved


def load_dataset(data_dirs, prefix_labels=False, folder_labels=False, legacy_c_compatible=False):
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]
    data_dirs = resolve_data_dirs(data_dirs)
    txt_files = []
    for data_dir in data_dirs:
        paths = sorted(glob.glob(os.path.join(data_dir, '*.txt')))
        if not paths:
            raise FileNotFoundError(f"No .txt files found in:\n  {data_dir}")
        txt_files.extend([(data_dir, p) for p in paths])

    all_features, all_labels = [], []
    print(f"\nLoading files from: {', '.join(data_dirs)}")
    print(f"{'File':<30}  {'Samples':>8}  {'Windows':>8}  Label")
    print("-" * 65)

    for data_dir, path in txt_files:
        base_label = os.path.splitext(os.path.basename(path))[0]
        if folder_labels:
            folder_name = os.path.basename(os.path.normpath(data_dir))
            label = folder_name.replace('_test', '').replace('_train', '')
        else:
            label = base_label
            if prefix_labels and len(data_dirs) > 1:
                label = f"{os.path.basename(os.path.normpath(data_dir))}_{base_label}"

        raw = load_file(path)
        if raw.size == 0:
            print(f"  ⚠  {label}: empty file, skipping.")
            continue

        if legacy_c_compatible:
            feats = segment_and_extract_legacy_c(raw, SAMPLE_RATE, WINDOW_MS, OVERLAP_PCT)
        else:
            filtered = bandpass_filter(raw, SAMPLE_RATE, BP_LOW, BP_HIGH, BP_ORDER)
            centered = filtered - (np.mean(filtered) if filtered.ndim == 1
                                   else np.mean(filtered, axis=0, keepdims=True))
            feats = segment_and_extract(centered, SAMPLE_RATE, WINDOW_MS, OVERLAP_PCT)

        if feats.shape[0] == 0:
            print(f"  ⚠  {label}: too short to segment, skipping.")
            continue

        print(f"  {label:<28}  {raw.shape[0]:>8,}  {feats.shape[0]:>8,}  ✓")
        all_features.append(feats)
        all_labels.extend([label] * feats.shape[0])

    if not all_features:
        raise ValueError("No usable data found after loading all files.")
    return np.vstack(all_features), np.array(all_labels)


# ── PLOTTING ──────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, label_names, eval_label='Evaluation'):
    cm  = confusion_matrix(y_true, y_pred, labels=label_names)
    acc = accuracy_score(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(5, len(label_names) * 1.4),
                                    max(4, len(label_names) * 1.2)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(ax=ax, colorbar=False, cmap='Oranges')
    ax.set_title(
        f'SVM Confusion Matrix  —  {eval_label}\n'
        f'Overall accuracy: {acc * 100:.1f}%',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()


def plot_pca_projection(X_train, y_train, pca_2d, X_test=None, y_test=None, classes=None):
    """Project data onto first 2 PCA axes for visualization (SVM has no built-in transform)."""
    Xt = pca_2d.transform(X_train)
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls in enumerate(classes):
        mask = y_train == cls
        ax.scatter(Xt[mask, 0], Xt[mask, 1],
                   label=f'{cls} (train)', alpha=0.45, s=18,
                   color=colors[i], marker='o')
    if X_test is not None and y_test is not None:
        Xte = pca_2d.transform(X_test)
        for i, cls in enumerate(classes):
            mask = y_test == cls
            if mask.sum() == 0:
                continue
            ax.scatter(Xte[mask, 0], Xte[mask, 1],
                       label=f'{cls} (test)', alpha=0.8, s=22,
                       color=colors[i], marker='x')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    title = 'PCA Projection (SVM)'
    if X_test is not None:
        title += ' — train vs test'
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(title='Class', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()


# ── MODEL EXPORT (linear SVM only) ────────────────────────────────────────────

def export_svm_model(svm, scaler, le, pca=None,
                     train_accuracy=None, test_accuracy=None,
                     feature_names=None, n_features_raw=12):
    """
    Export a LINEAR SVM to an STM32-compatible C header.
    The decision function is:  sign( w · x_scaled + b )
    After fusing scaler (and optional PCA) into the weights, the embedded
    code only needs raw features + pre-computed xmean/xstd arrays.
    """
    from datetime import datetime

    if svm.kernel != 'linear':
        print("\n⚠  Model export is only supported for --kernel linear.")
        print("   (RBF/poly SVMs require the full support-vector set at runtime.)")
        return

    # SVM coef_ shape: (n_classes*(n_classes-1)/2, n_features_in_svm_space)
    # For binary: (1, n_features)
    svm_w = np.ravel(svm.coef_[0])        # weights in (PCA or scaled) space
    svm_b = float(np.ravel(svm.intercept_)[0])

    # Back-project through PCA if used
    if pca is not None:
        raw_vec = pca.components_.T @ svm_w
        pca_note = (f"// PCA Explained Variance: "
                    f"{np.sum(pca.explained_variance_ratio_)*100:.1f}% "
                    f"({pca.n_components_} components)\n")
        mode_str = "svm_linear_pca"
    else:
        raw_vec  = svm_w
        pca_note = "// PCA: disabled\n"
        mode_str = "svm_linear"

    # Absorb StandardScaler: decision = dot(raw_vec, (x-mean)/std) + b
    raw_w = raw_vec / scaler.scale_
    raw_b = svm_b - np.dot(scaler.mean_ / scaler.scale_, raw_vec)

    # Mirror LDA export format: two columns [+w, -w] so the same STM32 code works
    Wg_scaled = np.column_stack([raw_w, -raw_w])
    Cg        = np.array([raw_b, -raw_b])

    classes = le.classes_
    n_cls   = len(classes)
    feat_dim = len(raw_w)

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    header  = "// " + "=" * 60 + "\n"
    header += "// Auto-generated SVM model for STM32\n"
    header += f"// mode: {mode_str}\n"
    header += f"// kernel: linear\n"
    header += f"// generated: {ts}\n"
    header += f"// num_class: {n_cls}\n"
    header += f"// feature_dim: {feat_dim}\n"
    if feature_names:
        header += f"// features: {', '.join(feature_names[:n_features_raw])}\n"
    header += f"// classes: {', '.join(f'{c} ({i})' for i, c in enumerate(classes))}\n"
    if train_accuracy is not None:
        header += f"// train_accuracy: {train_accuracy:.2%}\n"
    if test_accuracy is not None:
        header += f"// test_accuracy: {test_accuracy:.2%}\n"
    header += "// " + "=" * 60 + "\n\n"

    def fmt_array(name, arr, cols=2):
        lines = [f"float {name}[{len(arr)}] = {{"]
        for i, v in enumerate(arr):
            sep = "," if i < len(arr) - 1 else ""
            lines.append(f"  {v:>16.6f}{sep}")
        lines.append("};")
        return "\n".join(lines)

    body  = fmt_array("Wg_init", Wg_scaled.flatten(order='F')) + "\n\n"
    body += fmt_array("Cg_init", Cg) + "\n\n"
    body += fmt_array("xstd_init",  scaler.scale_) + "\n\n"
    body += fmt_array("xmean_init", scaler.mean_) + "\n\n"
    body += pca_note

    full = header + body

    project_root = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    out_h   = os.path.join(project_root, 'EMG_SVM_Model_STM32.h')
    out_txt = os.path.join(project_root, 'EMG_SVM_Model_STM32_commented.txt')

    with open(out_h, 'w') as f:
        f.write(full)

    commented_lines = []
    for line in full.splitlines():
        if line.startswith('//') or line.strip() == '':
            commented_lines.append(line)
        else:
            commented_lines.append(f"// {line}")
    with open(out_txt, 'w') as f:
        f.write("\n".join(commented_lines) + "\n")

    print(full)
    print(f"\n✓ SVM model exported to: {out_h}")
    print(f"✓ Commented export written to: {out_txt}")


def print_svm_model_stats(svm: SVC, X_eval: np.ndarray = None):
    """Print model size, rough FLOP estimate, and host runtime for one sample."""
    # Parameter count + rough operation estimate per forward pass
    if svm.kernel == 'linear':
        n_feat = int(svm.coef_.shape[1])
        n_params = int(svm.coef_.size + svm.intercept_.size)
        approx_flops = int(2 * n_feat)  # ~n mult + n add
        extra_note = None
    else:
        n_feat = int(svm.support_vectors_.shape[1])
        n_sv = int(svm.support_vectors_.shape[0])
        n_params = int(svm.support_vectors_.size + svm.dual_coef_.size + svm.intercept_.size)
        approx_flops = int(n_sv * (3 * n_feat + 4))  # arithmetic-only rough estimate
        extra_note = f"  Kernel exp() calls/sample (approx): {n_sv}"

    param_mem_mb = (n_params * 4) / (1024.0 * 1024.0)  # float32-style estimate

    print("\nModel size (SVM classifier)")
    print(f"  Total parameters: {n_params:,} ({n_params})")
    print(f"  Trainable parameters: {n_params:,} ({n_params})")
    print(f"  Approx. parameter memory: {param_mem_mb:.6f} MB")
    print(f"  Approx. FLOPs per single-window forward pass: {approx_flops:,} ({approx_flops})")
    if extra_note is not None:
        print(extra_note)

    if X_eval is None or X_eval.size == 0:
        return

    # Runtime benchmark (host): single-sample predict
    x1 = X_eval[:1]
    for _ in range(200):
        svm.predict(x1)
    n_iter = 3000
    t0 = time.perf_counter()
    for _ in range(n_iter):
        svm.predict(x1)
    t1 = time.perf_counter()
    us_per_sample = (t1 - t0) * 1e6 / float(n_iter)

    print("  Runtime (host, estimated):")
    print(f"    Classifier only: {us_per_sample:.4f} us/sample")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train an SVM classifier on labeled EMG data folders.')

    # ── Data args (mirrors classify_emg_lda.py) ──
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--data-dir',  nargs='+',
        help='Directories for combined training + CV.')
    group.add_argument('--train-dir', nargs='+',
        help='Directories for training only.')
    parser.add_argument('--test-dir', nargs='+',
        help='Directories for testing (requires --train-dir).')
    parser.add_argument('--prefix-folders', action='store_true')
    parser.add_argument('--folder-labels',  action='store_true',
        help='Use folder name as class label for all files inside it.')

    # ── SVM hyperparameters ──
    parser.add_argument('--kernel', default='rbf',
        choices=['linear', 'rbf', 'poly'],
        help='SVM kernel (default: rbf).')
    parser.add_argument('--C', type=float, default=1.0,
        help='SVM regularization parameter C (default: 1.0).')
    parser.add_argument('--gamma', default='scale',
        help="Kernel coefficient for rbf/poly: 'scale', 'auto', or a float (default: scale).")
    parser.add_argument('--degree', type=int, default=3,
        help='Degree for poly kernel (default: 3).')

    # ── PCA ──
    parser.add_argument('--pca',   action='store_true',
        help='Apply PCA after scaling before SVM.')
    parser.add_argument('--pca-n', type=int, default=None,
        help='Number of PCA components to keep (default: all).')

    # ── Misc ──
    parser.add_argument('--skip-cv',      action='store_true')
    parser.add_argument('--export-model', action='store_true',
        help='Export model to STM32 C header (linear kernel only).')

    # ── Legacy C-compatible feature extraction ──
    parser.add_argument('--legacy-c-compatible', action='store_true')
    parser.add_argument('--legacy-deadzone-zc',   type=float, default=LEGACY_DEADZONE_ZC)
    parser.add_argument('--legacy-deadzone-turn',  type=float, default=LEGACY_DEADZONE_TURN)
    parser.add_argument('--legacy-scale-mav',      type=float, default=LEGACY_SCALE_MAV)
    parser.add_argument('--legacy-scale-zc',       type=float, default=LEGACY_SCALE_ZC)
    parser.add_argument('--legacy-wl-samples',     type=int,   default=LEGACY_WL_SAMPLES)
    parser.add_argument('--legacy-winc-samples',   type=int,   default=LEGACY_WINC_SAMPLES)

    args = parser.parse_args()

    # Apply runtime overrides
    globals()['LEGACY_DEADZONE_ZC']   = args.legacy_deadzone_zc
    globals()['LEGACY_DEADZONE_TURN'] = args.legacy_deadzone_turn
    globals()['LEGACY_SCALE_MAV']     = args.legacy_scale_mav
    globals()['LEGACY_SCALE_ZC']      = args.legacy_scale_zc
    globals()['LEGACY_WL_SAMPLES']    = args.legacy_wl_samples
    globals()['LEGACY_WINC_SAMPLES']  = args.legacy_winc_samples

    # Parse gamma: allow float string
    try:
        gamma = float(args.gamma)
    except ValueError:
        gamma = args.gamma  # 'scale' or 'auto'

    print("=" * 65)
    print("  EMG Support Vector Machine Classifier")
    print("=" * 65)
    print(f"  kernel={args.kernel}  C={args.C}  gamma={args.gamma}  "
          f"{'pca='+str(args.pca_n) if args.pca else 'pca=off'}")
    print("=" * 65)

    # ── Load data ──────────────────────────────────────────────────────────────
    if args.train_dir is not None:
        if args.test_dir is None:
            X, y = load_dataset(args.train_dir,
                                 prefix_labels=args.prefix_folders,
                                 folder_labels=args.folder_labels,
                                 legacy_c_compatible=args.legacy_c_compatible)
            test_data = None
        else:
            X_train, y_train = load_dataset(args.train_dir,
                                             prefix_labels=args.prefix_folders,
                                             folder_labels=args.folder_labels,
                                             legacy_c_compatible=args.legacy_c_compatible)
            X_test, y_test = load_dataset(args.test_dir,
                                           prefix_labels=args.prefix_folders,
                                           folder_labels=args.folder_labels,
                                           legacy_c_compatible=args.legacy_c_compatible)
            test_data = (X_train, y_train, X_test, y_test)
    else:
        data_dirs = args.data_dir if args.data_dir is not None else [DATA_DIR]
        X, y = load_dataset(data_dirs,
                             prefix_labels=args.prefix_folders,
                             folder_labels=args.folder_labels,
                             legacy_c_compatible=args.legacy_c_compatible)
        test_data = None

    # ── Build SVM ─────────────────────────────────────────────────────────────
    def make_svm():
        return SVC(kernel=args.kernel, C=args.C, gamma=gamma,
                   degree=args.degree, probability=False, random_state=42)

    # ── Train / test split mode ────────────────────────────────────────────────
    if test_data is not None:
        X_train, y_train, X_test, y_test = test_data

        if args.legacy_c_compatible:
            X_train_sc, lmean, lstd = mapstd_fit_transform(X_train)
            X_test_sc = mapstd_apply(X_test, lmean, lstd)
            class _Scaler:
                pass
            scaler = _Scaler()
            scaler.mean_  = lmean
            scaler.scale_ = lstd
        else:
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc  = scaler.transform(X_test)

        pca = None
        if args.pca:
            pca = PCA(n_components=args.pca_n)
            X_train_sc = pca.fit_transform(X_train_sc)
            X_test_sc  = pca.transform(X_test_sc)
            print(f"Applied PCA: {pca.n_components_} components  "
                  f"({np.sum(pca.explained_variance_ratio_)*100:.1f}% variance)")

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        classes = le.classes_

        feature_names = (infer_feature_names_legacy(X_train.shape[1])
                         if args.legacy_c_compatible
                         else infer_feature_names(X_train.shape[1]))
        if args.pca:
            svm_feature_names = [f'PC{i+1}' for i in range(pca.n_components_)]
        else:
            svm_feature_names = feature_names

        print(f"\nClasses : {list(classes)}")
        print(f"Train windows : {X_train_sc.shape[0]:,}")
        print(f"Test  windows : {X_test_sc.shape[0]:,}")
        print(f"Features/window : {X_train_sc.shape[1]}  ({', '.join(svm_feature_names)})")

        svm = make_svm()
        svm.fit(X_train_sc, y_train_enc)

        y_pred_enc = svm.predict(X_test_sc)
        y_pred     = le.inverse_transform(y_pred_enc)
        test_acc   = accuracy_score(y_test, y_pred)

        train_pred_enc = svm.predict(X_train_sc)
        train_acc = accuracy_score(y_train_enc, train_pred_enc)

        print(f"\nTraining accuracy : {train_acc:.1%}")
        print(f"Test-set accuracy : {test_acc:.1%}")
        print(f"\nPer-class accuracy (test set):")
        for cls in classes:
            mask = y_test == cls
            if mask.sum() > 0:
                cls_acc = accuracy_score(y_test[mask], y_pred[mask])
                print(f"  {cls:<25}  {cls_acc:.1%}  ({mask.sum():,} windows)")

        if args.kernel == 'linear':
            print(f"\nSVM Decision Weights (in {'PCA' if pca else 'scaled feature'} space):")
            for name, val in zip(svm_feature_names, np.ravel(svm.coef_[0])):
                print(f"  {name:>8}: {val:.4f}")
        else:
            print(f"\nSupport vectors: {svm.n_support_}  (per class)")

        print_svm_model_stats(svm, X_eval=X_test_sc)

        if args.export_model:
            export_svm_model(svm, scaler, le, pca=pca,
                             train_accuracy=train_acc, test_accuracy=test_acc,
                             feature_names=feature_names,
                             n_features_raw=X_train.shape[1])

        # Visualization: 2-D PCA scatter
        viz_pca = PCA(n_components=2)
        viz_pca.fit(X_train_sc)
        plot_confusion_matrix(y_test, y_pred, classes, eval_label='Test set')
        plot_pca_projection(X_train_sc, y_train, viz_pca,
                            X_test=X_test_sc, y_test=y_test, classes=classes)
        plt.show()
        print("\nDone. Close the plot windows to exit.")
        return

    # ── CV / data-dir mode ────────────────────────────────────────────────────
    if args.legacy_c_compatible:
        X_sc, lmean, lstd = mapstd_fit_transform(X)
        class _Scaler:
            pass
        scaler = _Scaler()
        scaler.mean_  = lmean
        scaler.scale_ = lstd
    else:
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)

    pca = None
    if args.pca:
        pca = PCA(n_components=args.pca_n)
        X_sc = pca.fit_transform(X_sc)
        print(f"Applied PCA: {pca.n_components_} components  "
              f"({np.sum(pca.explained_variance_ratio_)*100:.1f}% variance)")

    le    = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_

    feature_names = (infer_feature_names_legacy(X.shape[1])
                     if args.legacy_c_compatible
                     else infer_feature_names(X.shape[1]))
    svm_feature_names = ([f'PC{i+1}' for i in range(pca.n_components_)]
                         if args.pca else feature_names)

    print(f"\nClasses : {list(classes)}")
    print(f"Total windows : {X_sc.shape[0]:,}")
    print(f"Features/window : {X_sc.shape[1]}  ({', '.join(svm_feature_names)})")

    if args.skip_cv:
        svm = make_svm()
        svm.fit(X_sc, y_enc)
        y_pred_enc = svm.predict(X_sc)
        y_pred = le.inverse_transform(y_pred_enc)
        acc = accuracy_score(y, y_pred)
        print(f"\nTraining accuracy (no CV): {acc:.1%}")
    else:
        svm = make_svm()
        cv  = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
        y_pred_enc = cross_val_predict(svm, X_sc, y_enc, cv=cv)
        y_pred = le.inverse_transform(y_pred_enc)
        acc = accuracy_score(y, y_pred)
        print(f"\n{CV_FOLDS}-fold CV accuracy : {acc:.1%}")
        print("\nPer-class accuracy:")
        for cls in classes:
            mask = y == cls
            cls_acc = accuracy_score(y[mask], y_pred[mask])
            print(f"  {cls:<25}  {cls_acc:.1%}  ({mask.sum():,} windows)")
        svm.fit(X_sc, y_enc)   # refit on full data for export/plots

    print_svm_model_stats(svm, X_eval=X_sc)

    if args.export_model:
        train_acc = accuracy_score(y_enc, svm.predict(X_sc))
        export_svm_model(svm, scaler, le, pca=pca,
                         train_accuracy=train_acc, test_accuracy=None,
                         feature_names=feature_names,
                         n_features_raw=X.shape[1])

    viz_pca = PCA(n_components=min(2, X_sc.shape[1]))
    viz_pca.fit(X_sc)
    plot_confusion_matrix(y, y_pred, classes,
                          eval_label=f'{CV_FOLDS}-fold CV' if not args.skip_cv else 'Train set')
    plot_pca_projection(X_sc, y, viz_pca, classes=classes)
    plt.show()
    print("\nDone. Close the plot windows to exit.")


if __name__ == '__main__':
    main()
