"""
classify_emg_lda.py
--------------------
Linear Discriminant Analysis classifier for labeled EMG recordings.

DATA FORMAT EXPECTED
    Each .txt file in DATA_DIR contains one recording, one class.
    The filename (without extension) is used as the class label.
    Single-channel: <timestamp_seconds>,<ADC_value>
    Multi-channel:  <timestamp_seconds>,<CH1>,<CH2>,<CH3>
    Example (single): 1782334057.035,1839
    Example (3-ch):   1782334057.035,1839,1845,1832

USAGE
    1. Put your labeled .txt files in one or more folders next to this script.
       Each folder can represent a different training set.
    2. Run one of these:
         - Combined training with CV:
             python classify_emg_lda.py --data-dir data/setA data/setB
         - Train/test split using separate folders:
             python classify_emg_lda.py --train-dir data/train --test-dir data/test
    3. Two windows will appear:
         - Confusion matrix (per-class accuracy)
         - 2-D LDA projection (how well classes separate)
"""

import argparse
import os
import glob
import time
import numpy as np
from typing import List, Tuple

import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data'))

SAMPLE_RATE = 1000      # Hz  — used for bandpass filter and ZC threshold
WINDOW_MS   = 200       # ms  — analysis window length
OVERLAP_PCT = 0.5       # 0.5 = 50% overlap between consecutive windows

# Bandpass filter (standard EMG: 20–450 Hz)
BP_LOW  = 20.0
BP_HIGH = 450.0
BP_ORDER = 4

# Zero-crossing threshold (helps ignore noise baseline crossings)
ZC_THRESHOLD = 10       # ADC counts

# Cross-validation folds
CV_FOLDS = 5

# Legacy C-compatible TD feature parameters
LEGACY_DEADZONE_ZC = 0.025
LEGACY_DEADZONE_TURN = 0.015
LEGACY_SCALE_MAV = 2.0
LEGACY_SCALE_ZC = 15.0
LEGACY_WL_SAMPLES = 100
LEGACY_WINC_SAMPLES = 50
# ──────────────────────────────────────────────────────────────────────────────


# ── SIGNAL PROCESSING ─────────────────────────────────────────────────────────

def bandpass_filter(signal: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter. Handles 1D (single) or 2D (multi-channel) signals."""
    nyq = fs / 2.0
    low_n  = low  / nyq
    high_n = high / nyq
    low_n  = max(1e-4, min(low_n,  0.9999))
    high_n = max(1e-4, min(high_n, 0.9999))
    if low_n >= high_n:
        return signal
    b, a = butter(order, [low_n, high_n], btype='band')
    
    if signal.ndim == 1:
        return filtfilt(b, a, signal)
    else:
        # Multi-channel: filter each channel separately
        filtered = np.zeros_like(signal)
        for ch in range(signal.shape[1]):
            filtered[:, ch] = filtfilt(b, a, signal[:, ch])
        return filtered


def extract_features(window: np.ndarray) -> np.ndarray:
    """
    Extract 4 time-domain features from EMG window(s).
    Window shape: (samples,) for single channel or (samples, channels) for multi.
    
    Features (per channel):
    - MAV:   Mean Absolute Value (signal magnitude)
    - ZC:    Zero-Crossing count (frequency content proxy)
    - WL:    Waveform Length (complexity/speed of changes)
    - SSC:   Slope Sign Changes (frequency content)
    
    Returns: 4 features for single channel, or 4*channels for multi-channel.
    """
    if window.ndim == 1:
        mav = np.mean(np.abs(window))
        wl = np.sum(np.abs(np.diff(window)))
        zc = np.sum((window[:-1] * window[1:] < 0) & (np.abs(window[1:] - window[:-1]) >= ZC_THRESHOLD))
        
        # Slope Sign Changes: count where slope changes sign
        slope = np.diff(window)
        ssc = np.sum((slope[:-1] * slope[1:] < 0))
        
        return np.array([mav, float(zc), wl, float(ssc)])
    else:
        # Multi-channel: stack features from each channel
        all_feats = []
        for ch in range(window.shape[1]):
            ch_data = window[:, ch]
            mav = np.mean(np.abs(ch_data))
            wl = np.sum(np.abs(np.diff(ch_data)))
            zc = np.sum((ch_data[:-1] * ch_data[1:] < 0) & (np.abs(ch_data[1:] - ch_data[:-1]) >= ZC_THRESHOLD))
            slope = np.diff(ch_data)
            ssc = np.sum((slope[:-1] * slope[1:] < 0))
            all_feats.extend([mav, float(zc), wl, float(ssc)])
        return np.array(all_feats)


def extract_features_legacy_c(window: np.ndarray) -> np.ndarray:
    """
    C-compatible TD features per channel:
      MAV, LEN, ZC, TURNS
    Mirrors tdfeats() logic from legacy C implementation as closely as possible.
    """
    if window.ndim == 1:
        window = window[:, None]

    win_length = window.shape[0]
    n_ch = window.shape[1]

    ruler = 1.0 / float(win_length)
    ruler_sq = ruler * ruler
    lscale = float(win_length) / 40.0
    tscale = (float(win_length) / 40.0) * 10.0

    # Per-window mean removal (legacy behavior)
    w = window - np.mean(window, axis=0, keepdims=True)

    feats = []
    for ch in range(n_ch):
        x = w[:, ch]

        mav = np.mean(np.abs(x))

        zero_count = 0.0
        turns = 0.0
        length = 0.0

        flag2 = 1
        for i in range(1, win_length - 1):
            fst = abs(x[i - 1])
            mid = abs(x[i])
            lst = abs(x[i + 1])

            # Zero crossings with deadzone hysteresis (legacy logic)
            if ((x[i] >= 0 and x[i - 1] >= 0) or (x[i] <= 0 and x[i - 1] <= 0)):
                flag1 = flag2
            else:
                if (mid < LEGACY_DEADZONE_ZC) and (fst < LEGACY_DEADZONE_ZC):
                    flag1 = flag2
                else:
                    flag1 = -flag2
            if flag1 != flag2:
                zero_count += 1.0
            flag2 = flag1

            # Turns (slope sign changes) with threshold
            if ((mid > fst and mid > lst) or (mid < fst and mid < lst)):
                if ((abs(mid) - abs(fst)) > LEGACY_DEADZONE_TURN) or ((abs(mid) - abs(lst)) > LEGACY_DEADZONE_TURN):
                    turns += 1.0

            # Waveform length (legacy formula)
            d = (fst - mid) / 20.0
            length += float(np.sqrt(d * d + ruler_sq))

        # Legacy feature scaling
        zero_count = (zero_count / LEGACY_SCALE_ZC) * 40.0 / float(win_length)
        mav = mav / LEGACY_SCALE_MAV
        length = (length - 1.0) / lscale
        turns = turns / tscale

        feats.extend([mav, length, zero_count, turns])

    return np.array(feats, dtype=float)


def segment_and_extract(values: np.ndarray, fs: float, window_ms: int, overlap: float) -> np.ndarray:
    """Slide window and extract features. Handles single or multi-channel input."""
    win_samples = int(fs * window_ms / 1000)
    step_samples = int(win_samples * (1.0 - overlap))
    n_ch = 1 if values.ndim == 1 else values.shape[1]
    
    if win_samples < 10 or len(values) < win_samples:
        return np.empty((0, 4 * n_ch))

    features = []
    start = 0
    while start + win_samples <= len(values):
        segment = values[start : start + win_samples]
        features.append(extract_features(segment))
        start += step_samples

    return np.array(features) if features else np.empty((0, 4 * n_ch))


def segment_and_extract_legacy_c(values: np.ndarray, fs: float, window_ms: int, overlap: float) -> np.ndarray:
    """Windowing + legacy C-compatible TD features."""
    # Use exact sample-based legacy settings (matching WL/WINC in C code)
    win_samples = int(LEGACY_WL_SAMPLES)
    step_samples = int(LEGACY_WINC_SAMPLES)
    n_ch = 1 if values.ndim == 1 else values.shape[1]

    if win_samples < 10 or len(values) < win_samples:
        return np.empty((0, 4 * n_ch))

    feats = []
    start = 0
    while start + win_samples <= len(values):
        seg = values[start:start + win_samples]
        feats.append(extract_features_legacy_c(seg))
        start += step_samples

    return np.array(feats) if feats else np.empty((0, 4 * n_ch))


def mapstd_fit_transform(X: np.ndarray):
    """Legacy mapstd-like normalization using sample std (ddof=1)."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=1)
    std = np.where(std == 0, 1.0, std)
    Xn = (X - mean) / std
    return Xn, mean, std


def mapstd_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_file(path):
    """
    Load EMG data from txt file. Auto-detects format:
    - Old: comma-separated, no header: timestamp,ch1,ch2,ch3
    - New: tab-separated, with header: timestamp    ch1    ch2    ch3
    Returns (N_samples, N_channels) array.
    """
    values = []
    with open(path, 'r') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # Skip header row if it exists (contains "timestamp" or "ch")
            if line_idx == 0 and ('timestamp' in line.lower() or 'ch' in line.lower()):
                continue
            
            # Try tab-separated first (new format), then comma-separated (old format)
            if '\t' in line:
                parts = line.split('\t')
            else:
                parts = line.split(',')
            
            try:
                # Skip first column (timestamp), read remaining as channels
                channels = [float(p.strip()) for p in parts[1:] if p.strip()]
                if channels:  # Only append if we got valid channels
                    values.append(channels)
            except ValueError:
                continue
    
    if not values:
        return np.empty((0, 1), dtype=float)
    return np.array(values, dtype=float)


def resolve_data_dirs(data_dirs):
    """Resolve directory paths relative to this script or project root."""
    if data_dirs is None:
        return []
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    resolved = []
    for path in data_dirs:
        if not os.path.isabs(path):
            # Try relative to script first, then relative to project root
            script_relative = os.path.abspath(os.path.join(SCRIPT_DIR, path))
            if os.path.isdir(script_relative):
                path = script_relative
            else:
                project_relative = os.path.abspath(os.path.join(SCRIPT_DIR, '..', path))
                path = project_relative
        resolved.append(path)
    return resolved


def load_dataset(data_dirs, prefix_labels=False, folder_labels=False, legacy_c_compatible=False):
    """
    Load all .txt files from one or more directories.
    Returns X (feature matrix) and y (string labels).

    If folder_labels is True, every file inside a directory shares the directory name as its class label.
    Otherwise each file name is used as its own label, with optional folder prefixing.
    """
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    data_dirs = resolve_data_dirs(data_dirs)

    txt_files = []
    for data_dir in data_dirs:
        paths = sorted(glob.glob(os.path.join(data_dir, '*.txt')))
        if not paths:
            raise FileNotFoundError(
                f"No .txt files found in:\n  {data_dir}\n"
                "Make sure your labeled recordings are there."
            )
        txt_files.extend([(data_dir, path) for path in paths])

    all_features = []
    all_labels   = []

    print(f"\nLoading files from: {', '.join(data_dirs)}")
    print(f"{'File':<30}  {'Samples':>8}  {'Windows':>8}  Label")
    print("-" * 65)

    for data_dir, path in txt_files:
        base_label = os.path.splitext(os.path.basename(path))[0]
        if folder_labels:
            folder_name = os.path.basename(os.path.normpath(data_dir))
            # Strip _test or _train suffix if present to map back to original class
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
            # Legacy path: no global filter/centering here; legacy extractor does per-window mean removal.
            feats = segment_and_extract_legacy_c(raw, SAMPLE_RATE, WINDOW_MS, OVERLAP_PCT)
        else:
            # Apply bandpass filter (per-channel for multi-channel data)
            filtered  = bandpass_filter(raw, SAMPLE_RATE, BP_LOW, BP_HIGH, BP_ORDER)

            # Centre signal (remove DC offset, per-channel for multi-channel)
            if filtered.ndim == 1:
                centered = filtered - np.mean(filtered)
            else:
                centered = filtered - np.mean(filtered, axis=0, keepdims=True)

            feats = segment_and_extract(centered, SAMPLE_RATE, WINDOW_MS, OVERLAP_PCT)

        if feats.shape[0] == 0:
            print(f"  ⚠  {label}: too short to segment, skipping.")
            continue

        n_samples = raw.shape[0]
        print(f"  {label:<28}  {n_samples:>8,}  {feats.shape[0]:>8,}  ✓")
        all_features.append(feats)
        all_labels.extend([label] * feats.shape[0])

    if not all_features:
        raise ValueError("No usable data found after loading all files.")

    X = np.vstack(all_features)
    y = np.array(all_labels)
    return X, y


# ── PLOTTING ──────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, label_names, eval_label='Evaluation'):
    cm   = confusion_matrix(y_true, y_pred, labels=label_names)
    acc  = accuracy_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(max(5, len(label_names) * 1.4),
                                    max(4, len(label_names) * 1.2)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(
        f'LDA Confusion Matrix  —  {eval_label}\n'
        f'Overall accuracy: {acc * 100:.1f}%',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()


def plot_lda_projection(X_train, y_train, lda_model, le, X_test=None, y_test=None):
    """Project data onto the first 2 LDA axes and scatter-plot."""
    X_train_lda = lda_model.transform(X_train)
    n_components = X_train_lda.shape[1]

    fig, ax = plt.subplots(figsize=(8, 6))
    classes  = le.classes_
    colors   = plt.cm.tab10(np.linspace(0, 1, len(classes)))

    for i, cls in enumerate(classes):
        train_mask = y_train == cls
        if n_components >= 2:
            ax.scatter(
                X_train_lda[train_mask, 0],
                X_train_lda[train_mask, 1],
                label=f'{cls} (train)', alpha=0.45, s=18, color=colors[i], marker='o'
            )
        else:
            ax.scatter(
                X_train_lda[train_mask, 0],
                np.zeros(train_mask.sum()) + i * 0.1,
                label=f'{cls} (train)', alpha=0.45, s=18, color=colors[i], marker='o'
            )

    if X_test is not None and y_test is not None:
        X_test_lda = lda_model.transform(X_test)
        for i, cls in enumerate(classes):
            test_mask = y_test == cls
            if test_mask.sum() == 0:
                continue
            if n_components >= 2:
                ax.scatter(
                    X_test_lda[test_mask, 0],
                    X_test_lda[test_mask, 1],
                    label=f'{cls} (test)', alpha=0.8, s=22,
                    color=colors[i], marker='x'
                )
            else:
                ax.scatter(
                    X_test_lda[test_mask, 0],
                    np.zeros(test_mask.sum()) + i * 0.1,
                    label=f'{cls} (test)', alpha=0.8, s=22,
                    color=colors[i], marker='x'
                )

    if n_components >= 2:
        ax.set_xlabel('LD1')
        ax.set_ylabel('LD2')
    else:
        ax.set_xlabel('LD1')
        ax.set_yticks([])

    title = 'LDA Projection'
    if X_test is not None and y_test is not None:
        title += ' (train vs test)'
    else:
        title += ' (training data)'

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(title='Class', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()


# ── MODEL EXPORT ──────────────────────────────────────────────────────────────

def infer_feature_names(n_features: int):
    """Infer feature names for 4-feature-per-channel EMG vectors."""
    base = ['MAV', 'ZC', 'WL', 'SSC']
    if n_features <= len(base):
        return base[:n_features]
    if n_features % 4 == 0:
        names = []
        n_channels = n_features // 4
        for ch in range(n_channels):
            names.extend([f"{b}_ch{ch+1}" for b in base])
        return names
    return [f"F{i+1}" for i in range(n_features)]


def infer_feature_names_legacy(n_features: int):
    base = ['MAV', 'LEN', 'ZC', 'TURNS']
    if n_features <= len(base):
        return base[:n_features]
    if n_features % 4 == 0:
        names = []
        n_channels = n_features // 4
        for ch in range(n_channels):
            names.extend([f"{b}_ch{ch+1}" for b in base])
        return names
    return [f"F{i+1}" for i in range(n_features)]

def export_model_metadata(
    scaler=None,
    pca=None,
    lda=None,
    le=None,
    train_accuracy=None,
    test_accuracy=None,
    n_classes=2,
    feature_dim=12,
    dataset_name="ResultClipSizeUp900",
    sample_size=900,
):
    """Generate STM32-compatible LDA model header with weights."""
    from datetime import datetime
    
    # Export full model if scaler + lda exist. PCA is optional.
    if scaler is not None and lda is not None:
        # Binary-class export format with symmetric class weights.
        lda_vec = np.ravel(lda.coef_[0])
        lda_bias = float(np.ravel(lda.intercept_)[0])

        # Back-project from PCA space if PCA is enabled; otherwise direct LDA.
        if pca is not None:
            # raw_standardized -> PCA -> LDA
            raw_vec = pca.components_.T @ lda_vec
            mode_str = "multipletrain_baseline:pca_lda"
            pca_note = f"// PCA Explained Variance: {np.sum(pca.explained_variance_ratio_)*100:.1f}% ({pca.n_components_} components)\n"
        else:
            # raw_standardized -> LDA
            raw_vec = lda_vec
            mode_str = "multipletrain_baseline:lda_only"
            pca_note = "// PCA: disabled (strict LDA only)\n"

        # Undo standardization so embedded inference can use raw features + xmean/xstd.
        # decision = dot(raw_vec, (x - mean)/std) + lda_bias
        # => decision = dot(raw_w, x) + raw_b
        raw_w = raw_vec / scaler.scale_
        raw_b = lda_bias - np.dot((scaler.mean_ / scaler.scale_), raw_vec)

        # Export two-class arrays in the expected format.
        Wg_scaled = np.column_stack([raw_w, -raw_w])
        Cg = np.array([raw_b, -raw_b])

        n_feat = len(raw_w)
        feature_names = infer_feature_names(n_feat)

        train_acc_str = f"{train_accuracy*100:.2f}%" if train_accuracy is not None else "N/A"
        test_acc_str = f"{test_accuracy*100:.2f}%" if test_accuracy is not None else "N/A"

        # Generate header
        header = f"""// ============================================================
// Auto-generated LDA model for STM32
// mode: {mode_str}
// dataset: {dataset_name}
// sample_size: {sample_size}
// train_accuracy: {train_acc_str}
// test_accuracy: {test_acc_str}
// generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
// num_class: {n_classes}
// feature_dim: {n_feat}
// features: {', '.join(feature_names)}
// classes: act1 (0), act2 (1)
// ============================================================

float Wg_init[{n_feat * 2}] = {{
"""
        
        # Format Wg (n_feat features x 2 classes)
        for i in range(n_feat):
            for j in range(2):
                val = Wg_scaled[i, j]
                header += f"    {val:12.6f}{',' if not (i == n_feat - 1 and j == 1) else ''}\n"
        
        header += f"""}};

float Cg_init[2] = {{
"""
        
        # Format Cg (2 classes)
        for i in range(2):
            val = Cg[i]
            header += f"    {val:12.6f}{',' if i < 1 else ''}\n"
        
        header += f"""}};

float xstd_init[{n_feat}] = {{
"""
        
        # Format scale (std)
        for i in range(n_feat):
            val = scaler.scale_[i]
            header += f"    {val:12.6f}{',' if i < n_feat - 1 else ''}\n"
        
        header += f"""}};

float xmean_init[{n_feat}] = {{
"""
        
        # Format mean
        for i in range(n_feat):
            val = scaler.mean_[i]
            header += f"    {val:12.6f}{',' if i < n_feat - 1 else ''}\n"
        
        header += """};

"""
        header += pca_note
        
        # Save to file
        output_file = os.path.join(SCRIPT_DIR, '..', 'EMG_LDA_Model_STM32.h')
        output_file = os.path.abspath(output_file)
        commented_output_file = os.path.join(SCRIPT_DIR, '..', 'EMG_LDA_Model_STM32_commented.txt')
        commented_output_file = os.path.abspath(commented_output_file)
        
        with open(output_file, 'w') as f:
            f.write(header)

        # Also save a fully commented version (all lines commented out)
        commented_lines = []
        for line in header.splitlines():
            if line.strip() == '':
                commented_lines.append('//')
            elif line.lstrip().startswith('//'):
                commented_lines.append(f"// {line.lstrip()[2:].lstrip()}")
            else:
                commented_lines.append(f"// {line}")
        commented_text = "\n".join(commented_lines) + "\n"

        with open(commented_output_file, 'w') as f:
            f.write(commented_text)
        
        print(header)
        print(f"\n✓ Model exported to: {output_file}\n")
        print(f"✓ Commented export written to: {commented_output_file}\n")
    else:
        # Fallback to simple metadata
        print("\n" + "// " + "=" * 58)
        print("// Auto-generated LDA model for STM32")
        print(f"// dataset: {dataset_name}")
        print(f"// sample_size: {sample_size}")
        if train_accuracy is not None:
            print(f"// train_accuracy: {train_accuracy:.2%}")
        if test_accuracy is not None:
            print(f"// test_accuracy: {test_accuracy:.2%}")
        print(f"// generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"// num_class: {n_classes}")
        print(f"// feature_dim: {feature_dim}")
        print("// " + "=" * 58 + "\n")


def print_lda_model_stats(lda, scaler=None, pca=None, X_raw_eval=None):
    """Print LDA parameter count, FLOPs estimate, model size, and runtime."""
    def _fmt_compact(n: int) -> str:
        if n >= 1_000_000:
            return f"{n / 1_000_000:.3f}M"
        if n >= 1_000:
            return f"{n / 1_000:.3f}K"
        return str(n)

    def _print_size_block(title: str, params: int, flops: int):
        mem_mb = (params * 4) / (1024.0 * 1024.0)
        print(f"\n{title}")
        print(f"  Total parameters: {params:,} ({_fmt_compact(params)})")
        print(f"  Trainable parameters: {params:,} ({_fmt_compact(params)})")
        print(f"  Approx. parameter memory: {mem_mb:.6f} MB")
        print(f"  Approx. FLOPs per single-window forward pass: {flops:,} ({_fmt_compact(flops)})")

    coef = np.atleast_2d(np.array(lda.coef_, dtype=float))
    intercept = np.ravel(np.array(lda.intercept_, dtype=float))

    # Classifier-space dimensions
    n_scores = coef.shape[0]
    n_features_cls = coef.shape[1]

    # Raw input feature dimension (before scaler/PCA)
    if scaler is not None and hasattr(scaler, 'mean_'):
        n_features_raw = int(np.ravel(scaler.mean_).size)
    elif pca is not None and hasattr(pca, 'components_'):
        n_features_raw = int(pca.components_.shape[1])
    else:
        n_features_raw = n_features_cls

    params_classifier = int(coef.size + intercept.size)

    params_preproc = 0
    if scaler is not None and hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
        params_preproc += int(np.ravel(scaler.mean_).size + np.ravel(scaler.scale_).size)
    if pca is not None and hasattr(pca, 'components_'):
        params_preproc += int(pca.components_.size)
        if hasattr(pca, 'mean_'):
            params_preproc += int(np.ravel(pca.mean_).size)

    params_total = params_classifier + params_preproc

    # FLOPs per sample (rough estimate)
    flops_classifier = int(n_scores * (2 * n_features_cls))

    flops_preproc = 0
    if scaler is not None:
        # (x - mean) / std per feature
        flops_preproc += int(2 * n_features_raw)
    if pca is not None and hasattr(pca, 'components_'):
        n_comp = int(pca.components_.shape[0])
        # PCA transform per sample: center + matrix multiply
        flops_preproc += int(n_features_raw + n_comp * (2 * n_features_raw - 1))

    flops_total = flops_classifier + flops_preproc

    _print_size_block("Model size (LDA classifier)", params_classifier, flops_classifier)

    # STM32 exported representation in this project (binary only):
    # Wg_init has +/- symmetric columns and Cg_init has +/- biases.
    n_classes = int(len(getattr(lda, 'classes_', []))) if hasattr(lda, 'classes_') else 0
    if n_classes == 2 and scaler is not None:
        params_export_classifier = int(2 * n_features_raw + 2)  # Wg_init + Cg_init
        flops_export_classifier = int(2 * (2 * n_features_raw))

        _print_size_block(
            "Model size (STM32 fused classifier)",
            params_export_classifier,
            flops_export_classifier,
        )

    # Runtime benchmark (vectorized; representative host-side estimate)
    if X_raw_eval is not None and len(X_raw_eval) > 0:
        X_raw_eval = np.asarray(X_raw_eval, dtype=float)
        reps = max(20, min(500, int(40000 / max(1, X_raw_eval.shape[0]))))

        # Classifier-only timing: assume already in classifier feature space
        X_cls = X_raw_eval
        if scaler is not None:
            X_cls = (X_cls - scaler.mean_) / scaler.scale_
        if pca is not None:
            X_cls = pca.transform(X_cls)

        t0 = time.perf_counter()
        for _ in range(reps):
            _ = lda.decision_function(X_cls)
        t1 = time.perf_counter()
        us_cls = ((t1 - t0) / (reps * X_cls.shape[0])) * 1e6

        print("  Runtime (host, estimated):")
        print(f"    Classifier only: {us_cls:.4f} us/sample")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train an LDA classifier on one or more EMG data folders.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--data-dir',
        nargs='+',
        help='One or more directories containing labeled .txt EMG files for combined training + CV.',
    )
    group.add_argument(
        '--train-dir',
        nargs='+',
        help='One or more directories containing labeled .txt EMG files for training only.',
    )
    parser.add_argument(
        '--test-dir',
        nargs='+',
        help='One or more directories containing labeled .txt EMG files for testing. Requires --train-dir.',
    )
    parser.add_argument(
        '--prefix-folders',
        action='store_true',
        help='Prefix labels with the source folder name when loading multiple data directories.',
    )
    parser.add_argument(
        '--folder-labels',
        action='store_true',
        help='Use the folder name as the class label for all files inside that folder.',
    )
    parser.add_argument(
        '--pca',
        action='store_true',
        help='Apply PCA after scaling before LDA.',
    )
    parser.add_argument(
        '--pca-n',
        type=int,
        default=None,
        help='Number of PCA components to keep (default: keep all).',
    )
    parser.add_argument(
        '--skip-cv',
        action='store_true',
        help='Skip cross-validation; only run LDA training (requires --data-dir or --train-dir/--test-dir).',
    )
    parser.add_argument(
        '--export-model',
        action='store_true',
        help='Export model metadata in C/STM32 format after training.',
    )
    parser.add_argument(
        '--legacy-c-compatible',
        action='store_true',
        help='Use C-legacy-compatible TD features and mapstd-style normalization to match legacy weights.',
    )
    parser.add_argument('--legacy-deadzone-zc', type=float, default=LEGACY_DEADZONE_ZC,
                        help='Legacy deadzone threshold for zero crossings (default: 0.025).')
    parser.add_argument('--legacy-deadzone-turn', type=float, default=LEGACY_DEADZONE_TURN,
                        help='Legacy deadzone threshold for turns/SSC (default: 0.015).')
    parser.add_argument('--legacy-scale-mav', type=float, default=LEGACY_SCALE_MAV,
                        help='Legacy MAV scale factor (default: 2).')
    parser.add_argument('--legacy-scale-zc', type=float, default=LEGACY_SCALE_ZC,
                        help='Legacy ZC scale factor (default: 15).')
    parser.add_argument('--legacy-wl-samples', type=int, default=LEGACY_WL_SAMPLES,
                        help='Legacy window length in samples (default: 100).')
    parser.add_argument('--legacy-winc-samples', type=int, default=LEGACY_WINC_SAMPLES,
                        help='Legacy window increment in samples (default: 50).')
    args = parser.parse_args()

    # Apply runtime overrides for legacy compatibility mode
    globals()['LEGACY_DEADZONE_ZC'] = float(args.legacy_deadzone_zc)
    globals()['LEGACY_DEADZONE_TURN'] = float(args.legacy_deadzone_turn)
    globals()['LEGACY_SCALE_MAV'] = float(args.legacy_scale_mav)
    globals()['LEGACY_SCALE_ZC'] = float(args.legacy_scale_zc)
    globals()['LEGACY_WL_SAMPLES'] = int(args.legacy_wl_samples)
    globals()['LEGACY_WINC_SAMPLES'] = int(args.legacy_winc_samples)

    print("=" * 65)
    print("  EMG Linear Discriminant Analysis Classifier")
    print("=" * 65)

    if args.train_dir is not None:
        if args.test_dir is None:
            X, y = load_dataset(
                args.train_dir,
                prefix_labels=args.prefix_folders,
                folder_labels=args.folder_labels,
                legacy_c_compatible=args.legacy_c_compatible,
            )
            test_data = None
        else:
            X_train, y_train = load_dataset(
                args.train_dir,
                prefix_labels=args.prefix_folders,
                folder_labels=args.folder_labels,
                legacy_c_compatible=args.legacy_c_compatible,
            )
            X_test, y_test = load_dataset(
                args.test_dir,
                prefix_labels=args.prefix_folders,
                folder_labels=args.folder_labels,
                legacy_c_compatible=args.legacy_c_compatible,
            )
            test_data = (X_train, y_train, X_test, y_test)
    else:
        data_dirs = args.data_dir if args.data_dir is not None else [DATA_DIR]
        X, y = load_dataset(
            data_dirs,
            prefix_labels=args.prefix_folders,
            folder_labels=args.folder_labels,
            legacy_c_compatible=args.legacy_c_compatible,
        )
        test_data = None

    if test_data is None:
        X_raw = X.copy()
        if args.skip_cv:
            print("\n[--skip-cv] Skipping cross-validation. Training on full dataset (no held-out test).")
            # Just train on all data without CV
            if args.legacy_c_compatible:
                X_scaled, legacy_mean, legacy_std = mapstd_fit_transform(X)
                class _ScalerObj:
                    pass
                scaler = _ScalerObj()
                scaler.mean_ = legacy_mean
                scaler.scale_ = legacy_std
            else:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            if args.pca:
                pca = PCA(n_components=args.pca_n)
                X_transformed = pca.fit_transform(X_scaled)
                print(f"Applied PCA: n_components={pca.n_components_}")
                feature_names = [f'PC{i+1}' for i in range(pca.n_components_)]
            else:
                X_transformed = X_scaled
                feature_names = infer_feature_names_legacy(X_transformed.shape[1]) if args.legacy_c_compatible else infer_feature_names(X_transformed.shape[1])

            le      = LabelEncoder()
            y_enc   = le.fit_transform(y)
            classes = le.classes_
            n_cls   = len(classes)

            print(f"\nClasses found  : {list(classes)}")
            print(f"Total windows  : {X_transformed.shape[0]:,}")
            print(f"Features/window: {X_transformed.shape[1]}  ({', '.join(feature_names)})")

            n_comp = min(n_cls - 1, X_transformed.shape[1])
            lda    = LinearDiscriminantAnalysis(n_components=n_comp, solver='svd')
            lda.fit(X_transformed, y_enc)

            # Train accuracy (on full data)
            y_pred_enc = lda.predict(X_transformed)
            y_pred     = le.inverse_transform(y_pred_enc)
            acc = accuracy_score(y, y_pred)
            print(f'\nTraining accuracy: {acc:.1%}')

            # Print LDA feature contributions
            scalings = getattr(lda, 'scalings_', None)
            coef = getattr(lda, 'coef_', None)
            print('\nLDA feature contributions:')
            if scalings is not None:
                arr = np.array(scalings)
                if arr.ndim == 1:
                    for name, val in zip(feature_names, arr):
                        print(f"  {name:>4}: {val:.4f}")
                else:
                    for idx, name in enumerate(feature_names):
                        vals = ', '.join(f"{x:.4f}" for x in arr[idx, :])
                        print(f"  {name:>4}: {vals}")
            elif coef is not None:
                arr = np.array(coef)
                if arr.ndim == 1:
                    for name, val in zip(feature_names, arr):
                        print(f"  {name:>4}: {val:.4f}")
                else:
                    for idx, name in enumerate(feature_names):
                        vals = ', '.join(f"{x:.4f}" for x in arr[:, idx])
                        print(f"  {name:>4}: {vals}")
            else:
                print('  (no scalings or coef_ available for this solver)')

            # Print additional model info
            if args.pca:
                print('\nPCA Loadings (first 2 components):')
                print(f'  Shape: {pca.components_.shape} (n_components x n_features)')
                for i in range(min(2, pca.components_.shape[0])):
                    print(f'  PC{i+1}: {pca.components_[i][:6]}...')
                print(f'\nPCA Explained Variance Ratio: {pca.explained_variance_ratio_}')
                print(f'  Total variance explained: {np.sum(pca.explained_variance_ratio_):.1%}')

            if hasattr(lda, 'explained_variance_ratio_'):
                print(f'\nLDA Explained Variance Ratio: {lda.explained_variance_ratio_}')

            print(f'\nLDA Intercept: {lda.intercept_}')
            print(f'LDA Classes: {lda.classes_}')

            # Export model metadata if requested
            if args.export_model:
                export_model_metadata(
                    scaler=scaler,
                    pca=pca if args.pca else None,
                    lda=lda,
                    le=le,
                    train_accuracy=acc,
                    test_accuracy=None,
                    n_classes=n_cls,
                    feature_dim=X.shape[1],
                    dataset_name="custom_data",
                    sample_size=X.shape[0],
                )

            print_lda_model_stats(lda=lda, scaler=scaler, pca=(pca if args.pca else None), X_raw_eval=X_raw)

            plt.show()
            return
        
        # Regular CV mode (if --skip-cv not set)
        if args.legacy_c_compatible:
            X_scaled, legacy_mean, legacy_std = mapstd_fit_transform(X)
            class _ScalerObj:
                pass
            scaler = _ScalerObj()
            scaler.mean_ = legacy_mean
            scaler.scale_ = legacy_std
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        if args.pca:
            pca = PCA(n_components=args.pca_n)
            X = pca.fit_transform(X_scaled)
            print(f"Applied PCA: n_components={pca.n_components_}")
            feature_names = [f'PC{i+1}' for i in range(pca.n_components_)]
        else:
            X = X_scaled
            feature_names = infer_feature_names_legacy(X.shape[1]) if args.legacy_c_compatible else infer_feature_names(X.shape[1])

        le      = LabelEncoder()
        y_enc   = le.fit_transform(y)
        classes = le.classes_
        n_cls   = len(classes)

        print(f"\nClasses found  : {list(classes)}")
        print(f"Total windows  : {X.shape[0]:,}")
        print(f"Features/window: {X.shape[1]}  ({', '.join(feature_names)})")

        n_comp = min(n_cls - 1, X.shape[1])
        lda    = LinearDiscriminantAnalysis(n_components=n_comp, solver='svd')

        cv      = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
        y_pred_enc = cross_val_predict(lda, X, y_enc, cv=cv)
        y_pred  = le.inverse_transform(y_pred_enc)

        acc = accuracy_score(y, y_pred)
        print(f"\n{CV_FOLDS}-fold CV accuracy : {acc * 100:.1f}%")

        print("\nPer-class accuracy:")
        for cls in classes:
            mask   = y == cls
            cls_acc = accuracy_score(y[mask], y_pred[mask])
            print(f"  {cls:<25}  {cls_acc * 100:.1f}%  ({mask.sum():,} windows)")

        lda.fit(X, y_enc)

        # Print LDA feature contributions for debugging/interpretation
        scalings = getattr(lda, 'scalings_', None)
        coef = getattr(lda, 'coef_', None)
        print('\nLDA feature contributions:')
        if scalings is not None:
            arr = np.array(scalings)
            if arr.ndim == 1:
                for name, val in zip(feature_names, arr):
                    print(f"  {name:>4}: {val:.4f}")
            else:
                for idx, name in enumerate(feature_names):
                    vals = ', '.join(f"{x:.4f}" for x in arr[idx, :])
                    print(f"  {name:>4}: {vals}")
        elif coef is not None:
            arr = np.array(coef)
            if arr.ndim == 1:
                for name, val in zip(feature_names, arr):
                    print(f"  {name:>4}: {val:.4f}")
            else:
                # coef shape (n_classes-1, n_features)
                for idx, name in enumerate(feature_names):
                    vals = ', '.join(f"{x:.4f}" for x in arr[:, idx])
                    print(f"  {name:>4}: {vals}")
        else:
            print('  (no scalings or coef_ available for this solver)')

        # Print additional model weights and parameters
        if args.pca:
            print('\nPCA Loadings (first 2 components):')
            print(f'  Shape: {pca.components_.shape} (n_components x n_features)')
            for i in range(min(2, pca.components_.shape[0])):
                print(f'  PC{i+1}: {pca.components_[i][:6]}...')  # Show first 6
            print(f'\nPCA Explained Variance Ratio: {pca.explained_variance_ratio_}')
            print(f'  Total variance explained: {np.sum(pca.explained_variance_ratio_):.1%}')

        if hasattr(lda, 'explained_variance_ratio_'):
            print(f'\nLDA Explained Variance Ratio: {lda.explained_variance_ratio_}')

        print(f'\nLDA Intercept: {lda.intercept_}')
        print(f'LDA Classes: {lda.classes_}')

        print_lda_model_stats(lda=lda, scaler=scaler, pca=(pca if args.pca else None), X_raw_eval=X_raw)

        plot_confusion_matrix(y, y_pred, classes, eval_label=f'{CV_FOLDS}-fold CV')
        plot_lda_projection(X, y, lda, le)
        plt.show()
    else:
        X_train, y_train, X_test, y_test = test_data
        X_test_raw = X_test.copy()

        if args.legacy_c_compatible:
            X_train_scaled, legacy_mean, legacy_std = mapstd_fit_transform(X_train)
            X_test_scaled = mapstd_apply(X_test, legacy_mean, legacy_std)
            class _ScalerObj:
                pass
            scaler = _ScalerObj()
            scaler.mean_ = legacy_mean
            scaler.scale_ = legacy_std
        else:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        if args.pca:
            pca = PCA(n_components=args.pca_n)
            X_train = pca.fit_transform(X_train_scaled)
            X_test = pca.transform(X_test_scaled)
            print(f"Applied PCA: n_components={pca.n_components_}")
            feature_names = [f'PC{i+1}' for i in range(pca.n_components_)]
        else:
            X_train = X_train_scaled
            X_test = X_test_scaled
            feature_names = infer_feature_names_legacy(X_train.shape[1]) if args.legacy_c_compatible else infer_feature_names(X_train.shape[1])

        le      = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        classes = le.classes_
        n_cls   = len(classes)

        print(f"\nClasses found in train set : {list(classes)}")
        print(f"Training windows  : {X_train.shape[0]:,}")
        print(f"Testing windows   : {X_test.shape[0]:,}")
        print(f"Features/window: {X_train.shape[1]}  ({', '.join(feature_names)})")

        n_comp = min(n_cls - 1, X_train.shape[1])
        lda    = LinearDiscriminantAnalysis(n_components=n_comp, solver='svd')
        lda.fit(X_train, y_train_enc)

        # Print LDA feature contributions for interpretation
        scalings = getattr(lda, 'scalings_', None)
        coef = getattr(lda, 'coef_', None)
        print('\nLDA feature contributions:')
        if scalings is not None:
            arr = np.array(scalings)
            if arr.ndim == 1:
                for name, val in zip(feature_names, arr):
                    print(f"  {name:>4}: {val:.4f}")
            else:
                for idx, name in enumerate(feature_names):
                    vals = ', '.join(f"{x:.4f}" for x in arr[idx, :])
                    print(f"  {name:>4}: {vals}")
        elif coef is not None:
            arr = np.array(coef)
            if arr.ndim == 1:
                for name, val in zip(feature_names, arr):
                    print(f"  {name:>4}: {val:.4f}")
            else:
                for idx, name in enumerate(feature_names):
                    vals = ', '.join(f"{x:.4f}" for x in arr[:, idx])
                    print(f"  {name:>4}: {vals}")
        else:
            print('  (no scalings or coef_ available for this solver)')

        # Print additional model weights and parameters
        if args.pca:
            print('\nPCA Loadings (first 2 components):')
            print(f'  Shape: {pca.components_.shape} (n_components x n_features)')
            for i in range(min(2, pca.components_.shape[0])):
                print(f'  PC{i+1}: {pca.components_[i][:6]}...')  # Show first 6
            print(f'\nPCA Explained Variance Ratio: {pca.explained_variance_ratio_}')
            print(f'  Total variance explained: {np.sum(pca.explained_variance_ratio_):.1%}')

        if hasattr(lda, 'explained_variance_ratio_'):
            print(f'\nLDA Explained Variance Ratio: {lda.explained_variance_ratio_}')

        print(f'\nLDA Intercept: {lda.intercept_}')
        print(f'LDA Classes: {lda.classes_}')

        y_test_enc = lda.predict(X_test)
        y_pred  = le.inverse_transform(y_test_enc)

        acc = accuracy_score(y_test, y_pred)
        print(f"\nTest-set accuracy : {acc * 100:.1f}%")

        print("\nPer-class accuracy (test set):")
        for cls in classes:
            mask   = y_test == cls
            if mask.sum() > 0:
                cls_acc = accuracy_score(y_test[mask], y_pred[mask])
                print(f"  {cls:<25}  {cls_acc * 100:.1f}%  ({mask.sum():,} windows)")

        # Export model metadata if requested
        if args.export_model:
            train_acc = np.mean(lda.predict(X_train) == y_train_enc)
            export_model_metadata(
                scaler=scaler,
                pca=pca if args.pca else None,
                lda=lda,
                le=le,
                train_accuracy=train_acc,
                test_accuracy=acc,
                n_classes=len(classes),
                feature_dim=X_test.shape[1],
                dataset_name="ResultClipSizeUp900",
                sample_size=900,
            )

        print_lda_model_stats(lda=lda, scaler=scaler, pca=(pca if args.pca else None), X_raw_eval=X_test_raw)

        plot_confusion_matrix(y_test, y_pred, classes, eval_label='Held-out test set')
        plot_lda_projection(X_train, y_train, lda, le, X_test=X_test, y_test=y_test)
        plt.show()

    print("\nDone. Close the plot windows to exit.")


if __name__ == '__main__':
    main()