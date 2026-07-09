import os
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Import processing helpers from classify_emg_lda
import sys, os
# Ensure project root is importable when the script is run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import scripts.classify_emg_lda as clfmod


def load_raw_records(data_dirs, folder_labels=False):
    data_dirs = clfmod.resolve_data_dirs(data_dirs)
    records = []  # list of (label, raw_values)
    for data_dir in data_dirs:
        for path in sorted(__import__('glob').glob(os.path.join(data_dir, '*.txt'))):
            base_label = os.path.splitext(os.path.basename(path))[0]
            if folder_labels:
                label = os.path.basename(os.path.normpath(data_dir))
                label = label.replace('_test', '').replace('_train', '')
            else:
                label = base_label
            raw = clfmod.load_file(path)
            if raw.size == 0:
                continue
            records.append((label, raw))
    return records


def build_dataset(records, fs, window_ms, overlap, bp_low, bp_high, bp_order):
    X_all = []
    y_all = []
    for label, raw in records:
        filtered = clfmod.bandpass_filter(raw, fs, bp_low, bp_high, bp_order)
        centered = filtered - np.mean(filtered)
        feats = clfmod.segment_and_extract(centered, fs, window_ms, overlap)
        if feats.size == 0:
            continue
        X_all.append(feats)
        y_all.extend([label] * feats.shape[0])
    if not X_all:
        return None, None
    X = np.vstack(X_all)
    y = np.array(y_all)
    return X, y


def evaluate(data_dirs, folder_labels=False):
    records = load_raw_records(data_dirs, folder_labels=folder_labels)
    if not records:
        print('No records found.')
        return

    window_list = [100, 200, 300]
    bp_low_list = [10, 20, 30]
    classifiers = {
        'LDA': LinearDiscriminantAnalysis(),
        'SVM': SVC(kernel='linear'),
        'RF': RandomForestClassifier(n_estimators=100, random_state=0),
    }

    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for win in window_list:
        for bp_low in bp_low_list:
            X, y = build_dataset(records, clfmod.SAMPLE_RATE, win, clfmod.OVERLAP_PCT, bp_low, clfmod.BP_HIGH, clfmod.BP_ORDER)
            if X is None:
                continue
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            for name, clf in classifiers.items():
                try:
                    scores = cross_val_score(clf, Xs, y, cv=cv, scoring='accuracy', n_jobs=1)
                    mean_acc = scores.mean()
                    results.append((mean_acc, win, bp_low, name, scores))
                    print(f'win={win}ms bp_low={bp_low} clf={name} acc={mean_acc:.3f} std={scores.std():.3f}')
                except Exception as e:
                    print(f'Error evaluating {name} at win={win}, bp_low={bp_low}: {e}')

    results.sort(reverse=True, key=lambda x: x[0])
    print('\nTop configurations:')
    for acc, win, bp_low, name, scores in results[:6]:
        print(f'  {name} win={win}ms bp_low={bp_low} mean_acc={acc:.3f} scores={np.round(scores,3)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', nargs='+', default=[os.path.abspath(os.path.join(clfmod.SCRIPT_DIR, '..', 'data'))])
    parser.add_argument('--folder-labels', action='store_true')
    args = parser.parse_args()
    evaluate(args.data_dir, folder_labels=args.folder_labels)
