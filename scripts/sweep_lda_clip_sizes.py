#!/usr/bin/env python3
"""Compare every ResultClipSizeUp* dataset for one EMG subject."""

import argparse
import glob
import os
import re
import sys

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import scripts.classify_emg_lda as emg


def build_dataset(dataset_dir):
    feature_sets = []
    labels = []
    groups = []

    for label in ("act1", "act2"):
        pattern = os.path.join(dataset_dir, label, "*.txt")
        for path in sorted(glob.glob(pattern)):
            raw = emg.load_file(path)
            if raw.size == 0:
                continue
            filtered = emg.bandpass_filter(
                raw,
                emg.SAMPLE_RATE,
                emg.BP_LOW,
                emg.BP_HIGH,
                emg.BP_ORDER,
            )
            centered = filtered - np.mean(filtered, axis=0, keepdims=True)
            features = emg.segment_and_extract(
                centered,
                emg.SAMPLE_RATE,
                emg.WINDOW_MS,
                emg.OVERLAP_PCT,
            )
            if not len(features):
                continue

            feature_sets.append(features)
            labels.extend([label] * len(features))
            trial_name = os.path.splitext(os.path.basename(path))[0]
            groups.extend([trial_name] * len(features))

    if not feature_sets:
        raise ValueError(f"No usable .txt recordings found in {dataset_dir}")

    return np.vstack(feature_sets), np.asarray(labels), np.asarray(groups)


def evaluate(dataset_dir):
    X, y, groups = build_dataset(dataset_dir)
    model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="svd"))

    window_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    window_predictions = cross_val_predict(model, X, y, cv=window_cv)

    grouped_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    grouped_predictions = cross_val_predict(
        model, X, y, groups=groups, cv=grouped_cv
    )

    return (
        len(y),
        accuracy_score(y, window_predictions),
        accuracy_score(y, grouped_predictions),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Find the best ResultClipSizeUp* dataset for LDA."
    )
    parser.add_argument(
        "subject_dir",
        help="Subject directory, for example Subject2GloveOn",
    )
    args = parser.parse_args()

    candidates = []
    pattern = os.path.join(args.subject_dir, "ResultClipSizeUp*")
    for path in glob.glob(pattern):
        match = re.fullmatch(r"ResultClipSizeUp(\d+)", os.path.basename(path))
        if match:
            candidates.append((int(match.group(1)), path))

    if not candidates:
        parser.error(f"No ResultClipSizeUp* directories found in {args.subject_dir}")

    results = []
    print(f"{'Clip size':>9}  {'Windows':>7}  {'Window CV':>10}  {'Trial CV':>9}")
    print("-" * 43)
    for size, path in sorted(candidates):
        windows, window_accuracy, grouped_accuracy = evaluate(path)
        results.append((window_accuracy, grouped_accuracy, windows, size))
        print(
            f"{size:>9}  {windows:>7}  "
            f"{window_accuracy:>9.1%}  {grouped_accuracy:>8.1%}"
        )

    best = max(results, key=lambda row: (row[0], row[1], row[2]))
    print(
        f"\nBest standard CV: ResultClipSizeUp{best[3]} "
        f"({best[0]:.1%}; trial-grouped {best[1]:.1%})"
    )


if __name__ == "__main__":
    main()
