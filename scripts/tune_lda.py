import argparse
import os
import sys
from typing import Any, Dict

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ensure project root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import scripts.classify_emg_lda as clfmod


def tune(X: np.ndarray, y: np.ndarray, cv_splits: int = 5) -> Dict[str, Any]:
    """
    Tune hyperparameters for the EMG LDA classifier using GridSearchCV.
    """
    # Create a pipeline with scaling, optional PCA, and LDA
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ('lda', LinearDiscriminantAnalysis(solver='lsqr'))
    ])

    # Define the parameter grid to search.
    # We provide a list of two dictionaries to handle the two main cases:
    # 1. No PCA is used (by setting the 'pca' step to 'passthrough').
    # 2. PCA is used, and we search over its number of components.
    param_grid = [
        {
            'pca': ['passthrough'],
            'lda__shrinkage': [None, 'auto', 0.01, 0.05, 0.1, 0.2, 0.5],
        },
        {
            'pca__n_components': [1, 2, 3, 4],
            'lda__shrinkage': [None, 'auto', 0.01, 0.05, 0.1, 0.2, 0.5],
        }
    ]

    # Set up the cross-validation strategy
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    # Set up GridSearchCV to do the heavy lifting
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='accuracy',
        cv=cv,
        verbose=1,
        n_jobs=-1  # Use all available CPU cores
    )

    print("Starting hyperparameter tuning...")
    grid_search.fit(X, y)

    print("\nTuning complete.")
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

    # Print a summary of the top results
    print('\nTop 6 results:')
    results = grid_search.cv_results_
    indices = np.argsort(results['mean_test_score'])[::-1]
    for i in range(min(6, len(indices))):
        idx = indices[i]
        params = results['params'][idx]
        score = results['mean_test_score'][idx]
        std = results['std_test_score'][idx]

        # Format params for readable output
        if params.get('pca') == 'passthrough':
            pca_val = 'None'
        else:
            pca_val = params.get('pca__n_components', 'Default')
        shrink_val = params.get('lda__shrinkage')

        print(f"  Accuracy: {score:.4f} (±{std:.4f}) -> PCA Comp: {pca_val}, Shrinkage: {shrink_val}")

    return grid_search.best_params_


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune hyperparameters for the EMG LDA classifier.")
    parser.add_argument(
        '--data-dir',
        nargs='+',
        default=[os.path.abspath(os.path.join(clfmod.SCRIPT_DIR, '..', 'data'))],
        help="Directory containing labeled .txt EMG files."
    )
    parser.add_argument(
        '--folder-labels',
        action='store_true',
        help="Use the folder name as the class label for all files inside."
    )
    args = parser.parse_args()

    X, y = clfmod.load_dataset(
        args.data_dir,
        prefix_labels=False,
        folder_labels=args.folder_labels
    )
    print(f'\nLoaded X shape: {X.shape}, y shape: {y.shape}')

    best_params = tune(X, y, cv_splits=5)
    print('\nDone. Best parameters:', best_params)


if __name__ == '__main__':
    main()
