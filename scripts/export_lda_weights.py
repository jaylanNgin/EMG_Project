#!/usr/bin/env python3
"""
Export trained LDA model weights and parameters.
Outputs weights in multiple formats (Python, C-compatible, JSON).
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import scripts.classify_emg_lda as clfmod

def export_weights():
    """Train model and export all weights."""
    
    # Load data
    X_train, y_train = clfmod.load_dataset(['data/act1', 'data/act2'], folder_labels=True)
    X_test, y_test = clfmod.load_dataset(['data/act1_test', 'data/act2_test'], folder_labels=True)
    
    print("=" * 70)
    print("  LDA MODEL WEIGHTS EXPORT")
    print("=" * 70)
    
    # Preprocessing
    scaler = clfmod.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    pca = clfmod.PCA(n_components=4)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # LDA
    le = clfmod.LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    
    lda = clfmod.LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(X_train_pca, y_train_enc)
    
    # ========== SCALER WEIGHTS ==========
    print("\n1. STANDARDSCALER PARAMETERS")
    print("-" * 70)
    print(f"Mean (for 12 original features):")
    for i, m in enumerate(scaler.mean_):
        print(f"  Feature {i:2d}: {m:12.6f}")
    
    print(f"\nStandard Deviation (for 12 original features):")
    for i, s in enumerate(scaler.scale_):
        print(f"  Feature {i:2d}: {s:12.6f}")
    
    # ========== PCA WEIGHTS ==========
    print("\n2. PCA COMPONENTS (Loadings)")
    print("-" * 70)
    print(f"Shape: {pca.components_.shape} (4 components × 12 features)")
    print(f"\nExplained Variance Ratio: {pca.explained_variance_ratio_}")
    print(f"Cumulative Variance: {np.cumsum(pca.explained_variance_ratio_)}")
    
    feature_names = ['MAV_ch1', 'ZC_ch1', 'WL_ch1', 'SSC_ch1',
                     'MAV_ch2', 'ZC_ch2', 'WL_ch2', 'SSC_ch2',
                     'MAV_ch3', 'ZC_ch3', 'WL_ch3', 'SSC_ch3']
    
    for pc_idx, pc in enumerate(pca.components_):
        print(f"\nPC{pc_idx+1} (explains {pca.explained_variance_ratio_[pc_idx]:.1%}):")
        for feat_idx, (fname, weight) in enumerate(zip(feature_names, pc)):
            print(f"  {fname:10s}: {weight:10.6f}")
    
    # ========== LDA WEIGHTS ==========
    print("\n3. LDA CLASSIFICATION WEIGHTS")
    print("-" * 70)
    print(f"Classes: {le.classes_} (encoded as {np.unique(y_train_enc)})")
    print(f"LDA Intercept: {lda.intercept_}")
    print(f"LDA Coef shape: {lda.coef_.shape}")
    print(f"\nLDA Coefficients (PCA → LDA):")
    for i, coef in enumerate(lda.coef_[0]):
        print(f"  PC{i+1}: {coef:10.6f}")
    
    # ========== PREDICTIONS ==========
    print("\n4. MODEL EVALUATION")
    print("-" * 70)
    y_pred = lda.predict(X_test_pca)
    y_test_enc = le.transform(y_test)
    acc = (y_pred == y_test_enc).mean()
    print(f"Test Accuracy: {acc:.1%}")
    
    # ========== C CODE OUTPUT ==========
    print("\n5. C CODE (STM32-compatible)")
    print("-" * 70)
    print("\n// Scaler means")
    print("float scaler_mean[12] = {")
    for i, m in enumerate(scaler.mean_):
        print(f"    {m:.6f}{',' if i < 11 else ''}")
    print("};")
    
    print("\n// Scaler standard deviations")
    print("float scaler_scale[12] = {")
    for i, s in enumerate(scaler.scale_):
        print(f"    {s:.6f}{',' if i < 11 else ''}")
    print("};")
    
    print("\n// PCA components (4x12)")
    print("float pca_components[4][12] = {")
    for i, pc in enumerate(pca.components_):
        print(f"    {{", end="")
        for j, w in enumerate(pc):
            print(f"{w:.6f}{',' if j < 11 else ''}", end="")
        print(f"}}{'' if i == 3 else ','}")
    print("};")
    
    print("\n// LDA coefficients")
    print("float lda_coef[4] = {")
    for i, coef in enumerate(lda.coef_[0]):
        print(f"    {coef:.6f}{',' if i < 3 else ''}")
    print("};")
    
    print("\n// LDA intercept")
    print(f"float lda_intercept = {lda.intercept_[0]:.6f};")
    
    # ========== JSON EXPORT ==========
    weights_dict = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "ResultClipSizeUp900",
        "accuracy": float(acc),
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist()
        },
        "pca": {
            "components": pca.components_.tolist(),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "n_components": int(pca.n_components_)
        },
        "lda": {
            "coef": lda.coef_.tolist(),
            "intercept": lda.intercept_.tolist(),
            "classes": le.classes_.tolist()
        }
    }
    
    json_path = "model_weights.json"
    with open(json_path, 'w') as f:
        json.dump(weights_dict, f, indent=2)
    print(f"\n✓ Weights exported to {json_path}")

if __name__ == '__main__':
    export_weights()
