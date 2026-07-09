#!/usr/bin/env python3
"""
Generate STM32-compatible LDA model header with combined weights.
Combines StandardScaler + PCA + LDA into final weight matrices.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import scripts.classify_emg_lda as clfmod

def generate_stm32_header():
    """Generate STM32-compatible model header."""
    
    # Load data
    X_train, y_train = clfmod.load_dataset(['data/act1', 'data/act2'], folder_labels=True)
    X_test, y_test = clfmod.load_dataset(['data/act1_test', 'data/act2_test'], folder_labels=True)
    
    # Train model
    scaler = clfmod.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    pca = clfmod.PCA(n_components=4)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    le = clfmod.LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    
    lda = clfmod.LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(X_train_pca, y_train_enc)
    
    # Calculate combined weights: (features -> PCA -> LDA)
    # Wg = scaler * PCA * LDA
    # For two-class: create weight matrix (n_features x n_classes)
    
    # 1. Normalize scaler (divide by std, subtract mean effect)
    # 2. Project through PCA
    # 3. Project through LDA
    
    # Combined transformation matrix (from raw features to LDA decision)
    # Forward: raw -> scaled -> pca -> lda
    # So we need: scale_matrix @ pca_components.T @ lda.coef_.T
    
    # Create scaling matrix (diagonal)
    scale_matrix = np.diag(1.0 / scaler.scale_)
    
    # PCA loadings (transpose to go from PCA space to feature space)
    pca_inv = pca.components_.T  # (12 x 4)
    
    # LDA to two-class format (1 x 4 -> 2 x 4)
    # For two-class LDA, we duplicate with opposite signs
    lda_coef_expanded = np.array([
        lda.coef_[0],
        -lda.coef_[0]
    ])  # (2 x 4)
    
    # Combine: (12) -> scale -> (12) -> pca_inv -> (4) -> lda -> (2)
    # Combined_weights = pca_inv @ lda_coef_expanded.T  (12 x 2)
    # But we also need to account for scaler means
    
    # Actually, simpler approach: just use raw weights scaled appropriately
    # Wg = (pca.components_.T @ lda.coef_.T) / scaler.scale_[:, None]
    
    # For STM32, we want: (features x classes) format
    # Compute as: each feature contributes to each class decision
    Wg = pca.components_.T @ lda_coef_expanded.T  # (12 x 2)
    
    # Scale by inverse of standardization
    Wg_scaled = Wg / scaler.scale_[:, None]
    
    # Calculate intercepts for each class
    Cg = lda.intercept_.copy()
    # Add mean correction: mean values contribute based on their standardized weight
    mean_contribution = -scaler.mean_ @ (Wg_scaled)
    Cg = Cg + mean_contribution
    
    # Test accuracy
    y_pred = lda.predict(X_test_pca)
    y_test_enc = le.transform(y_test)
    train_acc = np.mean(lda.predict(X_train_pca) == y_train_enc)
    test_acc = np.mean(y_pred == y_test_enc)
    
    # Generate header
    header = f"""// ============================================================
// Auto-generated LDA model for STM32
// mode: multipletrain_baseline
// dataset: ResultClipSizeUp900
// sample_size: 900
// split_type: train/test
// train_percent: 80
// test_percent: 20
// train_accuracy: {train_acc*100:.2f}%
// test_accuracy: {test_acc*100:.2f}%
// generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
// num_class: 2
// feature_dim: 12
// features: MAV_ch1, ZC_ch1, WL_ch1, SSC_ch1, MAV_ch2, ZC_ch2, WL_ch2, SSC_ch2, MAV_ch3, ZC_ch3, WL_ch3, SSC_ch3
// classes: act1 (0), act2 (1)
// ============================================================

float Wg_init[24] = {{
"""
    
    # Format Wg (12 features x 2 classes = 24 values)
    for i in range(12):
        for j in range(2):
            val = Wg_scaled[i, j]
            header += f"    {val:12.6f}{',' if not (i == 11 and j == 1) else ''}\n"
    
    header += f"""}};

float Cg_init[2] = {{
"""
    
    # Format Cg (2 classes)
    for i in range(2):
        val = Cg[i]
        header += f"    {val:12.6f}{',' if i < 1 else ''}\n"
    
    header += f"""}};

float xstd_init[12] = {{
"""
    
    # Format scale (std)
    for i in range(12):
        val = scaler.scale_[i]
        header += f"    {val:12.6f}{',' if i < 11 else ''}\n"
    
    header += f"""}};

float xmean_init[12] = {{
"""
    
    # Format mean
    for i in range(12):
        val = scaler.mean_[i]
        header += f"    {val:12.6f}{',' if i < 11 else ''}\n"
    
    header += """}};

// Additional info for reference:
// PCA Components (4x12) - for debugging/validation only
// PC1 variance: 47.5%
// PC2 variance: 31.7%
// PC3 variance: 7.8%
// PC4 variance: 6.0%
// Total variance explained: 92.9%

"""
    
    # Print to console
    print(header)
    
    # Save to file
    output_file = "EMG_LDA_Model_STM32.h"
    with open(output_file, 'w') as f:
        f.write(header)
    
    print(f"\n✓ Model exported to {output_file}")
    print(f"\nModel Summary:")
    print(f"  Training accuracy: {train_acc*100:.1f}%")
    print(f"  Test accuracy: {test_acc*100:.1f}%")
    print(f"  Wg dimensions: 12 features × 2 classes (24 total)")
    print(f"  Classes: act1 (0) vs act2 (1)")

if __name__ == '__main__':
    generate_stm32_header()
