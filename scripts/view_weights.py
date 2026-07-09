#!/usr/bin/env python3
"""
Display LDA classifier weights in a readable format.
Shows all weights used by the trained model.
"""

import json
import numpy as np
import os

def display_weights():
    """Load and display model weights from JSON."""
    
    json_file = os.path.expanduser("~/Desktop/EMG_Project/model_weights.json")
    
    if not os.path.exists(json_file):
        print(f"❌ Model weights file not found: {json_file}")
        print("   Run: python3 scripts/export_lda_weights.py")
        return
    
    with open(json_file, 'r') as f:
        weights = json.load(f)
    
    print("\n" + "=" * 80)
    print("  CLASSIFIER WEIGHTS SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 Model Info:")
    print(f"   Dataset: {weights.get('dataset', 'N/A')}")
    print(f"   Generated: {weights.get('timestamp', 'N/A')}")
    print(f"   Test Accuracy: {weights.get('accuracy', 0)*100:.1f}%")
    
    # ========== SCALER ==========
    print(f"\n1️⃣  STANDARDSCALER (Normalization)")
    print("-" * 80)
    scaler = weights.get('scaler', {})
    mean = scaler.get('mean', [])
    scale = scaler.get('scale', [])
    
    features = ['MAV_ch1', 'ZC_ch1', 'WL_ch1', 'SSC_ch1',
                'MAV_ch2', 'ZC_ch2', 'WL_ch2', 'SSC_ch2',
                'MAV_ch3', 'ZC_ch3', 'WL_ch3', 'SSC_ch3']
    
    print(f"{'Feature':<12} {'Mean':<14} {'StdDev':<14}")
    print("-" * 40)
    for i, fname in enumerate(features):
        if i < len(mean) and i < len(scale):
            print(f"{fname:<12} {mean[i]:>12.6f}  {scale[i]:>12.6f}")
    
    # ========== PCA ==========
    print(f"\n2️⃣  PCA (Dimensionality Reduction)")
    print("-" * 80)
    pca = weights.get('pca', {})
    var_ratio = pca.get('explained_variance_ratio', [])
    components = pca.get('components', [])
    
    print(f"Variance Explained per Component:")
    cumsum = 0
    for i, var in enumerate(var_ratio):
        cumsum += var
        print(f"   PC{i+1}: {var*100:5.1f}%  (cumulative: {cumsum*100:5.1f}%)")
    
    print(f"\nPCA Loadings (Components × Features):")
    if components:
        for pc_idx, pc in enumerate(components[:2]):  # Show first 2 for brevity
            print(f"\n   PC{pc_idx+1} contributions:")
            for feat_idx, (fname, weight) in enumerate(zip(features, pc)):
                if weight != 0:  # Only show non-zero
                    print(f"      {fname:<12}: {weight:>10.6f}")
    
    # ========== LDA ==========
    print(f"\n3️⃣  LDA (Classification)")
    print("-" * 80)
    lda = weights.get('lda', {})
    lda_coef = lda.get('coef', [[]])[0] if lda.get('coef') else []
    lda_intercept = lda.get('intercept', [0])[0] if lda.get('intercept') else 0
    classes = lda.get('classes', ['act1', 'act2'])
    
    print(f"Decision Function: decision = Wg @ features + Cg")
    print(f"\nLDA Coefficients (per PCA component):")
    for i, coef in enumerate(lda_coef):
        print(f"   PC{i+1}: {coef:>10.6f}")
    
    print(f"\nLDA Intercept (bias): {lda_intercept:>10.6f}")
    print(f"Classes: {classes}")
    
    # ========== COMBINED WEIGHTS ==========
    print(f"\n4️⃣  COMBINED PIPELINE")
    print("-" * 80)
    print(f"""
    Input: Raw 12-channel features (MAV, ZC, WL, SSC × 3 channels)
           ↓
    Step 1: Normalize each feature
           normalized = (feature - mean) / std
           ↓
    Step 2: Project to PCA space (reduce to 4 components)
           pca_features = normalized @ PCA_loadings
           ↓
    Step 3: LDA decision boundary
           decision = pca_features @ LDA_coef + intercept
           ↓
    Output: Class prediction (act1 if decision > 0, else act2)
    """)
    
    # ========== USAGE ==========
    print(f"\n5️⃣  C CODE ARRAYS (Ready for STM32)")
    print("-" * 80)
    print(f"See: EMG_LDA_Model_STM32.h for C-compatible weight arrays:")
    print(f"   - Wg_init[24]     : Combined weights for all features")
    print(f"   - Cg_init[2]      : Intercepts per class")
    print(f"   - xstd_init[12]   : Normalization scales")
    print(f"   - xmean_init[12]  : Normalization means")
    
    print("\n" + "=" * 80 + "\n")

if __name__ == '__main__':
    display_weights()
