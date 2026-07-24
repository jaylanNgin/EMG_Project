// ============================================================
// Auto-generated LDA model for STM32
// mode: multipletrain_prune:pruned
// dataset: /Users/jaylanngin/Desktop/EMG_Project/prune_2/ResultClipSizeUp900
// source_dataset: /Users/jaylanngin/Desktop/EMG_Project/ResultClipSizeUp900
// sample_size: 900
// split_type: alternate
// train_percent: 80
// test_percent: 20
// train_accuracy: 92.65%
// test_accuracy: 92.65%
// prune_trials_per_label: 2
// removed_trials: {'act2': ['act2/trial_10.txt', 'act2/trial_1.txt'], 'act1': ['act1/trial_3.txt', 'act1/trial_1.txt']}
// generated: 2026-07-16 10:21:47
// num_class: 2
// feature_dim: 12
// ============================================================

float Wg_init[24] = {
    -1.527399, 1.527398,
    0.504682, -0.504677,
    -0.030362, 0.030384,
    -1.140960, 1.140954,
    -2.184589, 2.184399,
    1.672425, -1.672196,
    0.875977, -0.875985,
    -0.100745, 0.100703,
    1.076337, -1.076356,
    -0.295471, 0.295510,
    0.307151, -0.307140,
    0.874393, -0.874400,
};

float Cg_init[2] = {
    -1.865017, -1.864997
};

float xstd_init[12] = {
    0.010537, 0.009385, 0.214632, 0.277625, 0.002080, 0.000465, 0.041835, 0.109173, 0.001893, 0.000457, 0.059861, 0.165827
};

float xmean_init[12] = {
    0.014680, -0.001375, 0.250458, 0.580392, 0.004276, -0.007489, 0.029542, 0.104706, 0.005595, -0.007292, 0.067189, 0.300981
};

