// ============================================================
// Auto-generated LDA model for STM32
// mode: multipletrain_prune:pruned
// dataset: /Users/jaylanngin/Desktop/EMG_Project/prune_4/ResultClipSizeUp900
// source_dataset: /Users/jaylanngin/Desktop/EMG_Project/ResultClipSizeUp900
// sample_size: 900
// split_type: alternate
// train_percent: 80
// test_percent: 20
// train_accuracy: 96.32%
// test_accuracy: 95.59%
// prune_trials_per_label: 4
// removed_trials: {'act2': ['act2/trial_10.txt', 'act2/trial_1.txt', 'act2/trial_4.txt', 'act2/trial_5.txt'], 'act1': ['act1/trial_3.txt', 'act1/trial_1.txt', 'act1/trial_6.txt', 'act1/trial_10.txt']}
// generated: 2026-07-16 10:21:47
// num_class: 2
// feature_dim: 12
// ============================================================

float Wg_init[24] = {
    -0.646911, 0.646903,
    -0.817317, 0.817325,
    -0.253067, 0.253105,
    -0.683080, 0.683072,
    -2.251760, 2.251498,
    0.950668, -0.950376,
    0.547844, -0.547850,
    -1.055020, 1.054973,
    1.845782, -1.845805,
    -0.419745, 0.419785,
    0.051360, -0.051349,
    1.610191, -1.610205,
};

float Cg_init[2] = {
    -3.093009, -3.093004
};

float xstd_init[12] = {
    0.011497, 0.010873, 0.226809, 0.253249, 0.002092, 0.000422, 0.037564, 0.108757, 0.001687, 0.000452, 0.058999, 0.145159
};

float xmean_init[12] = {
    0.016159, 0.000123, 0.279804, 0.622647, 0.004192, -0.007544, 0.025882, 0.100000, 0.006065, -0.007191, 0.080980, 0.341177
};

