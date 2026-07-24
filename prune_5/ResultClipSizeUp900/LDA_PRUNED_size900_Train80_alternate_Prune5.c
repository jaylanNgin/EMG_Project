// ============================================================
// Auto-generated LDA model for STM32
// mode: multipletrain_prune:pruned
// dataset: /Users/jaylanngin/Desktop/EMG_Project/prune_5/ResultClipSizeUp900
// source_dataset: /Users/jaylanngin/Desktop/EMG_Project/ResultClipSizeUp900
// sample_size: 900
// split_type: alternate
// train_percent: 80
// test_percent: 20
// train_accuracy: 99.02%
// test_accuracy: 95.59%
// prune_trials_per_label: 5
// removed_trials: {'act2': ['act2/trial_10.txt', 'act2/trial_1.txt', 'act2/trial_4.txt', 'act2/trial_5.txt', 'act2/trial_2.txt'], 'act1': ['act1/trial_3.txt', 'act1/trial_1.txt', 'act1/trial_6.txt', 'act1/trial_10.txt', 'act1/trial_2.txt']}
// generated: 2026-07-16 10:21:47
// num_class: 2
// feature_dim: 12
// ============================================================

float Wg_init[24] = {
    -1.550443, 1.550450,
    -0.850617, 0.850617,
    0.831066, -0.831050,
    -1.476800, 1.476789,
    -2.097632, 2.097449,
    0.367596, -0.367397,
    0.986152, -0.986165,
    -1.707056, 1.707045,
    3.277276, -3.277284,
    -0.796502, 0.796509,
    -0.821096, 0.821111,
    2.093906, -2.093922,
};

float Cg_init[2] = {
    -3.819486, -3.819490
};

float xstd_init[12] = {
    0.012032, 0.011452, 0.234697, 0.261055, 0.002201, 0.000437, 0.037654, 0.110442, 0.001606, 0.000385, 0.052996, 0.146556
};

float xmean_init[12] = {
    0.016129, 0.000035, 0.279738, 0.627451, 0.004141, -0.007560, 0.024575, 0.092549, 0.005967, -0.007217, 0.080784, 0.345098
};

