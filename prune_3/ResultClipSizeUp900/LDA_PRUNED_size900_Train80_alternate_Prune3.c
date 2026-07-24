// ============================================================
// Auto-generated LDA model for STM32
// mode: multipletrain_prune:pruned
// dataset: /Users/jaylanngin/Desktop/EMG_Project/prune_3/ResultClipSizeUp900
// source_dataset: /Users/jaylanngin/Desktop/EMG_Project/ResultClipSizeUp900
// sample_size: 900
// split_type: alternate
// train_percent: 80
// test_percent: 20
// train_accuracy: 93.53%
// test_accuracy: 92.65%
// prune_trials_per_label: 3
// removed_trials: {'act2': ['act2/trial_10.txt', 'act2/trial_1.txt', 'act2/trial_4.txt'], 'act1': ['act1/trial_3.txt', 'act1/trial_1.txt', 'act1/trial_6.txt']}
// generated: 2026-07-16 10:21:47
// num_class: 2
// feature_dim: 12
// ============================================================

float Wg_init[24] = {
    -1.644588, 1.644598,
    0.628241, -0.628245,
    0.451532, -0.451506,
    -2.144386, 2.144378,
    -2.445107, 2.444953,
    1.931973, -1.931782,
    0.745395, -0.745406,
    -0.165262, 0.165220,
    0.882766, -0.882799,
    -0.161515, 0.161563,
    0.294847, -0.294838,
    1.024407, -1.024415,
};

float Cg_init[2] = {
    -2.119649, -2.119633
};

float xstd_init[12] = {
    0.011272, 0.010070, 0.226027, 0.292623, 0.002141, 0.000482, 0.044037, 0.112898, 0.001914, 0.000466, 0.060305, 0.167888
};

float xmean_init[12] = {
    0.015143, -0.000808, 0.256941, 0.580471, 0.004522, -0.007444, 0.034353, 0.117412, 0.005593, -0.007286, 0.067608, 0.296471
};

