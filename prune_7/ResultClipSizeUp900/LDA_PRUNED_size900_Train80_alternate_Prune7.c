// ============================================================
// Auto-generated LDA model for STM32
// mode: multipletrain_prune:pruned
// dataset: /Users/jaylanngin/Desktop/EMG_Project/prune_7/ResultClipSizeUp900
// source_dataset: /Users/jaylanngin/Desktop/EMG_Project/ResultClipSizeUp900
// sample_size: 900
// split_type: alternate
// train_percent: 80
// test_percent: 20
// train_accuracy: 100.00%
// test_accuracy: 89.71%
// prune_trials_per_label: 7
// removed_trials: {'act2': ['act2/trial_10.txt', 'act2/trial_1.txt', 'act2/trial_4.txt', 'act2/trial_5.txt', 'act2/trial_2.txt', 'act2/trial_6.txt', 'act2/trial_3.txt'], 'act1': ['act1/trial_3.txt', 'act1/trial_1.txt', 'act1/trial_6.txt', 'act1/trial_10.txt', 'act1/trial_2.txt', 'act1/trial_4.txt', 'act1/trial_5.txt']}
// generated: 2026-07-16 10:21:47
// num_class: 2
// feature_dim: 12
// ============================================================

float Wg_init[24] = {
    -1.277428, 1.277507,
    -1.678998, 1.678948,
    0.293708, -0.293726,
    -0.563948, 0.563945,
    -16.696453, 16.696444,
    5.121295, -5.121263,
    1.954980, -1.954984,
    1.213982, -1.214001,
    9.046639, -9.046654,
    -4.453343, 4.453345,
    0.138645, -0.138640,
    -0.307864, 0.307863,
};

float Cg_init[2] = {
    -6.641977, -6.641986
};

float xstd_init[12] = {
    0.014525, 0.015027, 0.288037, 0.249611, 0.002303, 0.000453, 0.037679, 0.126632, 0.001583, 0.000399, 0.051298, 0.133212
};

float xmean_init[12] = {
    0.019881, 0.003476, 0.352941, 0.731765, 0.004397, -0.007522, 0.028235, 0.116471, 0.006223, -0.007137, 0.086274, 0.400000
};

