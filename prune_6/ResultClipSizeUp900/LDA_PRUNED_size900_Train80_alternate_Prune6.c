// ============================================================
// Auto-generated LDA model for STM32
// mode: multipletrain_prune:pruned
// dataset: /Users/jaylanngin/Desktop/EMG_Project/prune_6/ResultClipSizeUp900
// source_dataset: /Users/jaylanngin/Desktop/EMG_Project/ResultClipSizeUp900
// sample_size: 900
// split_type: alternate
// train_percent: 80
// test_percent: 20
// train_accuracy: 100.00%
// test_accuracy: 92.65%
// prune_trials_per_label: 6
// removed_trials: {'act2': ['act2/trial_10.txt', 'act2/trial_1.txt', 'act2/trial_4.txt', 'act2/trial_5.txt', 'act2/trial_2.txt', 'act2/trial_6.txt'], 'act1': ['act1/trial_3.txt', 'act1/trial_1.txt', 'act1/trial_6.txt', 'act1/trial_10.txt', 'act1/trial_2.txt', 'act1/trial_4.txt']}
// generated: 2026-07-16 10:21:47
// num_class: 2
// feature_dim: 12
// ============================================================

float Wg_init[24] = {
    -1.613323, 1.613371,
    -1.004745, 1.004717,
    0.496913, -0.496916,
    -1.221045, 1.221025,
    -4.690492, 4.690364,
    1.149564, -1.149446,
    1.499546, -1.499524,
    -0.755682, 0.755668,
    4.157544, -4.157500,
    -1.294383, 1.294334,
    -0.781867, 0.781875,
    2.126219, -2.126216,
};

float Cg_init[2] = {
    -4.188962, -4.188942
};

float xstd_init[12] = {
    0.011947, 0.011783, 0.244892, 0.256218, 0.002074, 0.000413, 0.035059, 0.113775, 0.001632, 0.000387, 0.048753, 0.142631
};

float xmean_init[12] = {
    0.016635, 0.000417, 0.298431, 0.659412, 0.004040, -0.007577, 0.023137, 0.095294, 0.005923, -0.007221, 0.080784, 0.352941
};

