// ============================================================
// Auto-generated LDA model for STM32
// mode: multipletrain_prune:pruned
// dataset: /Users/jaylanngin/Desktop/EMG_Project/prune_1/ResultClipSizeUp900
// source_dataset: /Users/jaylanngin/Desktop/EMG_Project/ResultClipSizeUp900
// sample_size: 900
// split_type: alternate
// train_percent: 80
// test_percent: 20
// train_accuracy: 92.02%
// test_accuracy: 95.59%
// prune_trials_per_label: 1
// removed_trials: {'act2': ['act2/trial_10.txt'], 'act1': ['act1/trial_3.txt']}
// generated: 2026-07-16 10:21:46
// num_class: 2
// feature_dim: 12
// ============================================================

float Wg_init[24] = {
    -1.392509, 1.392564,
    0.482697, -0.482728,
    -0.090268, 0.090245,
    -0.922477, 0.922485,
    -1.763500, 1.763563,
    1.283252, -1.283350,
    0.543665, -0.543648,
    -0.110395, 0.110398,
    0.728958, -0.729042,
    -0.124236, 0.124312,
    0.439062, -0.439059,
    0.665233, -0.665245,
};

float Cg_init[2] = {
    -1.756884, -1.756896
};

float xstd_init[12] = {
    0.010155, 0.008951, 0.203227, 0.262910, 0.002086, 0.000469, 0.042358, 0.109480, 0.001850, 0.000450, 0.057692, 0.161093
};

float xmean_init[12] = {
    0.014886, -0.001407, 0.252101, 0.587731, 0.004225, -0.007496, 0.029132, 0.100840, 0.005673, -0.007276, 0.067451, 0.304874
};

