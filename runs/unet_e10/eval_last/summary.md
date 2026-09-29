# Validation results

Checkpoint: `last.pt` (epoch 50). Val n = 92 cases (no background filter); train n = 258.
Metrics computed in the original 240x240x155 grid against full ground truth.
Empty-mask convention: both empty -> Dice 1, HD95 0; exactly one empty -> Dice 0, HD95 373.13 mm.

## Region overlap (all val cases, BraTS convention)

| Region | n | Dice mean ± SD | Dice median [IQR] | HD95 mm mean ± SD | HD95 mm median [IQR] |
|---|---|---|---|---|---|
| WT | 92 | 0.856 ± 0.116 | 0.894 [0.826, 0.925] | 9.279 ± 12.770 | 5.05 [3.16, 10.75] |
| TC | 92 | 0.785 ± 0.233 | 0.887 [0.738, 0.932] | 9.170 ± 16.414 | 4.24 [2.18, 8.32] |
| ET | 92 | 0.697 ± 0.283 | 0.819 [0.638, 0.892] | 30.695 ± 91.546 | 2.24 [1.41, 10.30] |

## Region overlap excluding convention-defined cases

Dice over cases with GT present; HD95 over cases with both GT and prediction present (penalty values excluded, since 373 mm dominates any mean).

| Region | n GT>0 | Dice mean ± SD | Dice median [IQR] | n both present | HD95 mm median [IQR] |
|---|---|---|---|---|---|
| WT | 92 | 0.856 ± 0.116 | 0.894 [0.826, 0.925] | 92 | 5.05 [3.16, 10.75] |
| TC | 92 | 0.785 ± 0.233 | 0.887 [0.738, 0.932] | 92 | 4.24 [2.18, 8.32] |
| ET | 86 | 0.746 ± 0.222 | 0.832 [0.683, 0.901] | 86 | 2.24 [1.41, 7.05] |

## Empty-mask outcomes

| Region | both present | FN: GT>0, pred empty | FP: GT empty, pred>0 | both empty |
|---|---|---|---|---|
| WT | 92 | 0 | 0 | 0 |
| TC | 92 | 0 | 0 | 0 |
| ET | 86 | 0 | 6 | 0 |

## Volume agreement (mL; 1 mm³ voxel = 0.001 mL)

Bias and limits of agreement are pred − GT (Bland–Altman). Relative error only over GT>0.

| Region | n | bias | 95% LoA | MAE | abs err median [IQR] | |rel err| % median [IQR] | Pearson r |
|---|---|---|---|---|---|---|---|
| WT | 92 | +2.16 | [-29.06, +33.38] | 12.09 | 8.89 [4.39, 17.33] | 13.5 [5.8, 21.5] | 0.962 |
| TC | 92 | -0.96 | [-23.63, +21.70] | 7.09 | 2.78 [1.49, 8.98] | 12.1 [5.9, 32.2] | 0.935 |
| ET | 92 | -0.96 | [-11.32, +9.40] | 3.30 | 1.93 [0.67, 4.17] | 13.9 [7.0, 34.1] | 0.958 |

## Dice by ground-truth volume (GT>0 cases)

Spearman ρ between GT volume and Dice quantifies size dependence.

| Region | GT volume | n | Dice median [IQR] | Dice < 0.5 | missed entirely |
|---|---|---|---|---|---|
| WT | (5, 20] mL | 4 | 0.676 [0.567, 0.734] | 25% | 0% |
| WT | (20, 50] mL | 23 | 0.848 [0.796, 0.880] | 4% | 0% |
| WT | > 50 mL | 65 | 0.910 [0.883, 0.929] | 0% | 0% |
| WT | **Spearman ρ = 0.50 (p = 5.2e-07)** | 92 |  |  |  |
| TC | (0, 1] mL | 1 | 0.080 [0.080, 0.080] | 100% | 0% |
| TC | (1, 5] mL | 8 | 0.459 [0.192, 0.711] | 50% | 0% |
| TC | (5, 20] mL | 30 | 0.882 [0.625, 0.915] | 13% | 0% |
| TC | (20, 50] mL | 27 | 0.929 [0.854, 0.944] | 7% | 0% |
| TC | > 50 mL | 26 | 0.894 [0.800, 0.940] | 0% | 0% |
| TC | **Spearman ρ = 0.34 (p = 8.7e-04)** | 92 |  |  |  |
| ET | (0, 1] mL | 4 | 0.353 [0.263, 0.455] | 75% | 0% |
| ET | (1, 5] mL | 16 | 0.582 [0.383, 0.784] | 44% | 0% |
| ET | (5, 20] mL | 39 | 0.833 [0.739, 0.876] | 8% | 0% |
| ET | (20, 50] mL | 19 | 0.911 [0.861, 0.919] | 0% | 0% |
| ET | > 50 mL | 8 | 0.892 [0.814, 0.913] | 0% | 0% |
| ET | **Spearman ρ = 0.58 (p = 6.1e-09)** | 86 |  |  |  |

## By grade (GT>0 cases)

| Region | Grade | n | Dice median [IQR] | GT mL median [IQR] |
|---|---|---|---|---|
| ET | HGG | 73 | 0.855 [0.768, 0.905] | 16.7 [8.6, 27.4] |
| ET | LGG | 13 | 0.419 [0.335, 0.651] | 2.2 [1.3, 3.9] |
| TC | HGG | 73 | 0.903 [0.809, 0.941] | 21.5 [14.7, 49.3] |
| TC | LGG | 19 | 0.741 [0.580, 0.872] | 39.1 [9.5, 88.3] |
| WT | HGG | 73 | 0.896 [0.816, 0.926] | 87.5 [46.3, 126.6] |
| WT | LGG | 19 | 0.888 [0.850, 0.922] | 74.9 [42.8, 141.3] |

## Crop loss on val cases

| Region | n GT>0 | cases with clipped tumor | max fraction clipped | worst-case Dice ceiling |
|---|---|---|---|---|
| WT | 92 | 58 | 6.14e-01 | 0.5569 |
| TC | 92 | 36 | 9.38e-01 | 0.1169 |
| ET | 86 | 29 | 9.51e-01 | 0.0943 |

## Worst 10 cases by WT Dice

| Case | Grade | WT GT mL | WT pred mL | WT Dice | TC Dice | ET Dice | WT HD95 | ET GT mL |
|---|---|---|---|---|---|---|---|---|
| BraTS20_Training_314 | LGG | 27.3 | 33.8 | 0.266 | 0.019 | 0.060 | 100.7 | 2.81 |
| BraTS20_Training_099 | HGG | 7.5 | 31.4 | 0.384 | 0.575 | 0.768 | 52.3 | 1.24 |
| BraTS20_Training_297 | LGG | 74.9 | 88.3 | 0.607 | 0.368 | 0.000 | 35.5 | 0.00 |
| BraTS20_Training_341 | HGG | 12.5 | 26.4 | 0.627 | 0.901 | 0.877 | 23.4 | 7.22 |
| BraTS20_Training_366 | HGG | 46.2 | 32.8 | 0.639 | 0.708 | 0.655 | 23.5 | 19.75 |
| BraTS20_Training_065 | HGG | 25.0 | 48.7 | 0.645 | 0.923 | 0.881 | 10.8 | 2.83 |
| BraTS20_Training_003 | HGG | 29.8 | 50.8 | 0.665 | 0.892 | 0.831 | 8.6 | 3.00 |
| BraTS20_Training_027 | HGG | 66.6 | 130.7 | 0.672 | 0.888 | 0.805 | 11.2 | 16.76 |
| BraTS20_Training_181 | HGG | 66.8 | 44.1 | 0.695 | 0.178 | 0.179 | 19.0 | 17.94 |
| BraTS20_Training_063 | HGG | 13.5 | 21.5 | 0.725 | 0.651 | 0.484 | 23.5 | 2.29 |
