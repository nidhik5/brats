# Where a 3D brain-tumor segmentation model fails, and how that was measured

A 3D U-Net trained on BraTS 2020 multimodal MRI and evaluated with BraTS-standard
region metrics. The evaluation focuses on **failure structure**: how accuracy depends
on tumor size, how accurate the predicted volumes are, how empty masks are
handled, and how much tumor the preprocessing silently discards.

This is a rewrite of an earlier Keras notebook (kept in [legacy/](legacy/)). The
original only tracked voxel accuracy and IoU on a validation set that was filtered
the same way as the training set.

## Pipeline

| Step | Script | Notes |
|---|---|---|
| Preprocess | `prepare_data.py` | Loads T1ce, T2 and FLAIR, applies a per-volume z-score over brain voxels, remaps label 4→3 and crops to `[56:184, 56:184, 13:141]` (128³). Fixes the misnamed case-355 seg file in the Kaggle copy. Measures how much tumor each case loses to the crop. |
| Split | `prepare_data.py` | Patient-level 75/25 split, stratified by grade (HGG/LGG), seed 42. The >99%-background filter is applied to **training only**. Validation keeps every case. |
| Train | `train.py` | 3D U-Net (16→256 filters, 4 levels, InstanceNorm; `--norm none` gives the original architecture), Dice + focal loss, Adam 1e-4, random flips, fp16 autocast. Saves a resumable checkpoint every epoch. |
| Evaluate | `evaluate.py` | Predicts on the crop and pastes the result back into the full 240×240×155 grid. Scores against the **uncropped** ground truth, so clipped tumor counts as missed. |

## Evaluation protocol

- **Regions** (after 4→3): WT = {1,2,3}, TC = {1,3}, ET = {3}.
- **Dice** per case and region.
- **HD95**: the 95th percentile of pooled symmetric surface distances in mm, where surface = mask minus its 6-connected erosion (same as `medpy.hd95`). It is checked against a brute-force implementation in `tests/`.
- **Empty masks** (BraTS convention; the tables report both views):

  | GT | Pred | Dice | HD95 |
  |---|---|---|---|
  | empty | empty | 1 | 0 |
  | empty | non-empty | 0 | 373.13 mm (volume diagonal) |
  | non-empty | empty | 0 | 373.13 mm |

  Summary tables give statistics both with these cases included and restricted to cases where GT (for Dice) or both masks (for HD95) are present. A single 373 mm penalty otherwise dominates the HD95 mean.
- **Volume**: voxels × spacing product / 1000, in mL (1 mm³ in BraTS). Reports Bland–Altman bias and 95% limits of agreement, absolute error, and relative error (GT > 0 only).
- **Size dependence**: Dice vs GT volume, a Spearman ρ, and Dice binned at 1 / 5 / 20 / 50 mL.
- **Checkpoint**: evaluation uses the final-epoch weights (`--weights last`) by default, so the reported set is not also used for model selection.

## Results

Run `runs/unet_e10`: trained 10 epochs, then resumed to **50 epochs** in the same directory. Uses
last-epoch weights (epoch 50, which is also the lowest val loss, so `best.pt` = `last.pt`), with
258 train and 92 val cases (val unfiltered: 73 HGG, 19 LGG). All metrics are computed in the full
240×240×155 grid. The full tables are in `runs/unet_e10/eval_last/summary.md`. The figures are
`dice_hd95_vs_volume.png`, `bland_altman.png` and `volume_scatter.png`; the worst cases are in `gallery/`.

| Region | Dice median [IQR] | Dice mean ± SD | HD95 mm median [IQR] | Volume bias mL [95% LoA] | Pearson r (volume) |
|---|---|---|---|---|---|
| WT | 0.894 [0.826, 0.925] | 0.856 ± 0.116 | 5.1 [3.2, 10.8] | +2.2 [−29.1, +33.4] | 0.962 |
| TC | 0.887 [0.738, 0.932] | 0.785 ± 0.233 | 4.2 [2.2, 8.3] | −1.0 [−23.6, +21.7] | 0.935 |
| ET | 0.819 [0.638, 0.892] | 0.697 ± 0.283 | 2.2 [1.4, 10.3] | −1.0 [−11.3, +9.4] | 0.958 |

These numbers include the BraTS empty-mask convention. On the 86 cases where GT contains enhancing
tumor, ET Dice is 0.746 ± 0.222 (median 0.832). The mean ET HD95 is 30.7 mm with the convention and
6.8 mm without it: six 373 mm penalties make up most of the mean.

**Training curve** (`history.png`, `history.csv`; Dice is on the cropped val grid):
- Val Dice rose quickly until about epoch 30 and was nearly flat after epoch 35:
  - WT: 0.77 at epoch 10, 0.86 at 30, 0.87–0.89 from 35 to 50.
  - ET: 0.58 at epoch 10, 0.69 at 30, 0.71–0.72 from 35 to 50.
- The train−val gap stayed small (WT 0.91 vs 0.87; ET 0.80 vs 0.72), so the model is near its
  capacity for this setup, not badly overfit.
- Each epoch took about 1.2 min on an RTX 3050 6 GB, so the 50 epochs took about 1 hour.

**Change from 10 epochs** (same val cases, same protocol):

| | WT | TC | ET |
|---|---|---|---|
| Median Dice | 0.820 → 0.894 | 0.720 → 0.887 | 0.658 → 0.819 |
| Median HD95 (mm) | 9.4 → 5.1 | 13.8 → 4.2 | 13.6 → 2.2 |
| Volume bias (mL) | +22.0 → +2.2 | +4.2 → −1.0 | +5.4 → −1.0 |
| Spearman ρ (Dice vs GT volume) | 0.76 → 0.50 | 0.56 → 0.34 | 0.77 → 0.58 |

## Failure analysis

**1. Accuracy still depends on tumor size, but less than it did.** Spearman ρ between GT volume
and Dice is now 0.50 for WT, 0.34 for TC and 0.58 for ET (all p < 1e-3). At 10 epochs it was 0.76,
0.56 and 0.77.
- ET: 3 of the 4 cases ≤ 1 mL are below 0.5 (median 0.35). Of the 16 cases between 1 and 5 mL,
  44% are below 0.5 (median 0.58). From 20 mL up, the median is about 0.9.
- TC: 5 of the 9 cases ≤ 5 mL are below 0.5.
- WT: 1 of the 4 cases under 20 mL is below 0.5.
- The model still never returned an empty mask where GT has the region (0 FN-empty outcomes).
  Small lesions are found but outlined imprecisely.

**2. ET is predicted in every case that has none, and a size threshold can't fix it.**
- All 6 ET-absent val cases get an ET prediction. All 6 are LGG: 262, 297, 305, 306, 319 and 330.
- The false-positive volumes are much smaller than at 10 epochs. They are 0.4, 0.6, 1.2, 4.5, 4.8
  and 13.2 mL; at 10 epochs, case 297 alone had 28 mL.
- They still cost Dice 0 and 373 mm HD95 each. That lowers the mean ET Dice from 0.746 to 0.697 and
  raises the mean ET HD95 from 6.8 to 30.7 mm.
- The usual BraTS post-processing is to zero ET when the predicted ET volume is below a threshold.
  Sweeping that threshold on these cases shows it can't separate the two groups:

  | Threshold | False positives removed (of 6) | True-ET cases zeroed |
  |---|---|---|
  | 0.5 mL | 1 | 3 |
  | 1 mL | 2 | 4 (incl. 311 at Dice 0.75) |
  | 2 mL | 3 | 8 |
  | 5 mL | 5 | 20 |

  The reason is that true small ET (0.1–2 mL, mostly LGG) and spurious ET fall in the same volume
  range. Deciding whether ET is present needs a case-level signal, such as grade or a presence
  classifier, not a voxel count.

**3. Volume bias is almost gone on average but is still proportional.**
- Mean bias is +2.2 mL for WT and about −1 mL for TC and ET. The limits of agreement halved
  (WT ±31 mL, from ±53).
- Error still trends with size (r between mean volume and error ≈ −0.35 in every region).
  Tumors below the median WT volume are over-called by +8.6 mL on average; larger ones are
  under-called by −4.3 mL.
- WT is over-segmented in 65% of cases. TC and ET lean the other way (39–40% over).
- The median absolute relative volume error is 12–14% in all three regions.

**4. The worst WT cases are either false positives or crop loss.**
- In 8 of the 10 worst WT cases, predicted volume exceeds GT:
  - 099: GT 7.5 mL, predicted 31.4 mL. It was 115 mL at 10 epochs.
  - 027: GT 66.6 mL, predicted 130.7 mL, with no crop loss. This is pure over-segmentation.
- The two under-calls, 366 and 181, both lose more than 40% of their WT to the crop (item 5).
- Case 314 has both problems. It loses 61% of WT to the crop and also has a spurious core-labelled
  blob, giving WT Dice 0.27.

**5. The fixed crop is now the binding limit for several cases.**
- 58/92 val cases lose some WT to the 128³ crop; 40 of those lose less than 5%.
- 7 cases lose at least 20% of WT: 314, 366, 181, 361, 232, 297 and 221.
- The largest losses are in TC. Crop loss sets the best Dice the model could possibly reach
  (ceiling = 2(1−f)/(2−f)):

  | Case | TC lost to crop | Dice ceiling | TC Dice |
  |---|---|---|---|
  | 314 | 94% | 0.12 | 0.02 |
  | 181 | 89% | 0.20 | 0.18 |
  | 232 | 64% | 0.52 | 0.23 |
  | 297 | 38% | 0.77 | 0.37 |
  | 366 | 39% | 0.76 | 0.71 |

- Case 181 reaches 90% of its ceiling and 366 reaches 93%. For these cases more training can't
  help; only a different input window can.
- The same holds for WT: 181 scores 0.69 against a ceiling of 0.74, and 366 scores 0.64 against
  0.72.
- Brain-centred cropping or sliding-window inference over the full volume would remove this cap.

**6. Grade: the LGG gap is now almost entirely ET.**
- WT: the grades are equal (median Dice 0.888 LGG vs 0.896 HGG).
- TC: LGG is still lower (0.741 vs 0.903) even though LGG cores are *larger* (39.1 vs 21.5 mL).
  The core in LGG is mostly non-enhancing and hard to separate from edema without T1.
- ET: on the 13 LGG cases with ET, the median is 0.42 vs 0.86 for HGG, driven by volume (2.2 vs
  16.7 mL, finding 1).
- ET presence: 6 of the 19 LGG cases have no ET, and every one of them gets a false positive
  (finding 2).
- LGG is only 19/92 val cases, so these gaps have wide uncertainty.

**Next experiments, in order of expected impact:**
1. Replace the fixed crop with brain-bbox or sliding-window inference. This lifts the hard ceiling
   in item 5 and costs no training.
2. Decide ET presence at the case level, with a small classifier head or a rule conditioned on
   predicted core composition, tuned on train folds. A volume threshold won't work (item 2).
3. Add T1 as a fourth channel, aimed at the LGG TC gap.
4. Run 5-fold cross-validation with more than one seed to get confidence intervals. At 92 cases,
   LGG subgroup numbers are noisy.

## Limitations

- **Random split mixes institutions.** BraTS 2020 pools scans from multiple sites and does not release site labels, so train and val share scanners and protocols. The numbers are in-distribution and likely optimistic for a new site.
- **Validation doubles as the test set.** BraTS validation labels are not public. Checkpoint selection is avoided by evaluating the last epoch, but hyperparameters were not tuned on a held-out set.
- **Fixed crop.** The 128³ window is not adaptive to the brain or tumor. Clipped tumor is measured and scored as missed, not ignored.
- **Three of four modalities.** T1 is dropped (input is T1ce, T2, FLAIR), inherited from the original project.
- **Single run.** 50 epochs, a single seed and split, no ensembling or test-time augmentation, no post-processing (e.g. small-ET-component removal, which BraTS entrants use to avoid empty-ET penalties).

## Setup (Windows, local NVIDIA GPU)

PowerShell. Adjust `D:\data\...` to wherever you have about 20 GB free.

```powershell
conda create -n brats python=3.11 -y
conda activate brats
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())"
```

Data (put `kaggle.json` from kaggle.com → Settings → API in `%USERPROFILE%\.kaggle\`):

```powershell
kaggle datasets download -d awsaf49/brats20-dataset-training-validation -p D:\data\brats2020 --unzip
```

Run:

```powershell
python -m pytest tests -q
python prepare_data.py --data-root D:\data\brats2020 --work-dir D:\data\brats_work
python train.py --work-dir D:\data\brats_work --run-dir runs\unet_e10 --epochs 10
python train.py --work-dir D:\data\brats_work --run-dir runs\unet_e10 --resume --epochs 50
python evaluate.py --work-dir D:\data\brats_work --run-dir runs\unet_e10
```

- **Out of memory** on a 6 GB GPU: add `--batch-size 1`.
- **Resume** after an interruption, or extend training: `--resume --epochs 20`.
- **Re-plot** without re-predicting: `python evaluate.py ... --skip-predict`.
- **View predictions** in ITK-SNAP: `--save-nifti`. The files use BraTS labels, with 3 mapped back to 4.
