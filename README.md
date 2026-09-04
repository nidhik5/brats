# BraTS 2020 Brain Tumor Segmentation (3D U-Net)

This notebook trains a 3D U-Net to segment brain tumors in MRI scans from the
[BraTS 2020](http://braintumorsegmentation.org/) (Multimodal Brain Tumor
Segmentation) dataset. It covers the full pipeline: loading raw `.nii`
volumes, preprocessing/normalizing them, cropping to a fixed size, building a
train/val split on disk, and training a 3D U-Net with a combined Dice + focal
loss.

## What it does

1. **Load & inspect a single sample** — Reads FLAIR, T1, T1ce, T2, and
   segmentation mask volumes for one patient with `nibabel`, min-max scales
   each MRI channel with `sklearn.preprocessing.MinMaxScaler`, and remaps
   mask label `4` → `3` (BraTS uses labels `{0,1,2,4}`; this collapses them
   to `{0,1,2,3}`).
2. **Visualize** — Plots FLAIR/T1/T2/T1ce slices and the mask side by side to
   sanity-check the data.
3. **Stack & crop** — Stacks T1ce, T2, and FLAIR into a single 3-channel
   volume and crops both the image and mask from `240×240×155` down to
   `128×128×128` (`[56:184, 56:184, 13:141]`) to remove empty border/background.
4. **Batch preprocessing** — Loops over every case in the training set,
   applies the same normalization/crop/relabel steps, one-hot encodes the
   mask into 4 classes, and saves each case as `.npy` image/mask pairs —
   skipping volumes where more than 99% of voxels are background
   (i.e. discarding cases with almost no tumor).
5. **Train/val split** — Uses `splitfolders` to split the saved `.npy` files
   into `train`/`val` directories (75/25).
6. **Data generator** — A custom Python generator (`img_loader`) that lazily
   loads batches of `.npy` image/mask files for training, avoiding loading
   the whole dataset into memory.
7. **Model** — A `simple_unet_model` function defining a standard 3D U-Net
   (4 downsampling/upsampling levels, `Conv3D` + `MaxPooling3D` in the
   encoder, `Conv3DTranspose` + skip connections in the decoder, softmax
   output over 4 classes) built with `keras`/`tensorflow`.
8. **Loss & metrics** — Combines Dice loss and categorical focal loss
   (via `segmentation_models_3D`) with equal class weights, and tracks
   accuracy and IoU during training.
9. **Training** — Compiles and fits the model with the `Adam` optimizer
   (`lr=1e-4`) using the train/val generators, then plots training/validation
   loss and accuracy curves.

## Dataset

- **Source**: [BraTS 2020 Training Data](https://www.med.upenn.edu/cbica/brats2020/data.html)
  (also available on [Kaggle](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)).
- Each patient case includes 4 MRI modalities (`_flair.nii`, `_t1.nii`,
  `_t1ce.nii`, `_t2.nii`) and a segmentation mask (`_seg.nii`).
- The notebook expects the dataset to live locally in Windows-style paths,
  e.g.:
  ```
  E:\BraTS 2020\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_<id>\BraTS20_Training_<id>_<modality>.nii
  ```
  **You will need to update these hardcoded paths** to match your own
  environment (see [Setup](#setup) below).

## Requirements

```
numpy
nibabel
tifffile
scikit-learn
tensorflow
keras
matplotlib
split-folders
segmentation_models_3D
```

Install with:

```bash
pip install numpy nibabel tifffile scikit-learn tensorflow keras matplotlib split-folders segmentation_models_3D
```

(The notebook itself has `!pip install nibabel`, `!pip install tifffile`,
`!pip install split-folders`, and `!pip install segmentation_models_3D`
commented out at the top of the relevant cells — uncomment and run them if
starting from a fresh environment.)

## Setup

1. Download and unzip the BraTS 2020 training data.
2. Update every hardcoded path in the notebook (search for `E:\BraTS 2020` /
   `E:/BraTS 2020`) to point at your local dataset location. Key variables:
   - `train_dataset_directory` — root of the raw `.nii` data
   - `t1_list`, `t2_list`, `t1ce_list`, `flair_list`, `mask_list` — glob
     patterns over the raw data
   - `input_folder` / `output_folder` — where preprocessed `.npy` files are
     written and split
   - `train_img_dir`, `mask_img_dir`, `val_img_dir`, `val_mask_dir` — final
     train/val directories used by the data generators
3. Create the output directories referenced above
   (`input_data_3channels/images`, `input_data_3channels/masks`, etc.)
   before running the batch preprocessing cell, since `np.save` will not
   create missing folders.

## Usage

Run the notebook top to bottom:

1. Cells 0–19: install deps, load one sample case, normalize, and visualize.
2. Cells 20–26: crop a sample to `128×128×128`, save as `.tif`/`.npy`, and
   one-hot encode its mask (exploratory — not required for training).
3. Cells 27–33: batch-preprocess **all** cases into normalized, cropped,
   one-hot-encoded `.npy` files, then split into train/val folders.
4. Cells 34–40: define and sanity-check the batch data generator.
5. Cells 41–46: define the 3D U-Net architecture and instantiate it.
6. Cells 47–50: define the Dice + focal combo loss, IoU metric, and val
   generator.
7. Cell 51: train the model (`model.fit`, default 10 epochs,
   `batch_size=2`).
8. Cell 52: plot training/validation loss and accuracy curves.
