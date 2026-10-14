# DualModal_TFN

This repository contains the implementation of **DualModal_TFN**, a dual-modal extension of the Time-Frequency Network (TFN) for mechanical fault diagnosis using **vibration** and **acoustic** signals.

The project follows the main idea of TFN: a physically meaningful time-frequency preprocessing layer is embedded into a CNN and trained jointly with the downstream classifier. In `DualModal_TFN`, two independent TFN branches are used to extract vibration and acoustic features, followed by shallow/deep feature fusion, adaptive feature weighting, and optional decision-level fusion.

For noisy experiments, the current v5 implementation can additionally enable an **internal learnable two-modal spectral branch**. This branch is part of the PyTorch model itself: it extracts pooled FFT magnitude, low-order cepstral coefficients, and short-lag autocorrelation features from both modalities, receives a train-split-only warm start, and is then trained jointly with the TFN branches. No external SVM/teacher is required at validation or test time.

---

## Notes

- The code supports two datasets: **University of Ottawa constant load and speed rolling-element bearing vibration and acoustic fault signature datasets** and **BJTU-RAO Bogie Datasets**.
- The main model is `DualModal_TFN`; single-modal baselines `Backbone_CNN`, `Random_CNN`, and `TFN_Gabor` are also included.
- `DualModal_TFN` requires two-channel input with:
  - channel 0: vibration;
  - channel 1: acoustic.
- With `--dual_gabor_mode auto`:
  - noisy runs use the faster **1D Gabor** front-end;
  - clean runs use the **2D STFT-Gabor** front-end.
- In equal-SNR noisy runs, the code can apply AWGN **after normalization and independently for each modality**, avoiding distortion caused by different vibration/acoustic signal scales.
- The v5 low-SNR path can use an internal spectral branch and EMA model selection while keeping final inference inside one `DualModal_TFN` model.

---

## Repo structure

The core model implementation is mainly located in `Models/TFconvlayer.py` and `Models/TFN.py`. If you only care about the architecture, these are the most important files.

```text
.
├── Datasets
│   ├── UORED.py
│   ├── BJTU_RAO.py
│   ├── Dataset_utils
│   │   ├── DatasetsBase.py
│   │   └── sequence_aug.py
│   └── get_files
│       └── generalfunciton.py
├── Models
│   ├── BackboneCNN.py
│   ├── TFconvlayer.py
│   ├── TFN.py
│   └── __init__.py
├── PostProcess
│   ├── TrainSequentially.py
│   ├── Acc_statistic.py
│   └── process_utils
├── tools
│   ├── compare_fixed_vs_learnable.py
│   ├── envcheck.py
│   └── frequency_analysis.py
├── utils
│   ├── logger.py
│   ├── mysummary.py
│   └── train_utils.py
└── main.py
```

- `Datasets`: dataset loading, windowing, normalization, augmentation, SNR noise injection, and train/validation/test splitting.
- `Models/TFconvlayer.py`: learnable 1D and 2D Gabor time-frequency convolution layers.
- `Models/TFN.py`: `TFN_Gabor`, `DualModal_TFN`, fusion modules, and the internal spectral head.
- `Models/BackboneCNN.py`: conventional 1D CNN baseline.
- `utils/train_utils.py`: complete training/validation/test pipeline, optimizer setup, EMA, logging, checkpoint handling, metrics, and visualization.
- `PostProcess`: sequential experiments and accuracy statistics/plotting.
- `tools`: environment check, frequency-domain analysis, and learnable-vs-fixed Gabor ablation.
- `main.py`: main entry point.

---

## Model overview

### Single-modal TFN

`TFN_Gabor` replaces the first conventional convolution with a learnable 1D Gabor filter bank. Each filter is parameterized mainly by:

- `f0`: normalized center frequency;
- `sigma`: Gaussian window width;
- `theta`: retained as a learnable parameter where applicable.

The Gabor output is passed to the CNN backbone for fault classification.

### DualModal_TFN

`DualModal_TFN` contains two independent TFN branches:

```text
Vibration ── Gabor-TFN branch ──┐
                                ├─ shallow fusion ──┐
Acoustic  ── Gabor-TFN branch ──┘                  │
                                                   ├─ adaptive weighting ─ classifier
Vibration feature ─ deep extractor ─┐              │
                                    ├─ deep fusion ┘
Acoustic feature  ─ deep extractor ─┘
```

Each branch produces a 64-dimensional feature vector. The model then performs:

1. channel-attention-based shallow feature fusion;
2. independent deep feature extraction followed by a second fusion stage;
3. concatenation of shallow and deep fused features;
4. adaptive feature weighting;
5. final classification.

When `--use_decision_fusion True` is enabled, the two unimodal branch logits are also added to the main logits with weight `--decision_alpha`.

For noisy equal-SNR experiments, the v5 model can additionally enable an internal spectral branch:

```text
Vibration ─┐
           ├─ FFT / cepstrum / autocorrelation features ─ MLP ─ spectral logits
Acoustic ──┘

TFN fused logits ─┐
                  ├─ learnable convex fusion ─ final logits
Spectral logits ──┘
```

---

# How to run?

## Dataset

### 1. UORED

The UORED loader expects the following directory structure:

```text
<UORED_ROOT>/
└── 1_CSV_Raw_Data_Files (.csv)/
    ├── 1_Healthy/
    │   └── *.csv
    ├── 2_Inner_Race_Faults/
    │   └── *.csv
    ├── 3_Outer_Race_Faults/
    │   └── *.csv
    ├── 4_Ball_Faults/
    │   └── *.csv
    └── 5_Cage_Faults/
        └── *.csv
```

Each CSV file must contain the columns:

```text
Accelerometer
Acoustic
```

The loader stacks them into a sample of shape `(2, L)`:

```text
channel 0 = Accelerometer / vibration
channel 1 = Acoustic
```

The default signal length is `2048`. Samples are generated with non-overlapping windows. Classes are balanced to the size of the smallest class before splitting.

The current UORED code performs a grouped deterministic split. Within each group, approximately **72% / 8% / 20%** of samples become train / validation / test data.

A cache file named `data_buffer_uored.pkl` is automatically created inside the dataset directory.

---

### 2. BJTU-RAO

The BJTU-RAO loader uses five left-axle-box conditions:

```text
M0_G0_LA0_RA0
M0_G0_LA1_RA0
M0_G0_LA2_RA0
M0_G0_LA3_RA0
M0_G0_LA4_RA0
```

For each class, the loader reads exactly:

```text
<BJTU_ROOT>/
├── M0_G0_LA0_RA0/
│   └── Sample_1/
│       └── data_leftaxlebox_M0_G0_LA0_RA0_20Hz_0kN.csv
├── M0_G0_LA1_RA0/
│   └── Sample_1/
│       └── data_leftaxlebox_M0_G0_LA1_RA0_20Hz_0kN.csv
├── M0_G0_LA2_RA0/
│   └── Sample_1/
│       └── data_leftaxlebox_M0_G0_LA2_RA0_20Hz_0kN.csv
├── M0_G0_LA3_RA0/
│   └── Sample_1/
│       └── data_leftaxlebox_M0_G0_LA3_RA0_20Hz_0kN.csv
└── M0_G0_LA4_RA0/
    └── Sample_1/
        └── data_leftaxlebox_M0_G0_LA4_RA0_20Hz_0kN.csv
```

The selected modalities are:

```text
CH17 = vibration acceleration
CH20 = sound intensity
```

If explicit `CH17` and `CH20` headers are unavailable, the loader only falls back when exactly four valid numeric columns exist, interpreting them as `CH17, CH18, CH19, CH20`.

Samples use non-overlapping windows. Within each class, chronological windows are split as:

```text
70% train
10% validation
20% test
```

Only the training split is shuffled after the chronological boundaries are fixed.

A cache file named `data_buffer_bjtu_Sample1_CH17_CH20_v2.pkl` is automatically created inside the dataset directory.

---

## Requirement

The archive does not include a pinned `requirements.txt`. The code directly uses the following packages:

```text
torch
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
openpyxl
joblib
```

A simple environment setup is:

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install torch numpy pandas matplotlib seaborn scipy scikit-learn openpyxl joblib
```

CUDA is optional. If CUDA is available, the training pipeline automatically uses GPU; otherwise it runs on CPU.

You can inspect the active Python environment with:

```bash
python tools/envcheck.py
```

---

## Run a single DualModal_TFN training

### UORED: clean data

```bash
python main.py \
  --data_dir "/path/to/UORED" \
  --data_name UORED \
  --model_name DualModal_TFN \
  --SNR clean \
  --max_epoch 100
```

For a clean run, `--dual_gabor_mode auto` selects the 2D STFT-Gabor front-end. The internal low-SNR spectral branch is not enabled.

### UORED: noisy data, e.g. -6 dB

```bash
python main.py \
  --data_dir "/path/to/UORED" \
  --data_name UORED \
  --model_name DualModal_TFN \
  --SNR -6 \
  --max_epoch 100
```

`main.py` currently defaults to `--SNR -6`. Therefore, if you want a clean experiment, explicitly use:

```bash
--SNR clean
```

When train and validation SNR are both defined and equal, `DualModal_TFN` enters its low-SNR robust mode. With the default settings this enables:

- balanced per-sample/per-modality AWGN after normalization;
- `auto -> 1d` Gabor front-end;
- internal learnable two-modal spectral branch;
- train-split-only spectral-head warm start;
- EMA-based validation model comparison.

You can also set training and evaluation SNR separately:

```bash
python main.py \
  --data_dir "/path/to/UORED" \
  --data_name UORED \
  --model_name DualModal_TFN \
  --train_SNR -6 \
  --val_SNR -6
```

### BJTU-RAO

```bash
python main.py \
  --data_dir "/path/to/BJTU_RAO_Bogie_Datasets" \
  --data_name BJTU-RAO \
  --model_name DualModal_TFN \
  --SNR clean \
  --max_epoch 100
```

For a noisy BJTU-RAO experiment:

```bash
python main.py \
  --data_dir "/path/to/BJTU_RAO_Bogie_Datasets" \
  --data_name BJTU-RAO \
  --model_name DualModal_TFN \
  --SNR -6 \
  --max_epoch 100
```

---

## Run the single-modal baselines

Available models are:

```text
Backbone_CNN
Random_CNN
TFN_Gabor
DualModal_TFN
```

For single-modal models, use `--single_modality` to explicitly select vibration or acoustic data.

Vibration TFN:

```bash
python main.py \
  --data_dir "/path/to/UORED" \
  --data_name UORED \
  --model_name TFN_Gabor \
  --single_modality vib \
  --SNR clean
```

Acoustic TFN:

```bash
python main.py \
  --data_dir "/path/to/UORED" \
  --data_name UORED \
  --model_name TFN_Gabor \
  --single_modality aco \
  --SNR clean
```

---

## Important arguments

| Argument | Default | Description |
|---|---:|---|
| `--data_name` | `UORED` | `UORED` or `BJTU-RAO` |
| `--data_signalsize` | `2048` | signal/window length |
| `--model_name` | `DualModal_TFN` | model to train |
| `--SNR` | `-6` | common SNR in dB; use `clean`/`none` to disable |
| `--train_SNR` | `None` | training SNR; falls back to `--SNR` |
| `--val_SNR` | `None` | validation/test SNR; falls back to training SNR |
| `--batch_size` | `32` | batch size |
| `--seed` | `999` | training/noise random seed |
| `--split_seed` | `999` | data split seed |
| `--mid_channel` | `64` | number of Gabor preprocessing channels |
| `--gabor1d_kernel_size` | `63` | 1D Gabor kernel size |
| `--gabor2d_kernel_size` | `15` | 2D Gabor kernel size |
| `--dual_gabor_mode` | `auto` | `auto`, `1d`, or `2d` |
| `--use_decision_fusion` | `True` | add branch-level decision fusion |
| `--decision_alpha` | `0.1` | auxiliary branch-logit weight |
| `--aux_loss` | `True` | use unimodal auxiliary CE losses |
| `--aux_weight` | `0.2` | weight of each unimodal auxiliary loss |
| `--dual_internal_spectral` | `True` | enable internal spectral branch in noisy robust mode |
| `--internal_spectral_bins` | `128` | pooled FFT bins per modality |
| `--internal_spectral_hidden` | `128` | spectral MLP hidden width |
| `--internal_spectral_alpha` | `0.80` | initial spectral convex-fusion weight |
| `--internal_spectral_pretrain_epochs` | `15` | internal spectral warm-start epochs |
| `--dual_ema` | `True` | use EMA comparison in noisy DualModal runs |
| `--ema_decay` | `0.990` | maximum EMA decay |
| `--lr` | `1e-3` | backbone learning rate |
| `--gabor_lr` | `1e-5` | 1D Gabor parameter learning rate |
| `--dual_gabor2d_lr` | `5e-4` | 2D Gabor parameter learning rate |
| `--max_epoch` | `100` | training epochs |
| `--save_model` | `False` | save best `.pth` checkpoint to disk |

Run the complete argument list with:

```bash
python main.py --help
```

---

## Run the full accuracy pipeline

`PostProcess/TrainSequentially.py` currently runs the following UORED experiment grid:

- models: `Backbone_CNN`, `Random_CNN`, `TFN_Gabor`, `DualModal_TFN`;
- `mid_channel`: `16`, `32`, `64`, `128` where applicable;
- 50 epochs per run;
- 2 repeated runs;
- output root: `checkpoint/Acc-UORED`.

Run:

```bash
python PostProcess/TrainSequentially.py
```

**Important:** this script does not pass `--data_dir` on its generated command line. Before using it, either:

1. change the default `--data_dir` in `main.py`; or
2. modify `PostProcess/TrainSequentially.py` to append your dataset path to each command.

After all runs finish, summarize the logs and plot accuracy statistics:

```bash
python PostProcess/Acc_statistic.py
```

The aggregated results are saved under:

```text
checkpoint/Acc-UORED/postfiles/
```

---

## Output files

Each single run creates a directory similar to:

```text
checkpoint/
└── DualModal_TFN-UORED-time-trainSNR-6.0-<timestamp>/
```

Typical outputs include:

```text
training.log
Records.mat
MinorRecords.mat
acc output.txt
test_labels.npy
test_preds.npy
postprosess/
├── confusion_matrix_train.jpg
├── confusion_matrix_val.jpg
├── confusion_matrix_test.jpg
├── Minor_loss_acc.jpg
├── gabor_params_scatter.jpg
├── gabor_params.npz
├── gabor_params.json
├── gabor_kernels_example.jpg
└── gabor_val_sample_*_responses.jpg
```

The final test report includes accuracy and macro precision/recall/F1 when available.

If:

```bash
--save_model True
```

the best validation model is also written as a `.pth` file whose filename contains the selected epoch and validation accuracy.

---

## Frequency-domain analysis

`tools/frequency_analysis.py` can compute signal PSD/spectrograms and inspect frequency responses of CNN/Gabor filters.

Example:

```bash
python tools/frequency_analysis.py \
  --data_dir "/path/to/dataset" \
  --out_dir ./analysis_out
```

Optional trained checkpoints can be passed with:

```bash
--backbone_ckpt <checkpoint>
--dual_ckpt <checkpoint>
```

The default sampling rate used by this analysis script is `21000 Hz`; change it with `--fs` when necessary.

---

## Learnable vs. fixed Gabor comparison

A small ablation utility is included to compare learnable Gabor parameters with frozen Gabor parameters under the same initialization:

```bash
python tools/compare_fixed_vs_learnable.py \
  --data_dir "/path/to/dataset"
```

This tool is intended as a lightweight diagnostic/ablation script rather than the full training pipeline in `main.py`.

---

## Reproducibility

The main training script sets NumPy, Python, and PyTorch random seeds and enables deterministic cuDNN behavior.

For repeated independent runs, change:

```bash
--seed <new_seed>
```

Use `--split_seed` separately if you intentionally want a different dataset split.

For BJTU-RAO, the train/validation/test boundaries are chronological within each class; only the training subset is shuffled afterward.

---

