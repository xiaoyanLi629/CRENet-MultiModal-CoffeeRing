# Coffee-Ring Effect Analysis for Water Quality Assessment

> **A multi-modal machine learning framework for automated environmental condition classification and element separation quantification via coffee-ring pattern analysis**

---

## Background

When a droplet containing suspended particles evaporates on a surface, the solutes tend to migrate toward the droplet edge, forming a distinctive ring-shaped deposit known as the "coffee-ring effect." This phenomenon is governed by capillary flow dynamics and is sensitive to environmental conditions such as temperature and humidity.

This project leverages the coffee-ring effect as an analytical tool for water quality assessment. By depositing water samples under controlled environmental conditions and analyzing the resulting ring patterns, we can characterize the elemental composition of the water. The key insight is that different environmental conditions produce varying degrees of element separation within the ring pattern — and machine learning can automatically classify these conditions and quantify the separation quality.

The framework processes three complementary data modalities collected from each coffee-ring sample:

- **Optical microscopy images** — capture macroscopic ring morphology (color, shape, ring width)
- **Scanning Electron Microscopy (SEM) images** — reveal microstructure details (particle distribution, crystal formation)
- **Energy-Dispersive X-ray Spectroscopy (EDS) maps** — provide spatial distribution of chemical elements (Na, Cl, Ca, Mg, K, S, Si)

A novel metric called the **Element Separation Index (ESI)** is introduced to quantitatively measure how well elements are spatially separated within the coffee-ring pattern. Higher ESI values indicate better element separation, which improves the analytical utility of the technique for water quality characterization.

---

## Key Results

| Metric | Value |
|:---|:---:|
| CRENetEfficient (multi-modal, late concat) | **88.9% +/- 5.8%** |
| EfficientNet-B0 (optical-only, pretrained) | **92.0% +/- 2.3%** |
| Early fusion (channel-inflated, pretrained) | **91.6% +/- 3.6%** |
| CRENet scratch (multi-modal, self-attention fusion; late-concat scratch: 80.4% +/- 8.2%) | **76.9% +/- 1.1%** |
| Leave-one-recipe-out: CRENetEfficient / EfficientNet-B0 | 66.2% / 72.0% |
| Optimal Condition (12 statistical analyses, 8/12 criteria) | **F** (23--26 C, 45--50% RH) |

---

## Project Structure

```
CRENet.../                       ← Unified project (merged with former Nanochromatography/Project 2 on 2026-07-13)
│
├── README.md                    ← This file
│
├── paper/                       ← PRICAI 2026 paper (Springer LNCS)
│   ├── main.tex                 ← Main paper (12 pages)
│   ├── supplementary.tex        ← Supplementary materials (7 pages)
│   ├── main.pdf                 ← Compiled main paper
│   ├── supplementary.pdf        ← Compiled supplementary
│   ├── references.bib           ← BibTeX bibliography
│   ├── llncs.cls / splncs04.bst ← Springer LNCS style files
│   ├── figure/                  ← Figures for the paper
│   └── appendix/                ← Appendix figures (SEM/EDS visualisations)
│
├── experiments/                 ← Deep learning code (models, training, evaluation)
│   ├── ml_framework.py          ← Original ML framework (data loading, ESI)
│   ├── models.py                ← Model definitions (CRENet, CRENetEfficient, baselines)
│   ├── train_utils.py           ← Training utilities, 5-fold CV, clean ESI
│   ├── run_experiments.py       ← Unified runner: python -m experiments.run_experiments
│   ├── esi_calibration.py       ← ESI weight calibration
│   ├── augmented_dataset.py     ← Data augmentation wrapper
│   ├── run_all.sh               ← Run all phases sequentially
│   └── run_pipeline.sh          ← Intelligent pipeline with auto-decisions
│
├── data/                        ← All experimental data (225 samples)
│   ├── optical/                 ← PREPROCESSED optical images, enhanced, used for training (A–I)
│   ├── sem/                     ← PREPROCESSED SEM images, enhanced (A–I)
│   ├── eds/                     ← EDS element maps used for training (A–I)
│   └── raw_data/                ← ORIGINAL unprocessed data (from Project 2)
│       ├── optical/             ← Camera originals of data/optical (1280x1024)
│       ├── sem/                 ← Originals of data/sem (2560x1920)
│       ├── photo_pattern/       ← Phone photos of residue patterns (not used in training)
│       └── eds/                 ← Original EDS exports (4 files differ from data/eds)
│
├── statistical_analysis/        ← MATLAB/R/Python statistical pipeline (2020–2022).
│   │                              NOT legacy: generates the PERMANOVA/MANOVA/ANOSIM/
│   │                              ANOVA/CMDS results reported in the paper
│   ├── extract_*.m / aggregate_*.m / merge_*.m / export_*.m
│   │                              ← Particle feature extraction pipeline (regionprops)
│   ├── feature_analysis/        ← PERMANOVA/MANOVA/ANOSIM/EDS-ANOVA scripts + summaries
│   ├── pixel_analysis/          ← KL/JS divergence analysis
│   ├── eds_analysis/            ← EDS processing (svd_denoise, image overlapping)
│   └── image_feature_processing/← Mann matrix processing
│
├── dataset_paper/               ← Scientific Data descriptor paper (self-contained git repo)
│   ├── data/                    ← Published dataset copy: optical/sem/eds (preprocessed)
│   │                              + corrected_images (450) + features + metadata CSVs
│   ├── experiment/              ← Validation analysis code for the data paper
│   ├── manuscript/              ← Data descriptor manuscript (LaTeX)
│   └── README.md, LICENSE       ← Public repo files
│
├── reference_materials/         ← Background documents (from Project 2)
│   ├── Detroit_water_quality_reports/  ← 2017–2019 annual reports (research context)
│   └── sample_data_sheet.xlsx   ← Sample metadata
│
└── results/                     ← All experiment results (clean, no data leakage)
    ├── phase0/                  ← CRENet + RF + SVM + clean ESI (baseline)
    ├── phase2/                  ← SOTA baselines
    ├── phase3/                  ← Fusion strategy ablation
    ├── phase6/                  ← CRENetEfficient (final model, 88.89%)
    ├── esi_calibration/         ← ESI calibration results
    └── logs/                    ← Training logs
```

### What each directory contains

**`ml_framework.py`** is the single Python file that implements the entire ML pipeline. It includes the multi-modal CNN model definition (`MultiModalCNN`), the Element Separation Index calculation (`ElementSeparationIndex`), data loading and preprocessing, model training and evaluation, and result visualization. All code is in one file for portability.

**`paper/`** contains the conference paper prepared for submission to ICANN 2026 (International Conference on Artificial Neural Networks), formatted according to the Springer LNCS template. To compile: run `pdflatex main && bibtex main && pdflatex main && pdflatex main` from within this directory.

**`data/`** holds the preprocessed (brightness/contrast enhanced) experimental data actually fed to the models, organized by modality (optical, SEM, EDS) and within each modality by environmental condition (subdirectories A through I). Each condition represents a specific combination of temperature and humidity. There are 225 samples (9 conditions x 25 samples/condition). The unprocessed originals live in `data/raw_data/` — see `data/raw_data/README.txt` for the exact differences.

**`statistical_analysis/`** contains the classical analysis pipeline that predates the CNN work but is *actively cited by the paper*: all PERMANOVA/MANOVA/ANOSIM/ANOVA tables, the regionprops particle-feature extraction described in the supplementary methods, and the CMDS/Jensen-Shannon analyses. Keep it for reproducibility during review.

**`results/`** stores all outputs from completed experiment phases, including saved model weights (`.pth` files), evaluation metrics, and training logs.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Input Data (2050 samples)              │
├──────────────┬──────────────────┬────────────────────────┤
│  Optical RGB │  SEM Grayscale   │    EDS Element Maps    │
│  512x512x3   │  512x512x1       │    7 x 256x256         │
└──────┬───────┴────────┬─────────┴──────────┬─────────────┘
       │                │                    │
       ▼                ▼                    ▼
┌──────────────┐ ┌──────────────┐  ┌────────────────┐
│ Optical CNN  │ │   SEM CNN    │  │  EDS Encoder   │
│ 5x Conv+BN   │ │ 5x Conv+BN   │  │  3x FC+Dropout │
│ +ReLU+Pool   │ │ +ReLU+Pool   │  │  → 128-dim     │
│ → 256-dim    │ │ → 256-dim    │  │                │
└──────┬───────┘ └──────┬───────┘  └──────┬─────────┘
       │                │                  │
       ▼                ▼                  │
  ┌─────────────────────────┐              │
  │  Multi-Head Attention   │              │
  │  (4 heads, 256-dim)     │              │
  │  → 256-dim              │              │
  └────────────┬────────────┘              │
               │                           │
               └─────────┬────────────────┘
                         ▼
                ┌────────────────┐
                │ Concatenation  │
                │ 256 + 128 = 384│
                └───────┬────────┘
                        ▼
                ┌────────────────┐
                │ Classification │
                │ FC 384→256→128 │
                │      → 9      │
                └───────┬────────┘
                        ▼
                ┌────────────────┐
                │  9 Conditions  │
                │    (A – I)     │
                └────────────────┘
```

---

## Experimental Conditions

Nine environmental conditions (A–I) were tested, crossing three temperature ranges (20–23, 23–26, 26–29 C) with three relative-humidity ranges (35–40, 40–45, 45–50%).

Per-condition mean ESI (data-driven weight alpha*, 70th-percentile detection, 100-point sub-sampling; see `esi_honest_analysis.py` and `results/esi_honest/`):

| Condition | Temperature | Humidity | ESI (mean +/- std) |
|:---------:|:-----------:|:--------:|:------------------:|
| F | 23–26 C | 45–50% | 0.520 +/- 0.041 |
| G | 26–29 C | 35–40% | 0.519 +/- 0.029 |
| H | 26–29 C | 40–45% | 0.517 +/- 0.025 |
| C | 20–23 C | 45–50% | 0.516 +/- 0.021 |
| I | 26–29 C | 45–50% | 0.516 +/- 0.027 |
| E | 23–26 C | 40–45% | 0.512 +/- 0.034 |
| B | 20–23 C | 40–45% | 0.511 +/- 0.038 |
| D | 23–26 C | 35–40% | 0.509 +/- 0.039 |
| A | 20–23 C | 35–40% | 0.509 +/- 0.033 |

Condition F ranks highest descriptively, and this rank-1 identity is stable across alpha in [0, 0.9]. However, the between-condition ESI differences are **not statistically significant** (one-way ANOVA p = 0.92) and the top-ranked condition is sensitive to the detection threshold and sub-sampling budget. The identification of **Condition F** as the optimal condition therefore rests on the 12 morphology-based statistical analyses (PERMANOVA, MANOVA, ANOSIM, ANOVA; F ranks optimal in 8 of 12), with ESI serving as complementary descriptive evidence.

---

## Running the Code

### Prerequisites

Install the required Python packages:

```bash
pip install torch torchvision numpy scipy scikit-learn opencv-python matplotlib pandas
```

Tested with Python 3.8+ and PyTorch 1.10+. A GPU is recommended but not required — the framework will automatically use CUDA if available.

### Basic usage

Run the full analysis pipeline (training + evaluation + ESI calculation):

```bash
python ml_framework.py --data_path ./experiment/data
```

### Save models and results

To save trained models, figures, and reports:

```bash
python ml_framework.py --data_path ./experiment/data --save_models --output_dir ./production_results
```

### All command-line arguments

| Argument | Type | Default | Description |
|:---|:---:|:---:|:---|
| `--data_path` | str | *required* | Path to data directory containing `optical/`, `sem/`, `eds/` subdirectories |
| `--epochs` | int | 100 | Number of training epochs for the CNN model |
| `--batch_size` | int | 8 | Batch size for CNN training |
| `--learning_rate` | float | 0.0001 | Learning rate for Adam optimizer |
| `--save_models` | flag | off | If set, saves trained models, figures, and reports to `output_dir` |
| `--output_dir` | str | `./results` | Directory to save all outputs |

### Expected data directory structure

The `--data_path` directory must contain three subdirectories, each with condition folders A through I:

```
data_path/
├── optical/
│   ├── A/    ← optical images for condition A
│   ├── B/
│   └── ...I/
├── sem/
│   ├── A/    ← SEM images for condition A
│   ├── B/
│   └── ...I/
└── eds/
    ├── A/    ← EDS element maps for condition A
    ├── B/
    └── ...I/
```

### Compiling the paper

To compile the ICANN 2026 paper from LaTeX source:

```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

The triple `pdflatex` run ensures all cross-references and citations are resolved.

---

## Statistical Methods

The framework uses multiple statistical methods to validate results beyond ML classification accuracy:

| Method | Purpose |
|:---|:---|
| **PERMANOVA** | Tests whether multivariate elemental compositions differ significantly across conditions |
| **MANOVA** | Parametric test for simultaneous differences across multiple response variables |
| **ANOVA** | Per-element analysis of variance across conditions |
| **ANOSIM** | Non-parametric test for multivariate group similarity |
| **Jensen-Shannon Divergence** | Measures distributional similarity between element profiles of different conditions |
| **Classical MDS** | Reduces high-dimensional element data to 2D for visualization of condition clustering |

---

## Conference

**ICANN 2026** — International Conference on Artificial Neural Networks
- Format: Springer Lecture Notes in Computer Science (LNCS)
- Template: `llncs` class with `runningheads` option
- Bibliography style: `splncs04`

---

## Model Performance Comparison (5-Fold Cross-Validation)

```
Accuracy (%)
 100 ┬
  95 ┤  ######
  90 ┤  ###### 92.0%
  85 ┤  ######  ######
  80 ┤  ######  ###### 88.9%
  75 ┤  ######  ######  ######
  70 ┤  ######  ######  ###### 76.9%
     └──┴──────┴─┴──────┴─┴──────┴──
      EffNet-B0  CRENet   CRENet
      (optical)  Efficient (scratch)
```

---

## Element Separation Index (ESI)

The ESI is a novel metric introduced in this work to quantitatively evaluate the quality of element separation in coffee-ring patterns. It combines two components:

**ESI = alpha * S_intra + beta * S_inter**, with beta = 1 - alpha

- **S_intra** (intra-element clustering): mean of 1 / (1 + mean pairwise distance) per element — how tightly the deposits of each element cluster.
- **S_inter** (inter-element separation): normalised mean minimum distance between the deposits of different elements.
- **alpha**: calibrated from data rather than fixed. Raw-variance calibration gives alpha* ~= 0.001 (S_inter-dominated); a scale-invariant (relative-variance) calibration gives alpha ~= 0.82. Both composites yield the same descriptive ranking, and neither yields statistically significant between-condition differences — see `esi_honest_analysis.py` for the significance tests and threshold/sub-sampling sensitivity analyses.

ESI is a **descriptive** index of elemental spatial organisation within this dataset (distances are in pixel units of the 256x256 EDS grid); it is not by itself a validated criterion for condition optimality.
