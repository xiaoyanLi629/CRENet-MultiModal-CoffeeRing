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
| CRENetEfficient (multi-modal) | **88.9% +/- 5.8%** |
| EfficientNet-B0 (optical-only) | **92.0%** |
| CRENet scratch (multi-modal) | **76.9% +/- 1.1%** |
| Optimal Condition (ESI) | **F** (23--26 C, 45--50% RH) |

---

## Project Structure

```
Thesis_3/
│
├── README.md                    ← This file
│
├── paper/                       ← ICANN 2026 conference paper (Springer LNCS)
│   ├── main.tex                 ← Main paper (12 pages)
│   ├── supplementary.tex        ← Supplementary materials (7 pages)
│   ├── main.pdf                 ← Compiled main paper
│   ├── supplementary.pdf        ← Compiled supplementary
│   ├── references.bib           ← BibTeX bibliography
│   ├── llncs.cls / splncs04.bst ← Springer LNCS style files
│   ├── figure/                  ← Figures for the paper
│   └── appendix/                ← Appendix figures (SEM/EDS visualisations)
│
├── experiments/                 ← All code (models, training, evaluation)
│   ├── ml_framework.py          ← Original ML framework (data loading, ESI)
│   ├── models.py                ← Model definitions (CRENet, CRENetEfficient, baselines)
│   ├── train_utils.py           ← Training utilities, 5-fold CV, clean ESI
│   ├── run_experiments.py       ← Unified runner: python -m experiments.run_experiments
│   ├── esi_calibration.py       ← ESI weight calibration
│   ├── augmented_dataset.py     ← Data augmentation wrapper
│   ├── run_all.sh               ← Run all phases sequentially
│   └── run_pipeline.sh          ← Intelligent pipeline with auto-decisions
│
├── data/                        ← Raw experimental data (225 samples)
│   ├── optical/                 ← Optical microscopy images (conditions A–I)
│   ├── sem/                     ← SEM images (conditions A–I)
│   └── eds/                     ← EDS element maps (conditions A–I)
│
└── results/                     ← All experiment results (clean, no data leakage)
    ├── phase0/                  ← CRENet + RF + SVM + clean ESI (baseline)
    ├── phase1/                  ← Modality ablation (7 configurations)
    ├── phase2/                  ← SOTA baselines
    ├── phase3/                  ← Fusion strategy ablation
    ├── phase6/                  ← CRENetEfficient (final model, 88.89%)
    ├── esi_calibration/         ← ESI calibration results
    └── logs/                    ← Training logs
```

### What each directory contains

**`ml_framework.py`** is the single Python file that implements the entire ML pipeline. It includes the multi-modal CNN model definition (`MultiModalCNN`), the Element Separation Index calculation (`ElementSeparationIndex`), data loading and preprocessing, model training and evaluation, and result visualization. All code is in one file for portability.

**`paper/`** contains the conference paper prepared for submission to ICANN 2026 (International Conference on Artificial Neural Networks), formatted according to the Springer LNCS template. To compile: run `pdflatex main && bibtex main && pdflatex main && pdflatex main` from within this directory.

**`data/`** holds the raw experimental data organized by modality (optical, SEM, EDS) and within each modality by environmental condition (subdirectories A through I). Each condition represents a specific combination of temperature and humidity. There are 225 samples (9 conditions x 25 samples/condition).

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

Nine environmental conditions (A–I) were tested, varying temperature and humidity to determine the optimal setting for element separation:

| Condition | Temperature | Humidity | ESI Score | Rank |
|:---------:|:-----------:|:--------:|:---------:|:----:|
| **F** | **23–26 C** | **45–50%** | **0.2323 +/- 0.0296** | **1** |
| E | 23–26 C | 45–50% | 0.2156 +/- 0.0278 | 2 |
| D | 23–26 C | 30–35% | 0.2089 +/- 0.0301 | 3 |
| B | 18–22 C | 45–50% | 0.1923 +/- 0.0287 | 4 |
| A | 18–22 C | 30–35% | 0.1847 +/- 0.0312 | 5 |
| H | 27–30 C | 45–50% | 0.1789 +/- 0.0321 | 6 |
| C | 18–22 C | 60–65% | 0.1756 +/- 0.0334 | 7 |
| G | 27–30 C | 30–35% | 0.1634 +/- 0.0345 | 8 |
| I | 27–30 C | 60–65% | 0.1567 +/- 0.0358 | 9 |

**Condition F** (23–26 C, 45–50% RH) achieved the highest ESI score, indicating the best element separation for analytical purposes. Moderate temperature and humidity create optimal evaporation dynamics that maximize the spatial separation of dissolved elements within the coffee-ring pattern.

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

**ESI = alpha * S_inter + beta * S_intra**

- **S_inter** (inter-element separation): Measures the spatial distance between centroids of different element distributions. Higher values mean elements are deposited in more distinct spatial zones.
- **S_intra** (intra-element clustering): Measures how tightly each element's deposits are clustered. Higher values mean each element forms a more concentrated deposit.
- **alpha = 0.6, beta = 0.4**: Weighting coefficients emphasizing inter-element separation over intra-element clustering.

ESI ranges from 0 (no separation) to 1 (perfect separation). Values above 0.2 indicate analytically useful separation for water quality assessment.
