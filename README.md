# Secrets Everywhere: Auditing Memorization in Mobility Prediction Models

This repository contains the full experimental artifact for the paper  
**“Secrets Everywhere: Auditing Memorization in Mobility Prediction Models”**, submitted to ACM CCS.

The code evaluates mobility (next-location) prediction models for
**memorization of training data**, following the methodology described in the paper.
It supports preprocessing real-world mobility datasets, constructing reference
sets for different memorization abstractions, training multiple predictive models,
and computing trajectory-level memorization metrics and extractability signals.

---

## Project Structure

The repository is organized as a **three-stage pipeline**:

1. **Data processing and reference-set construction**
├── 1-data_processing/        # dataset normalization, preprocessing, reference sets
2. **Model training and memorization computation**
├── 2-predictive_models/      # model implementations and memorization computation
3. **Result analysis and figure generation**
├── 3-result_analysis.py      # aggregation and plotting of final results

---

## 1. Data Processing (`1-data_processing/`)

This directory implements all preprocessing steps and reference-set construction
procedures described in the paper.

### Raw datasets

```
1-data_processing/Raw_Datasets/
```

This directory is intentionally empty.  
All datasets used in the paper are **publicly available** and must be downloaded
by the user from their official sources (see the paper for citations and links).
After downloading, place the raw files into this directory.

### Preprocessing pipeline

The preprocessing stage consists of the following notebooks:

- `0-normalize_columns.ipynb`  
  Normalizes column names and formats across datasets to enforce a common schema
  (user ID, location, timestamp), regardless of the original dataset format.

- `1-preprocessing.ipynb`  
  Executes the full preprocessing pipeline described in the paper, including:
  - temporal resampling to fixed 30-minute intervals
  - anchor (home/work) inference
  - missing data completion
  - user and day filtering
  - segmentation into fixed-length trajectories

### Memorization reference sets

The following notebooks construct training sets and reference sets corresponding
to the three memorization abstractions studied in the paper:

- `memorization_analysis_type1.ipynb`  
  Location-level memorization

- `memorization_analysis_type2.ipynb`  
  Anchor-pair memorization

- `memorization_analysis_type3.ipynb`  
  Segment-level memorization

Each notebook produces per input dataset path:
- a fixed training set of trajectories
- a large reference set of plausible alternatives for each training trajectory

### Mobility characterization

- `mobility_characterization.ipynb`  
  Computes mobility statistics (e.g., entropy, regularity) and summarizes reference
  set properties used in the empirical analysis.

Shared utility functions used throughout preprocessing and analysis are located in:

```
1-data_processing/Helpers/
```

---

## 2. Predictive Models and Memorization Computation (`2-predictive_models/`)

This directory contains implementations or adapted wrappers for all mobility
prediction models evaluated in the paper.

### Included models

- **Markov (2nd order)**  
  `markov/`

- **LSTM-based and attention-based models** (via DeepMove codebase)  
  `DeepMove/`

- **Graph-Flashback**  
  `Graph-Flashback/`

- **LSTPM**  
  `LSTPM/`

Official implementations are used whenever available. Minor adaptations were made
to integrate:
- the unified preprocessing format,
- perplexity computation,
- trajectory-level memorization auditing.

Each model directory includes a `USAGE.md` file describing how to run that model
individually.

### Automated memorization evaluation

The script:

```
2-predictive_models/computation.py
```

acts as a **wrapper pipeline** that:
1. Loads the preprocessed datasets and reference sets with requirement to fill in the path to the files obtained in stage 1
2. Trains each model on the corresponding training trajectories
3. Computes memorization metrics for each memorization abstraction, dataset,
   and model
4. Stores trajectory-level memorization scores for downstream analysis

Running this script reproduces the core quantitative results reported in the paper.

---

## 3. Result Analysis and Visualization

Final aggregation and figure generation are performed by:

```
3-result_analysis.py
```

This script loads the memorization scores and model outputs produced in the
previous stage and generates the plots and statistics reported in the paper’s
Findings section.

---

## Reproducibility Notes

- All datasets used are public and must be obtained independently.
- The full pipeline is deterministic given fixed random seeds.
- Deep learning models require a GPU for practical runtimes.
- Intermediate outputs are cached to avoid recomputation.
- Exact numerical results may vary slightly due to nondeterminism in neural
  network training, but all reported trends are robust.

For quick validation, users may run the pipeline on a subset of users or a single
dataset.

---

## License

This project is released under the license specified in the `LICENSE` file.
