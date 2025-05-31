# Representation Learning Methods for Single-Cell Microscopy are Confounded by Background Cells

This repository contains code used in our paper to evaluate how background information in single-cell crops impacts deep learning crop-based models for single-cell microscopy. 

The repository is organized into three main directories:
- `swap_experiments/`: Code to train and evaluate SVM classifiers across the five background swap experiments described in the paper, assessing the impact of background on localization classification.

- `pifia_logistic_regression/`: Code for training and evaluating multinomial logistic regression models using PIFiA single-cell feature profiles to predict localization proportions.

- `generate_data_example/`: Example code for generating synthetic single-cell crops by swapping segmented center cells into donor background images.

We provide all model feature data (as zip files to download) here: https://drive.google.com/drive/u/0/folders/137GNjw4Cz9tLs8l2D4RuMLoX4axvJceX. The PIFiA dataset (containing the single-cell crops) can be found here: https://thecellvision.org/.

Specific details for each directory are below.

---

## `swap_experiments/`
**Data:**  
Download and unzip `features.zip` and place it in:

    swap_experiments/features/

This contains the feature representations for each model for data corresponding to all five experiment types. 

**Files:**
- `run_svm.py`:  
  Trains SVMs for all combinations of center-cell localizations (and background localization, if applicable) across all five experiment types.  
  Results are saved in `swap_experiments/results/`.

- `analyze_swap.ipynb`:  
  Loads classification results and reproduces:
  - **Table 1**: Mean classification accuracy per model and experiment type  
  - **Figure 2**: Sensitivity to background content by localization class

---

## `pifia_logistic_regression/`
**Data:**  
Download and unzip `pifia_feature_sets.zip` into:

    pifia_logistic_regression/pifia_feature_sets/

Model weights are and will be saved to:

    pifia_logistic_regression/trained_models/

**Files:**
- `train_lr.py`:  
  Trains two models:
  - One using original single-cell crops  
  - One using background-masked crops  
  Trained weights are saved in `trained_models/`.

- `analyze_lr.ipynb`:  
  Applies models to proteins with heterogeneous localization, computes KL divergence, and reproduces **Figure 3A** from the paper.

---

## `generate_data_example/`
**Files:**
- `generate_data_example.ipynb`:  
  Shows how to generate synthetic background-swapped crops by overlaying a segmented center cell crop onto a background crop.


