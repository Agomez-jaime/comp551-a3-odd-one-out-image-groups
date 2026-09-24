# Odd-One-Out Image Groups — Kaggle Competition

COMP 551 (Applied Machine Learning), McGill University — Mini-Project 3, Group 58.

**Team:** Hadi El Nawfal, Andrea Gómez Jaime, Bora Sanal

A 5-class classification problem run as an in-class Kaggle competition: each
example is a group of 5 grayscale 32×32 images, where 4 share a hidden common
property and 1 is the "odd one out." The task is to predict the index (0–4) of
the outlier image — without ever being told what the hidden shared property is.

## Approach

- **Baseline**: logistic regression on the flattened 5-image group, included
  in the starter code as a sanity check (performs close to the 20% random-chance
  floor for this 5-way problem).
- **Our model ("Odd-One-Out Classifier V11")**: a convolutional neural network
  built in PyTorch, trained with:
  - **Stratified K-fold cross-validation** for robust hyperparameter selection.
  - **Stochastic Weight Averaging (SWA)** with a dedicated SWA learning-rate
    schedule and batch-norm recalibration (`update_bn`), to improve
    generalization over a single-checkpoint model.
- Final predictions were submitted as a CSV to the class Kaggle leaderboard
  for the full 2,000-group test set.

## Files

- `code.ipynb` — full notebook: data loading, baseline, model architecture,
  training loop, evaluation, and Kaggle submission generation.
- `Main.pdf` — competition report, including architecture details,
  hyperparameter choices, and parameter count.

## Running it

The notebook expects `x_train.npy` / `y_train.npy` (and the corresponding test
files) mounted from Google Drive, as provided by the assignment's Kaggle
competition data. Update the Drive paths at the top of the notebook and run
all cells to reproduce training and the submission CSV.

## Authors

Hadi El Nawfal, Andrea Gómez Jaime, Bora Sanal
