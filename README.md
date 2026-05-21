# Optical Greek Letters

Representation learning for ancient Greek letterforms across time.

This repository contains data, notebooks, reusable PyTorch code, and a saved lightweight CNN checkpoint for classifying Greek letter cliplets and studying the learned letterform embeddings.

If you are working from the parent workspace, the actual project is nested here:

```bash
cd greek-letter-vision
```

Most notebooks and paths assume this directory is the current working directory.

## What Is Included

- `source.py`: shared model, loss, augmentation, dataset, training, and evaluation code.
- `cnn_training.ipynb`: main training workflow for the lightweight CNN with similarity-weighted supervised contrastive learning.
- `cnn_embeddings_clustering.ipynb`: embedding extraction and clustering experiments.
- `Inference.ipynb`: minimal saved-model inference example.
- `best_cnn_letter_model.pth`: saved `CNN2D` weights for quick inference.
- `data/hellchar`: Hell-Char metadata and cliplets, used for training and in-distribution evaluation.
- `data/palitchar`: PaLit-Char metadata and cliplets, used as a near-time out-of-distribution evaluation set.
- `data/medchar`: Med-Char metadata and cliplets, used as a later-time out-of-distribution evaluation set.

## Setup

The code was developed in a Python/Jupyter environment with PyTorch and common scientific Python packages. There is no pinned environment file in this anonymized release, but the notebooks and `source.py` use:

```bash
pip install torch torchvision numpy opencv-python pillow scikit-learn matplotlib seaborn pandas jupyter
```

For GPU training, install the PyTorch build appropriate for your CUDA version from the official PyTorch instructions.

## Quick Inference

Run from the project directory:

```bash
jupyter notebook Inference.ipynb
```

The notebook loads `best_cnn_letter_model.pth`, samples an image from `data/palitchar/cliplets`, preprocesses it to `64x64` grayscale, and predicts one of the 24 Greek letter classes:

```python
from source import CNN2D, preprocess_image_2d
import torch

labels = [
    "Alpha", "Beta", "Chi", "Delta", "Epsilon", "Eta", "Gamma", "Iota",
    "Kappa", "Lambda", "Mu", "Nu", "Omega", "Omicron", "Phi", "Pi",
    "Psi", "Rho", "Sigma", "Tau", "Theta", "Upsilon", "Xi", "Zeta",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN2D(num_classes=24, image_size=(64, 64)).to(device)
model.load_state_dict(torch.load("best_cnn_letter_model.pth", map_location=device))
model.eval()
```

## Training Workflow

The main path is `cnn_training.ipynb`:

1. Load Hell-Char cliplets and metadata from `data/hellchar`.
2. Preprocess images to normalized `64x64` grayscale arrays.
3. Split known-letter samples into train, validation, and test partitions.
4. Train `CNN2D` using cross-entropy plus optional similarity-weighted supervised contrastive loss.
5. Update the class similarity matrix from learned class prototypes.
6. Evaluate with classification reports and confusion matrices.

The core training function is `train_cnn2d` in `source.py`.

## Method Summary

The model learns both class predictions and a 512-dimensional embedding for each letter image.

The classifier is trained with:

```text
loss = cross_entropy + lambda_scl * similarity_weighted_supcon
```

The contrastive term pulls examples of the same letter together while weighting negatives according to a class-similarity matrix. That matrix can be computed dynamically from class prototypes in the current embedding space, optionally blended with expert-defined priors.

The augmentation pipeline includes ordinary image perturbations and a lacunae-inspired transform, `RandomLacunae`, that masks irregular missing regions to mimic damaged manuscript surfaces.

## Data Notes

The datasets have different metadata schemas:

- Hell-Char: `filename`, `letter`, `TM`, `number`, `year`, `region`.
- PaLit-Char: `filename`, `tm`, `publication`, `year_post_quem`, `year_ante_quem`, `century`.
- Med-Char: `ID`, `uid`, `filename`, `year`, `inferred_label`, `letter`.

The cliplet filenames encode the letter label and manuscript/document identifier. The notebooks derive labels and metadata from those filenames and CSV files.

## Project State

This is a research release, not a packaged Python library. The notebooks are the primary workflows, and `source.py` collects the reusable components exported from those notebooks. Paths are relative to this nested project directory.
