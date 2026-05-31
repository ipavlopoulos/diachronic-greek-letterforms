# Diachronic Greek Letterforms

Representation learning for ancient Greek letterforms across time.

This repository accompanies the paper on robust representation learning for historical Greek handwriting. It contains Greek letter cliplet datasets, reusable PyTorch code, trained model weights, and notebooks for classification, representation extraction, and exploratory analysis of learned letterform embeddings.

The main model is a lightweight CNN trained to classify 24 Greek letter classes while also producing a 512-dimensional representation for each letter image. These representations can be used for nearest-neighbor search, clustering, prototype analysis, and other paleographic workflows.

## Quick Start

Clone the repository and enter the project directory:

```bash
git clone https://github.com/ipavlopoulos/diachronic-greek-letterforms.git
cd diachronic-greek-letterforms
```

Install the Python packages used by the notebooks and scripts:

```bash
pip install torch torchvision numpy opencv-python pillow scikit-learn matplotlib seaborn pandas jupyter ipywidgets
```

Then open the representation demo:

[![Open `representation_demo.ipynb` in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ipavlopoulos/diachronic-greek-letterforms/blob/main/notebooks/representation_demo.ipynb)

```bash
jupyter notebook notebooks/representation_demo.ipynb
```

GitHub's notebook preview is static and may occasionally show a rendering error for notebooks; use Colab or local Jupyter to run cells and recompute outputs. If the repository is private, Colab needs to be connected to a GitHub account with access, or the repository needs to be public.

Or export embeddings for a directory of cliplets:

```bash
python extract_representations.py data/palitchar/cliplets --output palitchar_representations.csv
```

For GPU training, install the PyTorch build appropriate for your CUDA version from the official PyTorch instructions. The released checkpoint and demo run on CPU.

## Repository Contents

| Path | Purpose |
| --- | --- |
| `source.py` | Shared model definitions, losses, augmentation, datasets, training, evaluation, and representation helpers. |
| `best_cnn_letter_model.pth` | Released `CNN2D` checkpoint for classification and representation extraction. |
| `extract_representations.py` | Command-line CSV exporter for 512-dimensional letterform representations. |
| `create_qualitative_examples.py` | Recreates the compact visual panel showing augmentation, confusability, cross-dataset Gamma examples, and visual variability. |
| `notebooks/representation_demo.ipynb` | Small guided notebook for loading the checkpoint, previewing a cliplet, extracting representations, inspecting nearest neighbors, and trying a user-uploaded image. |
| `notebooks/Inference.ipynb` | Minimal classification example with the saved model. |
| `notebooks/cnn_training.ipynb` | Main training workflow for the lightweight CNN with lacuna-driven augmentation and similarity-weighted supervised contrastive learning. |
| `notebooks/resnet_training.ipynb` | ResNet-18 training workflow that regenerates `best_resnet_supcon_model.pth` from the released Hell-Char data. |
| `notebooks/cnn_embeddings_clustering.ipynb` | Larger exploratory notebook for embedding extraction and clustering experiments. |
| `notebooks/data_overview.ipynb` | Quick data inventory notebook for released metadata and cliplet folders. |
| `data/hellchar` | Hell-Char metadata and cliplets for training and in-distribution evaluation. |
| `data/palitchar` | PaLit-Char metadata and cliplets for near-period out-of-distribution evaluation. |
| `data/medchar` | Med-Char metadata and cliplets for later-period diachronic evaluation. |
| `visual_artifacts/qualitative_visual_examples.png` | Generated qualitative panel used to illustrate visual challenges and examples. |

## Representation Extraction

The released checkpoint can be used as a transparent letterform encoder. Each input image produces one L2-normalized 512-dimensional vector from the penultimate CNN layer.

Shape guide:

- one image -> embedding array with shape `(1, 512)`
- 40 images -> embedding array with shape `(40, 512)`
- exported CSV for 40 images -> shape `(40, 516)` because it includes 4 metadata columns plus 512 embedding columns

Python example:

```python
from source import extract_letterform_representations, load_letterform_model

model = load_letterform_model("best_cnn_letter_model.pth", device="cpu")
embeddings, logits = extract_letterform_representations(
    model,
    ["data/palitchar/cliplets/Alpha_10352_001.jpg"],
    device="cpu",
    return_logits=True,
)

print(embeddings.shape)  # (1, 512)
```

CSV export example:

```bash
python extract_representations.py data/palitchar/cliplets --output palitchar_representations.csv
```

The CSV contains:

- `filename`
- `predicted_label`
- `confidence`
- `embedding_norm`
- `embedding_000` through `embedding_511`

For an interactive walkthrough, open `notebooks/representation_demo.ipynb`.

## Classification Inference

Run from the project directory:

```bash
jupyter notebook notebooks/Inference.ipynb
```

The notebook loads `best_cnn_letter_model.pth`, samples an image from `data/palitchar/cliplets`, preprocesses it to `64x64` grayscale, and predicts one of the 24 Greek letter classes.

A minimal code sketch:

```python
import torch
from source import LETTER_LABELS, extract_letterform_representations, load_letterform_model

model = load_letterform_model("best_cnn_letter_model.pth", device="cpu")
embeddings, logits = extract_letterform_representations(
    model,
    ["data/palitchar/cliplets/Alpha_10352_001.jpg"],
    device="cpu",
    return_logits=True,
)

probabilities = torch.softmax(torch.tensor(logits), dim=1)
predicted_index = int(probabilities.argmax(dim=1)[0])
print(LETTER_LABELS[predicted_index])
```

## Training Workflow

The main path for the released lightweight CNN is `notebooks/cnn_training.ipynb`:

1. Load Hell-Char cliplets and metadata from `data/hellchar`.
2. Preprocess images to normalized `64x64` grayscale arrays.
3. Split known-letter samples into train, validation, and test partitions.
4. Train `CNN2D` using cross-entropy plus optional similarity-weighted supervised contrastive loss.
5. Update the class similarity matrix from learned class prototypes.
6. Evaluate with classification reports and confusion matrices.

The core training function is `train_cnn2d` in `source.py`.

For ResNet-18 reproduction, use `notebooks/resnet_training.ipynb`. It reuses the same
Hell-Char split, lacuna-style augmentation, similarity-weighted supervised
contrastive objective, and evaluation function, then saves
`best_resnet_supcon_model.pth`. The ResNet checkpoint is not included in the
repository, so this notebook is the reproducible path for regenerating it.

## Method Summary

The model learns both class predictions and a 512-dimensional embedding for each letter image. It combines supervised classification with a representation-learning objective designed for visually confusable historical letterforms.

The classifier is trained with:

```text
loss = cross_entropy + lambda_scl * similarity_weighted_supcon
```

The contrastive term pulls examples of the same letter together while weighting negatives according to a class-similarity matrix. That matrix can be computed dynamically from class prototypes in the current embedding space, optionally blended with expert-defined priors.

The augmentation pipeline includes ordinary image perturbations and a lacunae-inspired transform, `RandomLacunae`, that masks irregular missing regions to mimic damaged manuscript surfaces.

## Visual Artifacts

The qualitative visual panel can be regenerated from the released cliplets:

```bash
python create_qualitative_examples.py
```

It writes `visual_artifacts/qualitative_visual_examples.png`, showing side-by-side augmentation examples, confusable forms, Gamma across datasets, and representative background/degradation variability.

## Data

The release contains three letter-level datasets:

- Hell-Char: training and in-distribution evaluation data derived from Hellenistic papyri.
- PaLit-Char: near-period evaluation data from later papyri/literary witnesses.
- Med-Char: later diachronic evaluation data from Byzantine minuscule manuscripts.

The datasets have different metadata schemas:

- Hell-Char: `filename`, `letter`, `TM`, `number`, `year`, `region`.
- PaLit-Char: `filename`, `tm`, `publication`, `year_post_quem`, `year_ante_quem`, `century`.
- Med-Char: `ID`, `uid`, `filename`, `year`, `inferred_label`, `letter`.

The cliplet filenames encode the letter label and manuscript/document identifier. The notebooks derive labels and metadata from those filenames and CSV files.

## Expected Input Images

The model expects a cropped image containing a single Greek letterform, similar to the images in the `cliplets` directories. Grayscale and RGB files are both accepted; images are converted to grayscale and resized to `64x64`.

For best results, use images with:

- one isolated character,
- minimal surrounding text,
- enough margin that strokes are not cut off,
- manuscript-like foreground/background contrast.

Full manuscript pages should be segmented into individual letter crops before using this model.

## Project State

This is a research release, not a packaged Python library. The notebooks are the primary workflows, and `source.py` collects reusable components exported from those notebooks.

Generated files such as `representation_demo_output.csv` are intentionally not required for running the project; they can be recreated from the demo notebook or `extract_representations.py`.

## Citation

If you use this code, data, or the released letterform representations, please cite the accompanying paper, to be presented at ICDAR in Vienna:

Pavlopoulos, J., Barbakos, S., Ferretti, L., Voulgarakis, D., Paparrigopoulou, A., Konstantinidou, M., De Gregorio, G., Marthot-Santaniello, I., Platanou, P., and Essler, H. Learning Diachronic Representations of Ancient Greek Letterforms. To appear/presented at ICDAR, Vienna.

```bibtex
@inproceedings{pavlopoulos_diachronic_greek_letterforms,
  title     = {Learning Diachronic Representations of Ancient Greek Letterforms},
  author    = {Pavlopoulos, John and Barbakos, Spyros and Ferretti, Lavinia and Voulgarakis, Dionysis and Paparrigopoulou, Asimina and Konstantinidou, Maria and De Gregorio, Giuseppe and Marthot-Santaniello, Isabelle and Platanou, Paraskevi and Essler, Holger},
  booktitle = {Proceedings of the International Conference on Document Analysis and Recognition (ICDAR)},
  address   = {Vienna, Austria},
  note      = {To appear},
  url       = {https://github.com/ipavlopoulos/diachronic-greek-letterforms}
}
```
