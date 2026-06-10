"""Per-letter prototype comparison: cosine medoid vs. central mean.

For every letter whose optimal number of sub-clusters (by Silhouette) is >= 2,
spectral-cluster the letter's training embeddings and, for each sub-cluster,
render two representations side by side:

  - top row    : the cosine MEDOID (the paper's prototype; a real exemplar,
                 closest-to-centroid under cosine distance in embedding space).
  - bottom row : the pixel-wise MEAN of the ``--n-central`` images closest to the
                 cluster centroid (a denoised composite of the cluster core).

One PNG per letter is written to ``--output-dir``. Uses the same embedding
pipeline (training-time Normalize) and clustering rule as
``reproduce_letter_forms.py``.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import LabelEncoder

from source import ResNetClassifier
from scripts._hellchar import (
    load_hellchar,
    preprocess_gray,
    split_indices,
    extract_model_embeddings,
    set_seed,
)
from scripts.reproduce_letter_forms import optimal_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=ROOT / "models/resnet_lf_dscl/best_resnet_lf_dscl_model.pth")
    ap.add_argument("--summary", type=Path,
                    default=ROOT / "models/resnet_lf_dscl/resnet_lf_dscl_summary.json")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--output-dir", type=Path,
                    default=ROOT / "visual_artifacts/letter_forms_mean_vs_medoid")
    ap.add_argument("--n-central", type=int, default=10,
                    help="Number of most-centered images to average for the mean row.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_hellchar(args.data_dir)
    le = LabelEncoder()
    labels = le.fit_transform(data["letter"])
    classes = le.classes_.tolist()
    train_idx, val_idx, test_idx, y_train, y_val, y_test = split_indices(labels, args.seed, 0.2, 0.1)
    trainval_idx = np.concatenate([train_idx, val_idx])
    y_trainval = np.concatenate([y_train, y_val])

    gray = np.array([preprocess_gray(p) for p in data.path])
    gray_train = gray[trainval_idx][:, None, :, :]

    n_classes = len(json.load(open(args.summary))["classes"])
    model = ResNetClassifier(num_classes=n_classes, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    emb, emb_labels = extract_model_embeddings(
        model, torch.tensor(gray_train, dtype=torch.float32),
        torch.tensor(y_trainval), device, args.batch_size, args.num_workers)

    made = {}
    for c, letter in enumerate(classes):
        sub = np.where(emb_labels == c)[0]
        if len(sub) == 0:
            continue
        e = emb[sub]
        k = optimal_k(e, seed=args.seed)
        if k < 2:
            continue
        cl = SpectralClustering(n_clusters=k, assign_labels="discretize", random_state=args.seed,
                                affinity="nearest_neighbors",
                                n_neighbors=min(10, len(e) - 1)).fit_predict(e)

        fig, axes = plt.subplots(2, k, figsize=(k * 2.0, 4.4))
        axes = np.atleast_2d(axes)
        for j in range(k):
            idx = np.where(cl == j)[0]
            centroid = e[idx].mean(0, keepdims=True)
            order = cdist(e[idx], centroid, metric="cosine").ravel().argsort()
            top = idx[order[:args.n_central]]
            medoid_img = gray[trainval_idx[sub[idx[order[0]]]]]
            mean_img = gray[trainval_idx[sub[top]]].mean(0)
            axes[0, j].imshow(medoid_img, cmap="gray", vmin=0, vmax=1)
            axes[0, j].axis("off")
            axes[0, j].set_title(f"cluster {j+1} (n={len(idx)})", fontsize=9)
            axes[1, j].imshow(mean_img, cmap="gray", vmin=0, vmax=1)
            axes[1, j].axis("off")
        fig.suptitle(letter, fontsize=13, weight="bold")
        fig.text(0.015, 0.70, "medoid", rotation=90, va="center", fontsize=10, weight="bold")
        fig.text(0.015, 0.27, f"mean of {args.n_central}", rotation=90, va="center", fontsize=9, weight="bold")
        plt.tight_layout(rect=[0.05, 0, 1, 0.95])
        fig.savefig(args.output_dir / f"{letter}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        made[letter] = int(k)

    json.dump(made, open(args.output_dir / "summary.json", "w"), indent=1)
    print(f"Wrote {len(made)} comparison strips to {args.output_dir}")
    print("letters (k):", made)


if __name__ == "__main__":
    main()
