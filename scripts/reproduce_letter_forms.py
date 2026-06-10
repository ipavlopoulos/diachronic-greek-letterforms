"""Regenerate per-letter form prototypes (Fig 2 + Appendix B).

Replicates the clustering-notebook logic: for each letter, pick the optimal
number of sub-clusters by Silhouette score, spectral-cluster the letter's
training embeddings, and pick each cluster's prototype as the COSINE medoid
(closest-to-centroid under cosine distance, matching paper section 3.4).
"""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

from source import ResNetClassifier
from scripts._hellchar import (
    load_hellchar,
    preprocess_gray,
    split_indices,
    extract_model_embeddings,
    set_seed,
)


def optimal_k(emb, max_k=10, seed=42):
    best_k, best_s = 1, -1.0
    if len(emb) < 5:
        return 1
    for k in range(2, min(max_k, len(emb) // 2 + 1)):
        try:
            cl = SpectralClustering(n_clusters=k, assign_labels="discretize",
                                    random_state=seed, affinity="nearest_neighbors",
                                    n_neighbors=min(10, len(emb) - 1)).fit_predict(emb)
            if len(np.unique(cl)) < k:
                continue
            s = silhouette_score(emb, cl)
        except Exception:
            continue
        if s > best_s:
            best_s, best_k = s, k
    return best_k


def cosine_medoid_indices(emb, k, seed=42):
    """Return indices (into emb) of each cluster's cosine medoid."""
    if k == 1:
        centroid = emb.mean(0, keepdims=True)
        d = cdist(emb, centroid, metric="cosine").ravel()
        return [int(d.argmin())]
    cl = SpectralClustering(n_clusters=k, assign_labels="discretize",
                            random_state=seed, affinity="nearest_neighbors",
                            n_neighbors=min(10, len(emb) - 1)).fit_predict(emb)
    medoids = []
    for c in range(k):
        idx = np.where(cl == c)[0]
        if len(idx) == 0:
            continue
        centroid = emb[idx].mean(0, keepdims=True)
        d = cdist(emb[idx], centroid, metric="cosine").ravel()
        medoids.append(int(idx[d.argmin()]))
    return medoids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=ROOT / "models/resnet_lf_dscl/best_resnet_lf_dscl_model.pth")
    ap.add_argument("--summary", type=Path,
                    default=ROOT / "models/resnet_lf_dscl/resnet_lf_dscl_summary.json")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "runs/figures_20260605/letter_forms")
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
    train_idx, val_idx, test_idx, y_train, y_val, y_test = split_indices(
        labels, args.seed, 0.2, 0.1)
    trainval_idx = np.concatenate([train_idx, val_idx])
    y_trainval = np.concatenate([y_train, y_val])

    gray = np.array([preprocess_gray(p) for p in data.path])
    gray_train = gray[trainval_idx][:, None, :, :]

    classes_ckpt = json.load(open(args.summary))["classes"]
    model = ResNetClassifier(num_classes=len(classes_ckpt), pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    emb, emb_labels = extract_model_embeddings(
        model, torch.tensor(gray_train, dtype=torch.float32),
        torch.tensor(y_trainval), device, args.batch_size, args.num_workers)

    summary = {}
    for c, letter in enumerate(classes):
        sub = np.where(emb_labels == c)[0]
        if len(sub) == 0:
            continue
        e = emb[sub]
        k = optimal_k(e, seed=args.seed)
        medoid_local = cosine_medoid_indices(e, k, seed=args.seed)
        medoid_global = trainval_idx[sub[medoid_local]]   # indices into full `gray`
        summary[letter] = {"optimal_k": k, "n": int(len(sub))}

        fig, axes = plt.subplots(1, len(medoid_global), figsize=(len(medoid_global) * 2.0, 2.0))
        if len(medoid_global) == 1:
            axes = [axes]
        for ax, gidx in zip(axes, medoid_global):
            ax.imshow(gray[gidx], cmap="gray")
            ax.axis("off")
        plt.tight_layout()
        fig.savefig(args.output_dir / f"{letter}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    json.dump(summary, open(args.output_dir / "letter_forms_summary.json", "w"), indent=1)
    print("Letters with >1 form:", {k: v["optimal_k"] for k, v in summary.items() if v["optimal_k"] > 1})
    print(f"Wrote {len(summary)} letter-form strips to {args.output_dir}")


if __name__ == "__main__":
    main()
