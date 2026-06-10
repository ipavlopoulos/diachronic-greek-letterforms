"""Evaluate a trained backbone on Hell-Char: classification (Table 1) and
clustering of the learned embeddings (Table 2).

This reproduces, for a single checkpoint, the numbers reported in the paper's
classification and clustering tables. The held-out split is the letter-stratified
20% test set (``random_state=42``), and embeddings are extracted with the same
``Normalize((0.5,), (0.5,))`` used at training time -- both are required to match
the paper.

Examples
--------
    # ResNet18 + LF + DSCL (the paper's main model)
    python scripts/evaluate.py --backbone resnet \
        --checkpoint runs/resnet_lf_dscl_lr1e-4_bs16/best_resnet_lf_dscl_model.pth

    # lightweight CNN
    python scripts/evaluate.py --backbone fcnn --checkpoint runs/fcnn_lf_dscl/best.pth
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import (
    adjusted_rand_score,
    classification_report,
    normalized_mutual_info_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mode

sys.path.insert(0, ".")
from source import CNN2D, ResNetClassifier  # noqa: E402

NORM = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])


def build_model(backbone, num_classes, device):
    if backbone == "fcnn":
        return CNN2D(num_classes=num_classes, image_size=(64, 64)).to(device)
    if backbone == "resnet":
        return ResNetClassifier(num_classes=num_classes, pretrained=False).to(device)
    # Transformer backbones (preliminary / future-work models in the paper).
    from scripts.train_timm_backbone import TimmClassifier
    arch = "vit_small_patch16_224" if backbone == "vit" else "convnextv2_tiny"
    return TimmClassifier(arch, num_classes, pretrained=False, in_chans=1, image_size=64).to(device)


def load_hellchar(folder):
    files = sorted(f for f in os.listdir(folder) if f.endswith((".jpg", ".jpeg", ".png")))
    files = [f for f in files if f.split("_")[0] != "Unknown"]
    le = LabelEncoder()
    y = le.fit_transform([f.split("_")[0] for f in files])
    X = np.array([
        cv2.resize(np.array(Image.open(os.path.join(folder, f)).convert("L")),
                   (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        for f in files
    ])
    return X, y, le


def embed(model, X, ids, device, as_embedding):
    out = []
    with torch.no_grad():
        for k in range(0, len(ids), 64):
            batch = torch.stack([
                NORM(Image.fromarray((X[i] * 255).astype(np.uint8))) for i in ids[k:k + 64]
            ]).to(device)
            if as_embedding:
                z = torch.nn.functional.normalize(model.get_embeddings(batch), dim=1)
            else:
                z = model(batch)
            out.append(z.cpu().numpy())
    return np.vstack(out)


def majority_map(cluster_ids, true):
    m = {c: mode(true[np.where(cluster_ids == c)[0]], keepdims=False).mode for c in np.unique(cluster_ids)}
    return np.array([m.get(c, -1) for c in cluster_ids])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="models/resnet_lf_dscl/best_resnet_lf_dscl_model.pth")
    ap.add_argument("--backbone", choices=["fcnn", "resnet", "vit", "convnext"], default="resnet")
    ap.add_argument("--data-dir", default="data/hellchar/cliplets")
    ap.add_argument("--seed", type=int, default=42, help="Stratified split seed (42 in the paper).")
    ap.add_argument("--per-letter", action="store_true", help="Also print the per-letter classification report.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, le = load_hellchar(args.data_dir)
    n_classes = len(le.classes_)
    idx = np.arange(len(X))
    tr, te, ytr, yte = train_test_split(idx, y, test_size=0.2, random_state=args.seed, stratify=y)

    model = build_model(args.backbone, n_classes, device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # --- Classification (Table 1) ---
    preds = embed(model, X, te, device, as_embedding=False).argmax(1)
    report = classification_report(yte, preds, target_names=le.classes_, zero_division=0, output_dict=True)
    print(f"\n[Classification]  Accuracy={report['accuracy']:.3f}  macro-F1={report['macro avg']['f1-score']:.3f}")
    if args.per_letter:
        for c in le.classes_:
            r = report[c]
            print(f"  {c:<9} P={r['precision']:.2f} R={r['recall']:.2f} F1={r['f1-score']:.2f} (n={int(r['support'])})")

    # --- Clustering of embeddings (Table 2) ---
    Z_tr = embed(model, X, tr, device, as_embedding=True)
    Z_te = embed(model, X, te, device, as_embedding=True)
    km = KMeans(n_clusters=n_classes, random_state=42, n_init=20).fit(Z_tr)
    km_pred = majority_map(km.predict(Z_te), yte)
    sp = majority_map(SpectralClustering(n_clusters=n_classes, assign_labels="discretize",
                                         random_state=42, affinity="nearest_neighbors",
                                         n_neighbors=10).fit_predict(Z_te), yte)
    ah = AgglomerativeClustering(n_clusters=n_classes, linkage="ward", metric="euclidean").fit_predict(Z_tr)
    ah_pred = majority_map(KNeighborsClassifier(5).fit(Z_tr, ah).predict(Z_te), yte)
    print("\n[Clustering of embeddings]  NMI / ARI")
    for name, pred in [("k-means ", km_pred), ("Spectral", sp), ("Agglom. ", ah_pred)]:
        print(f"  {name}  NMI={normalized_mutual_info_score(yte, pred):.3f}  "
              f"ARI={adjusted_rand_score(yte, pred):.3f}")


if __name__ == "__main__":
    main()
