import argparse, json, os, sys
from pathlib import Path
os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from source import ResNetClassifier, build_S_from_prototypes
from scripts._hellchar import load_hellchar, preprocess_gray, split_indices, set_seed, to_model_input


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "models/resnet_lf_dscl/best_resnet_lf_dscl_model.pth")
    ap.add_argument("--summary", type=Path, default=ROOT / "models/resnet_lf_dscl/resnet_lf_dscl_summary.json")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "runs/figures_20260605")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_hellchar(args.data_dir)
    le = LabelEncoder(); labels = le.fit_transform(data["letter"]); classes = le.classes_.tolist()
    tr, va, te, ytr, yva, yte = split_indices(labels, args.seed, 0.2, 0.1)
    gray = np.array([preprocess_gray(p) for p in data.path])[:, None, :, :]

    model = ResNetClassifier(num_classes=len(json.load(open(args.summary))["classes"]), pretrained=False).to(dev)
    model.load_state_dict(torch.load(args.checkpoint, map_location=dev)); model.eval()

    # ----- confusion matrix on test split -----
    Xte = to_model_input(gray[te])
    preds = []
    with torch.no_grad():
        for s in range(0, len(Xte), 64):
            preds.append(model(Xte[s:s+64].to(dev)).argmax(1).cpu().numpy())
    preds = np.concatenate(preds)
    cm = confusion_matrix(yte, preds, labels=range(len(classes)))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Label"); plt.ylabel("True Label"); plt.tight_layout()
    plt.savefig(args.output_dir / "resnet18_confusion.png", dpi=200); plt.close()

    cmn = cm.astype(float) / cm.sum(1, keepdims=True).clip(min=1)
    pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cmn[i, j] > 0:
                pairs.append((cmn[i, j], classes[i], classes[j], int(cm[i, j])))
    pairs.sort(reverse=True)
    print("## Top confused pairs (true -> pred, rate, count):")
    for r, ti, pj, c in pairs[:12]:
        print(f"  {ti} -> {pj}: {r:.2f} (n={c})")

    # ----- dynamic similarity matrix (DSCL S) over training set -----
    Xtr = to_model_input(gray[tr]); Ytr = torch.tensor(ytr)
    loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=64, shuffle=False)
    S = build_S_from_prototypes(model, loader, dev, len(classes)).cpu().numpy()
    plt.figure(figsize=(10, 8))
    sns.heatmap(S, xticklabels=classes, yticklabels=classes, cmap="viridis")
    plt.tight_layout(); plt.savefig(args.output_dir / "resnet18_S_dynamic.png", dpi=200); plt.close()
    spairs = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            spairs.append((S[i, j], classes[i], classes[j]))
    spairs.sort(reverse=True)
    print("## Top similarity pairs (DSCL S):")
    for v, a, b in spairs[:10]:
        print(f"  {a}-{b}: {v:.2f}")


if __name__ == "__main__":
    main()
