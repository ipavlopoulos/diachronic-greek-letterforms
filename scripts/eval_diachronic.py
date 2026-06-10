import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from sklearn.metrics import classification_report

from source import ResNetClassifier, image_path_to_tensor


_TYPO_MAP = {"Lamba": "Lambda", "Lmbda": "Lambda", "ChI": "Chi", "SIgma": "Sigma", "Sigmai": "Sigma"}


def norm_letter(name):
    return _TYPO_MAP.get(name, name)


def true_letter(filename):
    return norm_letter(filename.split("_")[0])


def evaluate(model, csv_path, cliplet_dir, classes, device, batch_size):
    df = pd.read_csv(csv_path)
    # Prefer the authoritative `letter` column when present (Med-Char); else parse filename.
    if "letter" in df.columns:
        gold = [norm_letter(str(l)) for l in df["letter"]]
    else:
        gold = [true_letter(fn) for fn in df["filename"]]
    paths = [cliplet_dir / fn for fn in df["filename"]]
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch = torch.cat(
                [image_path_to_tensor(p, device=device) for p in paths[start:start + batch_size]],
                dim=0,
            )
            idx = model(batch).argmax(dim=1)
            preds.extend(classes[int(i)] for i in idx.cpu())
    return gold, preds


def main():
    ap = argparse.ArgumentParser(description="Diachronic eval (PaLit-Char / Med-Char) for a ResNetClassifier checkpoint.")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "models/resnet_lf_dscl/best_resnet_lf_dscl_model.pth")
    ap.add_argument("--summary", type=Path, default=ROOT / "models/resnet_lf_dscl/resnet_lf_dscl_summary.json")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--datasets", nargs="+", default=["palitchar", "medchar"])
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    classes = json.load(open(args.summary))["classes"]
    model = ResNetClassifier(num_classes=len(classes), pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    for name in args.datasets:
        csv_path = args.data_dir / name / f"{name}.csv"
        cliplet_dir = args.data_dir / name / "cliplets"
        gold, preds = evaluate(model, csv_path, cliplet_dir, classes, device, args.batch_size)
        rep = classification_report(gold, preds, zero_division=0, output_dict=True)
        print(f"\n===== {name} =====")
        print(classification_report(gold, preds, zero_division=0))
        print(f"{name}: accuracy={rep['accuracy']:.4f} macroF1={rep['macro avg']['f1-score']:.4f} n={len(gold)}")


if __name__ == "__main__":
    main()
