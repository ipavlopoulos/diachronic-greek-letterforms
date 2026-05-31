import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import classification_report

from source import ResNetClassifier, image_path_to_tensor


def predict_medchar(model, medchar, cliplet_dir, classes, device, batch_size):
    image_paths = [cliplet_dir / filename for filename in medchar["filename"]]
    predictions = []
    confidences = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]
            batch = torch.cat([image_path_to_tensor(path, device=device) for path in batch_paths], dim=0)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = probs.max(dim=1)
            predictions.extend(classes[int(i)] for i in pred_idx.cpu())
            confidences.extend(float(v) for v in conf.cpu())
    medchar = medchar.copy()
    medchar["prediction"] = predictions
    medchar["confidence"] = confidences
    medchar["is_error"] = medchar["prediction"] != medchar["letter"]
    return medchar


def plot_error_year_boxplot(results, output_path):
    errors = results[results["is_error"]].copy()
    order = sorted(results["letter"].unique())
    counts = errors.groupby("letter").size().reindex(order, fill_value=0)

    plt.figure(figsize=(14.8, 7.9))
    ax = sns.boxplot(
        data=errors,
        x="letter",
        y="year",
        order=order,
        color="#d8e3f3",
        width=0.62,
        fliersize=2.5,
        linewidth=1.0,
    )
    sns.stripplot(
        data=errors,
        x="letter",
        y="year",
        order=order,
        color="#4c4c4c",
        size=2.2,
        jitter=0.18,
        alpha=0.45,
        ax=ax,
    )

    y_min = int(results["year"].min()) - 35
    y_max = int(results["year"].max()) + 45
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("")
    ax.set_ylabel("Year CE")
    ax.set_title("Years per true letter for misclassified Med-Char images")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)

    for idx, letter in enumerate(order):
        ax.text(
            idx,
            y_max - 18,
            str(int(counts.loc[letter])),
            ha="center",
            va="top",
            color="#b22222",
            fontsize=10,
            fontweight="bold",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Reproduce Figure 4: Med-Char error years by letter.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "runs" / "resnet_lf_dscl_lr1e-4_bs16" / "best_resnet_lf_dscl_model.pth",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "runs" / "resnet_lf_dscl_lr1e-4_bs16" / "resnet_lf_dscl_summary.json",
    )
    parser.add_argument("--medchar-csv", type=Path, default=ROOT / "data" / "medchar" / "medchar.csv")
    parser.add_argument("--cliplet-dir", type=Path, default=ROOT / "data" / "medchar" / "cliplets")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "paper" / "imgs" / "ood_error_year_boxplot_resnet_reproduced.png",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=ROOT / "runs" / "resnet_lf_dscl_lr1e-4_bs16" / "medchar_predictions_for_figure4.csv",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--use-existing-inferred-labels",
        action="store_true",
        help="Use medchar.csv inferred_label values instead of re-predicting from a checkpoint.",
    )
    args = parser.parse_args()

    medchar = pd.read_csv(args.medchar_csv)
    if args.use_existing_inferred_labels:
        results = medchar.copy()
        results["prediction"] = results["inferred_label"]
        results["confidence"] = None
        results["is_error"] = results["prediction"] != results["letter"]
    else:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        with open(args.summary) as handle:
            classes = json.load(handle)["classes"]
        model = ResNetClassifier(num_classes=len(classes), pretrained=False).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        results = predict_medchar(model, medchar, args.cliplet_dir, classes, device, args.batch_size)

    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.results_csv, index=False)
    plot_error_year_boxplot(results, args.output)

    report = classification_report(results["letter"], results["prediction"], zero_division=0, output_dict=True)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.results_csv}")
    print(f"Med-Char accuracy: {report['accuracy']:.4f}")
    print(f"Med-Char macro F1: {report['macro avg']['f1-score']:.4f}")
    print(f"Errors: {int(results['is_error'].sum())}/{len(results)}")


if __name__ == "__main__":
    main()
