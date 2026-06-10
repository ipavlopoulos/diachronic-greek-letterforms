import argparse
import json
import os
import random
from pathlib import Path
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from source import (
    CNN2D,
    ImageDatasetAugmented,
    RandomLacunae,
    RandomRectangles,
    custom_similarity_matrix,
    train_cnn2d,
    tta_transform,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess_image_2d(image_path, size=(64, 64), otsu=False):
    img = Image.open(image_path).convert("L")
    img_np = np.array(img)
    if otsu:
        _, img_np = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_np = 255 - img_np
    img_resized = cv2.resize(img_np, size, interpolation=cv2.INTER_AREA)
    return img_resized.astype(np.float32) / 255.0


def load_hellchar(data_dir):
    cliplet_dir = data_dir / "hellchar" / "cliplets"
    image_paths = sorted(
        path for path in cliplet_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    data = pd.DataFrame({"path": image_paths, "filename": [path.name for path in image_paths]})
    data["letter"] = data.filename.apply(lambda value: value.split("_")[0])
    return data[data["letter"] != "Unknown"].copy()


def build_loaders(args):
    data = load_hellchar(args.data_dir)
    image_data = np.array([preprocess_image_2d(path, otsu=args.otsu) for path in data.path])

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data["letter"])
    indices = np.arange(len(data))

    train_indices, test_indices, y_train, y_test = train_test_split(
        indices,
        labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )
    train_indices, val_indices, y_train, y_val = train_test_split(
        train_indices,
        y_train,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=y_train,
    )

    train_steps = []
    if not args.no_standard_augmentation:
        train_steps.extend([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomResizedCrop(size=(64, 64), scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])
    if args.use_lf and args.use_rectangular_erasure:
        raise ValueError("Use either LF lacuna erasure or rectangular erasure, not both.")
    nlac = (args.lacuna_min, args.lacuna_max)
    if args.use_lf:
        train_steps.append(RandomLacunae(num_lacunae=nlac, size_range=(0.02, 0.15), p=0.5, v=1))
    if args.use_rect_lacuna:
        # shape control: rectangles with the same count/area/morphology as LF
        train_steps.append(RandomRectangles(num_lacunae=nlac, size_range=(0.02, 0.15), p=0.5, v=1))
    train_steps.append(transforms.ToTensor())
    if args.use_rectangular_erasure:
        # scale max matched to LF's size_range (0.15) so the RE vs LF comparison is area-matched
        train_steps.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=1.0))
    train_steps.append(transforms.Normalize((0.5,), (0.5,)))
    train_transform = transforms.Compose(train_steps)
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        ImageDatasetAugmented(image_data[train_indices], y_train, transform=train_transform),
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        ImageDatasetAugmented(image_data[val_indices], y_val, transform=eval_transform),
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        ImageDatasetAugmented(image_data[test_indices], y_test, transform=eval_transform),
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, test_loader, label_encoder


def evaluate_to_report(model, test_loader, device, label_encoder):
    model.eval()
    predictions = []
    gold = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            predictions.extend(logits.argmax(dim=1).cpu().numpy())
            gold.extend(labels.numpy())
    pred_labels = label_encoder.inverse_transform(predictions)
    gold_labels = label_encoder.inverse_transform(gold)
    return classification_report(gold_labels, pred_labels, zero_division=0, output_dict=True)


def main():
    parser = argparse.ArgumentParser(description="Train fCNN variants on Hell-Char.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "fcnn_ce")
    parser.add_argument("--checkpoint-name", default="best_fcnn.pth")
    parser.add_argument("--summary-name", default="fcnn_summary.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lambda-scl", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--use-scheduler", action="store_true")
    parser.add_argument("--scheduler-patience", type=int, default=3)
    parser.add_argument("--scheduler-factor", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--otsu", action="store_true")
    parser.add_argument("--no-standard-augmentation", action="store_true")
    parser.add_argument("--use-lf", action="store_true")
    parser.add_argument("--use-rect-lacuna", action="store_true",
                        help="Shape-only control: rectangles matched to LF (area/count/morphology).")
    parser.add_argument("--lacuna-min", type=int, default=1, help="Min number of LF/rect regions.")
    parser.add_argument("--lacuna-max", type=int, default=4, help="Max number of LF/rect regions (set 1 for single-region).")
    parser.add_argument("--use-rectangular-erasure", action="store_true")
    parser.add_argument("--use-dscl", action="store_true")
    parser.add_argument("--contrastive-loss", choices=["dscl", "scl"], default="dscl")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / args.checkpoint_name
    summary_path = args.output_dir / args.summary_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Repository root: {ROOT}", flush=True)
    print(f"Output directory: {args.output_dir}", flush=True)
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    train_loader, val_loader, test_loader, label_encoder = build_loaders(args)
    num_classes = len(label_encoder.classes_)
    print(f"Classes: {num_classes}", flush=True)
    print(f"Train/val/test: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}", flush=True)
    print(f"LF: {args.use_lf}; rectangular erasure: {args.use_rectangular_erasure}; DSCL: {args.use_dscl}", flush=True)

    model = CNN2D(num_classes=num_classes, image_size=(64, 64)).to(device)
    train_losses, val_losses, val_accuracies = train_cnn2d(
        model,
        train_loader,
        val_loader,
        device,
        num_classes,
        num_epochs=args.epochs,
        lam_scl_weight=args.lambda_scl,
        use_swscl=args.use_dscl,
        use_tta=args.use_dscl,
        tta_transform=tta_transform if args.use_dscl and not args.no_standard_augmentation else None,
        similarity_matrix_fn=custom_similarity_matrix,
        contrastive_loss=args.contrastive_loss,
        update_S_every=3,
        patience=args.patience,
        save_path=str(checkpoint_path),
        learning_rate=args.learning_rate,
        use_scheduler=args.use_scheduler,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
    )

    model = CNN2D(num_classes=num_classes, image_size=(64, 64)).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    report = evaluate_to_report(model, test_loader, device, label_encoder)

    summary = {
        "checkpoint": str(checkpoint_path),
        "classes": label_encoder.classes_.tolist(),
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
        "test_size": len(test_loader.dataset),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "test_report": report,
        "settings": {
            **vars(args),
            "model": "fCNN",
            "loss": f"cross_entropy + lambda_scl * {args.contrastive_loss}" if args.use_dscl else "cross_entropy",
            "lf_lacuna_augmentation": args.use_lf,
            "rectangular_erasure_augmentation": args.use_rectangular_erasure,
            "standard_augmentation": not args.no_standard_augmentation,
            "dscl_similarity_weighting": args.use_dscl,
        },
    }
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, default=str)

    print(json.dumps({
        "checkpoint": str(checkpoint_path),
        "summary": str(summary_path),
        "epochs": len(train_losses),
        "best_val_accuracy": max(val_accuracies),
        "test_accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
