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
    ImageDatasetAugmented,
    RandomLacunae,
    ResNetClassifier,
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
    data["letter"] = data.filename.apply(lambda x: x.split("_")[0])
    data["TM"] = data.filename.apply(lambda x: int(x.split("_")[1]))
    data["number"] = data.filename.apply(lambda x: x.split("_")[2].split(".")[0])
    return data


def build_loaders(args):
    data = load_hellchar(args.data_dir)
    image_data = np.array([preprocess_image_2d(path, otsu=args.otsu) for path in data.path])

    known_data = data[data["letter"] != "Unknown"].copy()
    known_indices = known_data.index.tolist()
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(known_data["letter"])

    train_indices, test_indices, y_train, y_test = train_test_split(
        known_indices,
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

    data_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomResizedCrop(size=(64, 64), scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        RandomLacunae(num_lacunae=(0, 2), size_range=(0.02, 0.12), p=0.5, v=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_loader = DataLoader(
        ImageDatasetAugmented(image_data[train_indices], y_train, transform=data_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        ImageDatasetAugmented(image_data[val_indices], y_val, transform=test_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        ImageDatasetAugmented(image_data[test_indices], y_test, transform=test_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader, label_encoder


def evaluate_to_report(model, test_loader, device, label_encoder):
    model.eval()
    pred, gold = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predicted = outputs.argmax(dim=1).cpu().numpy()
            pred.extend(predicted)
            gold.extend(labels.numpy())
    pred_labels = label_encoder.inverse_transform(pred)
    gold_labels = label_encoder.inverse_transform(gold)
    return classification_report(gold_labels, pred_labels, zero_division=0, output_dict=True)


def main():
    parser = argparse.ArgumentParser(description="Train ResNet-18 with LF augmentation and DSCL on Hell-Char.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "resnet_lf_dscl")
    parser.add_argument("--checkpoint-name", default="best_resnet_lf_dscl_model.pth")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--lambda-scl", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--otsu", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / args.checkpoint_name
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

    model = ResNetClassifier(num_classes=num_classes, pretrained=not args.no_pretrained).to(device)
    train_losses, val_losses, val_accs = train_cnn2d(
        model,
        train_loader,
        val_loader,
        device,
        num_classes,
        num_epochs=args.epochs,
        lam_scl_weight=args.lambda_scl,
        use_swscl=True,
        use_tta=True,
        tta_transform=tta_transform,
        similarity_matrix_fn=custom_similarity_matrix,
        update_S_every=3,
        patience=args.patience,
        save_path=str(checkpoint_path),
        learning_rate=args.learning_rate,
    )

    model = ResNetClassifier(num_classes=num_classes, pretrained=False).to(device)
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
        "val_accuracies": val_accs,
        "test_report": report,
        "settings": vars(args),
    }
    summary_path = args.output_dir / "resnet_lf_dscl_summary.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, default=str)

    print(json.dumps({
        "checkpoint": str(checkpoint_path),
        "summary": str(summary_path),
        "test_accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
