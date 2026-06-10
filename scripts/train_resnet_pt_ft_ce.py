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
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from source import ImageDatasetAugmented, ResNetClassifier


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
    train_steps.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
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


def train(model, train_loader, val_loader, args, device, checkpoint_path):
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=args.scheduler_patience,
        factor=args.scheduler_factor,
    )
    loss_fn = nn.CrossEntropyLoss()
    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                logits = model(inputs)
                loss = loss_fn(logits, labels)
                val_loss += loss.item() * inputs.size(0)
                predicted = logits.argmax(dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_accuracy = correct / total
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Accuracy: {val_accuracy:.4f}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.", flush=True)
                break

    return history


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
    parser = argparse.ArgumentParser(description="Train plain ResNet18 with cross-entropy on Hell-Char.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "resnet18_pt_ft_ce")
    parser.add_argument("--checkpoint-name", default="best_resnet18_pt_ft_ce.pth")
    parser.add_argument("--summary-name", default="resnet18_pt_ft_ce_summary.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--scheduler-patience", type=int, default=3)
    parser.add_argument("--scheduler-factor", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--otsu", action="store_true")
    parser.add_argument("--no-standard-augmentation", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
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

    use_pretrained = not args.no_pretrained
    model = ResNetClassifier(num_classes=num_classes, pretrained=use_pretrained).to(device)
    history = train(model, train_loader, val_loader, args, device, checkpoint_path)

    model = ResNetClassifier(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    report = evaluate_to_report(model, test_loader, device, label_encoder)

    summary = {
        "checkpoint": str(checkpoint_path),
        "classes": label_encoder.classes_.tolist(),
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
        "test_size": len(test_loader.dataset),
        "history": history,
        "test_report": report,
        "settings": {
            **vars(args),
            "model": "ResNet18-PT+FT" if use_pretrained else "ResNet18",
            "pretrained": use_pretrained,
            "fine_tuned_layers": "all",
            "loss": "cross_entropy",
            "standard_augmentation": not args.no_standard_augmentation,
            "lf_lacuna_augmentation": False,
            "dscl_similarity_weighting": False,
            "standard_scl": False,
        },
    }
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, default=str)

    print(json.dumps({
        "checkpoint": str(checkpoint_path),
        "summary": str(summary_path),
        "epochs": len(history["train_loss"]),
        "best_val_accuracy": max(history["val_accuracy"]),
        "test_accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
