"""Shared Hell-Char helpers used by the figure-reproduction scripts
(``reproduce_letter_forms.py`` and ``reproduce_confusion_similarity.py``).

NORMALIZATION. Images are loaded as grayscale [0,1] arrays and mapped to the
model's [-1,1] input range with the training-time ``Normalize((0.5,), (0.5,))``
(via ``to_model_input``), consistent with ``evaluate.py``, the t-SNE figure, and
``source.test_transform``. (The earlier released version of these figure scripts
fed un-normalized [0,1] inputs; this matches the trained model instead.)
"""
import random

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess_gray(path, size=(64, 64), otsu=False):
    """Load a cliplet as a grayscale [0,1] array (no normalization yet)."""
    img_np = np.array(Image.open(path).convert("L"))
    if otsu:
        _, img_np = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_np = 255 - img_np
    img_resized = cv2.resize(img_np, size, interpolation=cv2.INTER_AREA)
    return img_resized.astype(np.float32) / 255.0


def to_model_input(images):
    """Map grayscale [0,1] array(s) to the model's [-1,1] input range.

    Matches the training-time ``Normalize((0.5,), (0.5,))``.
    """
    return (torch.as_tensor(images, dtype=torch.float32) - 0.5) / 0.5


def load_hellchar(data_dir):
    cliplet_dir = data_dir / "hellchar" / "cliplets"
    paths = sorted(
        path for path in cliplet_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    data = pd.DataFrame({"path": paths, "filename": [path.name for path in paths]})
    data["letter"] = data.filename.apply(lambda value: value.split("_")[0])
    return data[data["letter"] != "Unknown"].copy()


def split_indices(labels, seed, test_size, val_size):
    indices = np.arange(len(labels))
    train_indices, test_indices, y_train, y_test = train_test_split(
        indices, labels, test_size=test_size, random_state=seed, stratify=labels)
    train_indices, val_indices, y_train, y_val = train_test_split(
        train_indices, y_train, test_size=val_size, random_state=seed, stratify=y_train)
    return train_indices, val_indices, test_indices, y_train, y_val, y_test


class _ArrayDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return to_model_input(self.images[index]), int(self.labels[index])


def extract_model_embeddings(model, images, labels, device, batch_size, num_workers):
    """L2-normalized backbone embeddings for [0,1] grayscale image arrays.

    Inputs are mapped to [-1,1] with the training-time ``Normalize((0.5,),(0.5,))``
    so the embeddings match the trained model (see the module note)."""
    loader = DataLoader(_ArrayDataset(images, labels), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers,
                        pin_memory=torch.cuda.is_available())
    embeddings, out_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch, y in loader:
            z = F.normalize(model.get_embeddings(batch.to(device)), dim=1)
            embeddings.append(z.cpu().numpy())
            out_labels.append(y.numpy())
    return np.vstack(embeddings), np.concatenate(out_labels)
