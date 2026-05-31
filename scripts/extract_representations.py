import argparse
import csv
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source import LETTER_LABELS, extract_letterform_representations, load_letterform_model


def collect_image_paths(inputs):
    image_paths = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            image_paths.extend(sorted(path.glob("*.jpg")))
            image_paths.extend(sorted(path.glob("*.jpeg")))
            image_paths.extend(sorted(path.glob("*.png")))
        else:
            image_paths.append(path)
    return image_paths


def write_csv(output_path, image_paths, embeddings, logits):
    probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()
    predictions = probabilities.argmax(axis=1)

    with open(output_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["filename", "predicted_label", "confidence"]
        header.extend([f"embedding_{i:03d}" for i in range(embeddings.shape[1])])
        writer.writerow(header)

        for path, embedding, pred_idx, probs in zip(image_paths, embeddings, predictions, probabilities):
            writer.writerow([
                str(path),
                LETTER_LABELS[int(pred_idx)],
                float(probs[int(pred_idx)]),
                *embedding.astype(float).tolist(),
            ])


def main():
    parser = argparse.ArgumentParser(
        description="Extract 512-dimensional Greek letterform representations from cliplet images."
    )
    parser.add_argument("images", nargs="+", help="Image files or directories containing .jpg/.jpeg/.png cliplets.")
    parser.add_argument("--checkpoint", default="best_cnn_letter_model.pth", help="Path to the released CNN2D checkpoint.")
    parser.add_argument("--output", default="letterform_representations.csv", help="CSV file to write.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for representation extraction.")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cpu or cuda. Defaults to auto-detection.")
    args = parser.parse_args()

    image_paths = collect_image_paths(args.images)
    if not image_paths:
        raise SystemExit("No input images found.")

    model = load_letterform_model(args.checkpoint, device=args.device)
    embeddings, logits = extract_letterform_representations(
        model,
        image_paths,
        batch_size=args.batch_size,
        return_logits=True,
    )
    write_csv(args.output, image_paths, embeddings, logits)
    print(f"Wrote {len(image_paths)} representations to {args.output}")


if __name__ == "__main__":
    main()
