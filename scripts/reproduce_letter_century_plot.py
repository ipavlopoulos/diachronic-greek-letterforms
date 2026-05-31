import argparse
import base64
import html
import io
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from sklearn.manifold import TSNE

from source import ResNetClassifier, image_path_to_tensor


LETTER_ORDER = [
    "Alpha", "Beta", "Chi", "Delta", "Epsilon", "Eta", "Gamma", "Iota",
    "Kappa", "Lambda", "Mu", "Nu", "Omega", "Omicron", "Phi", "Pi",
    "Psi", "Rho", "Sigma", "Tau", "Theta", "Upsilon", "Xi", "Zeta",
]


CENTURY_COLORS = {
    9: "#67000d",
    10: "#a50f15",
    11: "#cb181d",
    12: "#ef3b2c",
    13: "#fb6a4a",
    14: "#fc9272",
}


def extract_embeddings(model, paths, device, batch_size):
    embeddings = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start:start + batch_size]
            batch = torch.cat([image_path_to_tensor(path, device=device) for path in batch_paths], dim=0)
            emb = F.normalize(model.get_embeddings(batch), dim=1)
            embeddings.append(emb.cpu().numpy())
    return np.vstack(embeddings)


def make_thumbnail_data_uri(path, size):
    img = Image.open(path).convert("RGB")
    img = ImageOps.contain(img, (size, size))
    canvas = Image.new("RGB", (size, size), "#ffffff")
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    handle = io.BytesIO()
    canvas.save(handle, format="PNG", optimize=True)
    encoded = base64.b64encode(handle.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def scale_points(coords, width, height, margin):
    x = coords[:, 0]
    y = coords[:, 1]
    x_scaled = margin + (x - x.min()) / (x.max() - x.min()) * (width - 2 * margin)
    y_scaled = height - margin - (y - y.min()) / (y.max() - y.min()) * (height - 2 * margin)
    return np.column_stack([x_scaled, y_scaled])


def choose_prototypes(df, points):
    rows = []
    for (letter, century), group in df.groupby(["letter", "century"], sort=False):
        idx = group.index.to_numpy()
        centroid = points[idx].mean(axis=0)
        nearest = idx[np.linalg.norm(points[idx] - centroid, axis=1).argmin()]
        rows.append({
            "index": int(nearest),
            "letter": letter,
            "century": int(century),
            "x": float(points[nearest, 0]),
            "y": float(points[nearest, 1]),
            "filename": df.loc[nearest, "filename"],
            "year": int(df.loc[nearest, "year"]),
            "path": df.loc[nearest, "path"],
        })
    rows.sort(key=lambda row: (row["century"], LETTER_ORDER.index(row["letter"]) if row["letter"] in LETTER_ORDER else 999))
    return rows


def layout_boxes(prototypes, width, height, margin, thumb_size, label_height):
    box_w = thumb_size
    box_h = thumb_size + label_height + 6
    boxes = []
    for i, proto in enumerate(prototypes):
        angle = (i % 8) * np.pi / 4
        radius = 20 + 8 * ((i // 8) % 3)
        x = proto["x"] + np.cos(angle) * radius - box_w / 2
        y = proto["y"] + np.sin(angle) * radius - box_h / 2
        boxes.append([x, y, box_w, box_h])

    min_x, max_x = margin * 0.30, width - margin * 0.30 - box_w
    min_y, max_y = margin * 0.45, height - margin * 0.20 - box_h

    for _ in range(260):
        moved = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                ax, ay, aw, ah = boxes[i]
                bx, by, bw, bh = boxes[j]
                overlap_x = min(ax + aw, bx + bw) - max(ax, bx)
                overlap_y = min(ay + ah, by + bh) - max(ay, by)
                if overlap_x > 0 and overlap_y > 0:
                    moved = True
                    acx, acy = ax + aw / 2, ay + ah / 2
                    bcx, bcy = bx + bw / 2, by + bh / 2
                    dx = acx - bcx
                    dy = acy - bcy
                    if abs(dx) + abs(dy) < 1e-6:
                        dx, dy = 1.0, 0.0
                    norm = (dx * dx + dy * dy) ** 0.5
                    push = min(overlap_x, overlap_y) * 0.58 + 0.7
                    ux, uy = dx / norm, dy / norm
                    boxes[i][0] += ux * push
                    boxes[i][1] += uy * push
                    boxes[j][0] -= ux * push
                    boxes[j][1] -= uy * push
        for box in boxes:
            box[0] = min(max(box[0], min_x), max_x)
            box[1] = min(max(box[1], min_y), max_y)
        if not moved:
            break
    return boxes


def svg_text(x, y, text, **attrs):
    attr_parts = []
    for key, value in attrs.items():
        attr_name = key[:-1] if key.endswith("_") else key
        attr_parts.append(f'{attr_name.replace("_", "-")}="{html.escape(str(value))}"')
    attr = " ".join(attr_parts)
    return f'<text x="{x:.2f}" y="{y:.2f}" {attr}>{html.escape(text)}</text>'


def make_svg(df, points, prototypes, output_svg, width, height, margin, thumb_size):
    label_height = 16
    boxes = layout_boxes(prototypes, width, height, margin, thumb_size, label_height)
    elements = []
    elements.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    elements.append("<style>")
    elements.append("""
        .axis { stroke: #d4d4d4; stroke-width: 1; }
        .point { opacity: 0.46; stroke: #fff; stroke-width: 0.35; }
        .leader { stroke: #777; stroke-width: 0.75; opacity: 0.50; }
        .thumb-frame { fill: #fff; stroke: #333; stroke-width: 0.85; rx: 2; }
        .label-bg { fill: #fff; stroke: #bbb; stroke-width: 0.4; opacity: 0.95; }
        .label { font-family: Arial, Helvetica, sans-serif; font-size: 10px; font-weight: 700; fill: #111; text-anchor: middle; }
        .small-label { font-family: Arial, Helvetica, sans-serif; font-size: 12px; fill: #333; }
        .title { font-family: Arial, Helvetica, sans-serif; font-size: 22px; font-weight: 700; fill: #111; text-anchor: middle; }
    """)
    elements.append("</style>")
    elements.append('<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>')
    elements.append(svg_text(width / 2, 32, "Med-Char letter-century prototypes in ResNet18+LF+DSCL embedding space", class_="title"))
    elements.append(f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" class="axis"/>')
    elements.append(f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" class="axis"/>')

    for century in sorted(df["century"].unique()):
        subset = df[df["century"] == century]
        color = CENTURY_COLORS.get(int(century), "#666666")
        for idx in subset.index:
            x, y = points[idx]
            elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.2" fill="{color}" class="point"><title>{html.escape(df.loc[idx, "letter"])} {int(df.loc[idx, "year"])}</title></circle>')

    legend_x = width - margin + 18
    legend_y = margin + 28
    elements.append(svg_text(legend_x - 5, legend_y - 18, "older = darker", class_="small-label"))
    for k, century in enumerate(sorted(df["century"].unique())):
        y = legend_y + k * 22
        color = CENTURY_COLORS.get(int(century), "#666666")
        elements.append(f'<circle cx="{legend_x}" cy="{y}" r="6.2" fill="{color}" opacity="0.90" stroke="#fff" stroke-width="0.5"/>')
        elements.append(svg_text(legend_x + 12, y + 4, f"{int(century)}th c.", class_="small-label"))

    for proto, box in zip(prototypes, boxes):
        x, y, _, _ = box
        anchor_x, anchor_y = proto["x"], proto["y"]
        center_x = x + thumb_size / 2
        center_y = y + label_height + 3 + thumb_size / 2
        color = CENTURY_COLORS.get(proto["century"], "#666666")
        label = f'{proto["letter"]} {proto["century"]}c'
        image_y = y + label_height + 4
        data_uri = make_thumbnail_data_uri(proto["path"], thumb_size)
        elements.append(f'<line x1="{anchor_x:.2f}" y1="{anchor_y:.2f}" x2="{center_x:.2f}" y2="{center_y:.2f}" class="leader"/>')
        elements.append(f'<circle cx="{anchor_x:.2f}" cy="{anchor_y:.2f}" r="5.5" fill="{color}" opacity="0.92" stroke="#fff" stroke-width="0.6"/>')
        elements.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{thumb_size:.2f}" height="{label_height:.2f}" class="label-bg" stroke="{color}"/>')
        elements.append(svg_text(x + thumb_size / 2, y + 11.7, label, class_="label"))
        elements.append(f'<rect x="{x:.2f}" y="{image_y:.2f}" width="{thumb_size:.2f}" height="{thumb_size:.2f}" class="thumb-frame" stroke="{color}"/>')
        elements.append(f'<image x="{x:.2f}" y="{image_y:.2f}" width="{thumb_size:.2f}" height="{thumb_size:.2f}" href="{data_uri}">')
        elements.append(f'<title>{html.escape(proto["filename"])}; year {proto["year"]}</title>')
        elements.append("</image>")

    elements.append(svg_text(margin, height - 18, "t-SNE 1", class_="small-label"))
    elements.append(svg_text(22, margin + 6, "t-SNE 2", class_="small-label", transform=f"rotate(-90 22 {margin + 6})"))
    elements.append("</svg>")
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    output_svg.write_text("\n".join(elements), encoding="utf-8")


def make_html(svg_path, html_path):
    svg = svg_path.read_text(encoding="utf-8")
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Letter-Century Plot</title>
  <style>
    body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; background: #f6f6f6; color: #222; }}
    header {{ padding: 14px 20px; background: #fff; border-bottom: 1px solid #ddd; }}
    main {{ padding: 16px; overflow: auto; }}
    .figure {{ background: #fff; border: 1px solid #ddd; width: max-content; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
    p {{ margin: 4px 0 0; font-size: 14px; }}
  </style>
</head>
<body>
  <header>
    <strong>Med-Char Letter-Century Plot</strong>
    <p>Fixed-size colour prototype thumbnails; marker colours use darker reds for earlier centuries.</p>
  </header>
  <main>
    <div class="figure">{svg}</div>
  </main>
</body>
</html>
"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_doc, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Recreate the Med-Char letter-century t-SNE plot as SVG and HTML.")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "resnet_lf_dscl_lr1e-4_bs16" / "best_resnet_lf_dscl_model.pth")
    parser.add_argument("--summary", type=Path, default=ROOT / "runs" / "resnet_lf_dscl_lr1e-4_bs16" / "resnet_lf_dscl_summary.json")
    parser.add_argument("--medchar-csv", type=Path, default=ROOT / "data" / "medchar" / "medchar.csv")
    parser.add_argument("--cliplet-dir", type=Path, default=ROOT / "data" / "medchar" / "cliplets")
    parser.add_argument("--svg-output", type=Path, default=ROOT / "visual_artifacts" / "letter_century_plot_resnet_reproduced.svg")
    parser.add_argument("--html-output", type=Path, default=ROOT / "visual_artifacts" / "letter_century_plot_resnet_reproduced.html")
    parser.add_argument("--paper-svg-output", type=Path, default=PROJECT_ROOT / "paper" / "imgs" / "letter_century_plot_resnet_reproduced.svg")
    parser.add_argument("--coords-output", type=Path, default=ROOT / "runs" / "resnet_lf_dscl_lr1e-4_bs16" / "letter_century_tsne_coordinates.csv")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--width", type=int, default=1900)
    parser.add_argument("--height", type=int, default=1600)
    parser.add_argument("--thumbnail-size", type=int, default=44)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    with open(args.summary) as handle:
        classes = json.load(handle)["classes"]

    df = pd.read_csv(args.medchar_csv)
    df = df[df["letter"].isin(classes)].copy().reset_index(drop=True)
    df["century"] = ((df["year"] - 1) // 100 + 1).astype(int)
    df["path"] = df["filename"].apply(lambda name: args.cliplet_dir / name)

    model = ResNetClassifier(num_classes=len(classes), pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    embeddings = extract_embeddings(model, df["path"].tolist(), device, args.batch_size)

    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        init="pca",
        learning_rate="auto",
        random_state=args.random_state,
    )
    coords = tsne.fit_transform(embeddings)
    points = scale_points(coords, args.width, args.height, margin=90)
    df["tsne_x"] = coords[:, 0]
    df["tsne_y"] = coords[:, 1]
    df["plot_x"] = points[:, 0]
    df["plot_y"] = points[:, 1]
    args.coords_output.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["path"]).to_csv(args.coords_output, index=False)

    prototypes = choose_prototypes(df, points)
    make_svg(df, points, prototypes, args.svg_output, args.width, args.height, margin=90, thumb_size=args.thumbnail_size)
    if args.paper_svg_output:
        args.paper_svg_output.parent.mkdir(parents=True, exist_ok=True)
        args.paper_svg_output.write_text(args.svg_output.read_text(encoding="utf-8"), encoding="utf-8")
    make_html(args.svg_output, args.html_output)
    print(f"Wrote {args.svg_output}")
    print(f"Wrote {args.html_output}")
    if args.paper_svg_output:
        print(f"Wrote {args.paper_svg_output}")
    print(f"Wrote {args.coords_output}")
    print(f"Prototype thumbnails: {len(prototypes)}")


if __name__ == "__main__":
    main()
