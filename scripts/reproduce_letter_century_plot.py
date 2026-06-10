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
from scripts.train_timm_backbone import TimmClassifier


LETTER_ORDER = [
    "Alpha", "Beta", "Chi", "Delta", "Epsilon", "Eta", "Gamma", "Iota",
    "Kappa", "Lambda", "Mu", "Nu", "Omega", "Omicron", "Phi", "Pi",
    "Psi", "Rho", "Sigma", "Tau", "Theta", "Upsilon", "Xi", "Zeta",
]


CENTURY_COLORS = {
    9: "#2166ac",
    10: "#4393c3",
    11: "#92c5de",
    12: "#f4a582",
    13: "#d6604d",
    14: "#b2182b",
}

CENTURY_MARKERS = {
    9: "circle",
    10: "square",
    11: "triangle",
    12: "diamond",
    13: "triangle-down",
    14: "hexagon",
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


def svg_marker(shape, cx, cy, r, fill, cls="point", title=None):
    t = f"<title>{html.escape(title)}</title>" if title else ""
    common = f'fill="{fill}" class="{cls}"'
    if shape == "square":
        s = r * 0.92
        return f'<rect x="{cx-s:.2f}" y="{cy-s:.2f}" width="{2*s:.2f}" height="{2*s:.2f}" {common}>{t}</rect>'
    if shape == "triangle":
        p = f"{cx:.2f},{cy-r*1.18:.2f} {cx-r*1.05:.2f},{cy+r*0.85:.2f} {cx+r*1.05:.2f},{cy+r*0.85:.2f}"
        return f'<polygon points="{p}" {common}>{t}</polygon>'
    if shape == "triangle-down":
        p = f"{cx:.2f},{cy+r*1.18:.2f} {cx-r*1.05:.2f},{cy-r*0.85:.2f} {cx+r*1.05:.2f},{cy-r*0.85:.2f}"
        return f'<polygon points="{p}" {common}>{t}</polygon>'
    if shape == "diamond":
        p = f"{cx:.2f},{cy-r*1.28:.2f} {cx+r*1.05:.2f},{cy:.2f} {cx:.2f},{cy+r*1.28:.2f} {cx-r*1.05:.2f},{cy:.2f}"
        return f'<polygon points="{p}" {common}>{t}</polygon>'
    if shape == "hexagon":
        pts = []
        for k in range(6):
            a = np.pi / 6 + k * np.pi / 3
            pts.append(f"{cx + r*1.05*np.cos(a):.2f},{cy + r*1.05*np.sin(a):.2f}")
        return f'<polygon points="{" ".join(pts)}" {common}>{t}</polygon>'
    return f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" {common}>{t}</circle>'


def make_svg(df, points, prototypes, output_svg, width, height, margin, thumb_size):
    label_height = 20
    boxes = layout_boxes(prototypes, width, height, margin, thumb_size, label_height)
    elements = []
    elements.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    elements.append("<style>")
    elements.append("""
        .grid { stroke: #ececec; stroke-width: 0.8; }
        .axis { stroke: #d4d4d4; stroke-width: 1; }
        .point { opacity: 0.50; stroke: #fff; stroke-width: 0.45; }
        .leader { stroke: #777; stroke-width: 0.85; opacity: 0.50; }
        .thumb-frame { fill: #fff; stroke: #333; stroke-width: 0.85; rx: 2; }
        .label-bg { fill: #fff; stroke: #bbb; stroke-width: 0.4; opacity: 0.95; }
        .label { font-family: Arial, Helvetica, sans-serif; font-size: 12px; font-weight: 700; fill: #111; text-anchor: middle; }
        .small-label { font-family: Arial, Helvetica, sans-serif; font-size: 15px; fill: #333; }
        .legend-label { font-family: Arial, Helvetica, sans-serif; font-size: 26px; font-weight: 600; fill: #222; }
        .legend-marker { opacity: 0.95; stroke: #fff; stroke-width: 0.8; }
    """)
    elements.append("</style>")
    elements.append('<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>')
    for step in range(1, 6):
        x = margin + step * (width - 2 * margin) / 6
        y = margin + step * (height - 2 * margin) / 6
        elements.append(f'<line x1="{x:.2f}" y1="{margin}" x2="{x:.2f}" y2="{height-margin}" class="grid"/>')
        elements.append(f'<line x1="{margin}" y1="{y:.2f}" x2="{width-margin}" y2="{y:.2f}" class="grid"/>')
    elements.append(f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" class="axis"/>')
    elements.append(f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" class="axis"/>')

    for century in sorted(df["century"].unique()):
        subset = df[df["century"] == century]
        color = CENTURY_COLORS.get(int(century), "#666666")
        shape = CENTURY_MARKERS.get(int(century), "circle")
        for idx in subset.index:
            x, y = points[idx]
            ttl = f'{df.loc[idx, "letter"]} {int(df.loc[idx, "year"])}'
            elements.append(svg_marker(shape, x, y, 6.4, color, "point", ttl))

    legend_x = width - margin - 130
    legend_y = margin + 34
    elements.append(svg_text(width - margin, legend_y - 26, "blue = old, red = recent", class_="legend-label", text_anchor="end"))
    for k, century in enumerate(sorted(df["century"].unique())):
        y = legend_y + k * 34
        color = CENTURY_COLORS.get(int(century), "#666666")
        shape = CENTURY_MARKERS.get(int(century), "circle")
        elements.append(svg_marker(shape, legend_x, y, 12, color, "legend-marker"))
        elements.append(svg_text(legend_x + 24, y + 9, f"{int(century)}th c.", class_="legend-label"))

    for proto, box in zip(prototypes, boxes):
        x, y, _, _ = box
        anchor_x, anchor_y = proto["x"], proto["y"]
        center_x = x + thumb_size / 2
        center_y = y + label_height + 3 + thumb_size / 2
        color = CENTURY_COLORS.get(proto["century"], "#666666")
        shape = CENTURY_MARKERS.get(proto["century"], "circle")
        label = f'{proto["letter"]} {proto["century"]}c'
        image_y = y + label_height + 4
        data_uri = make_thumbnail_data_uri(proto["path"], thumb_size)
        elements.append(f'<line x1="{anchor_x:.2f}" y1="{anchor_y:.2f}" x2="{center_x:.2f}" y2="{center_y:.2f}" class="leader"/>')
        elements.append(svg_marker(shape, anchor_x, anchor_y, 8.0, color, "legend-marker"))
        elements.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{thumb_size:.2f}" height="{label_height:.2f}" class="label-bg" stroke="{color}"/>')
        elements.append(svg_text(x + thumb_size / 2, y + 14.2, label, class_="label"))
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
    <p>Fixed-size colour prototype thumbnails; marker colours run from blue (older) to red (more recent).</p>
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
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "resnet_lf_dscl" / "best_resnet_lf_dscl_model.pth")
    parser.add_argument("--summary", type=Path, default=ROOT / "models" / "resnet_lf_dscl" / "resnet_lf_dscl_summary.json")
    parser.add_argument("--medchar-csv", type=Path, default=ROOT / "data" / "medchar" / "medchar.csv")
    parser.add_argument("--cliplet-dir", type=Path, default=ROOT / "data" / "medchar" / "cliplets")
    parser.add_argument("--svg-output", type=Path, default=ROOT / "visual_artifacts" / "letter_century_plot_resnet_reproduced.svg")
    parser.add_argument("--html-output", type=Path, default=ROOT / "visual_artifacts" / "letter_century_plot_resnet_reproduced.html")
    parser.add_argument("--paper-svg-output", type=Path, default=PROJECT_ROOT / "paper" / "imgs" / "letter_century_plot_resnet_reproduced.svg")
    parser.add_argument("--coords-output", type=Path, default=ROOT / "runs" / "resnet_lf_dscl" / "letter_century_tsne_coordinates.csv")
    parser.add_argument("--device", default=None)
    parser.add_argument("--timm-model", default=None, help="If set, load a TimmClassifier (e.g. convnextv2_tiny) instead of ResNet.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--width", type=int, default=2400)
    parser.add_argument("--height", type=int, default=2000)
    parser.add_argument("--thumbnail-size", type=int, default=74)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    with open(args.summary) as handle:
        classes = json.load(handle)["classes"]

    df = pd.read_csv(args.medchar_csv)
    df = df[df["letter"].isin(classes)].copy().reset_index(drop=True)
    df["century"] = ((df["year"] - 1) // 100 + 1).astype(int)
    df["path"] = df["filename"].apply(lambda name: args.cliplet_dir / name)

    if args.timm_model:
        model = TimmClassifier(args.timm_model, len(classes), pretrained=False, in_chans=1, image_size=64).to(device)
    else:
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
    points = scale_points(coords, args.width, args.height, margin=100)
    df["tsne_x"] = coords[:, 0]
    df["tsne_y"] = coords[:, 1]
    df["plot_x"] = points[:, 0]
    df["plot_y"] = points[:, 1]
    args.coords_output.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["path"]).to_csv(args.coords_output, index=False)

    prototypes = choose_prototypes(df, points)
    make_svg(df, points, prototypes, args.svg_output, args.width, args.height, margin=100, thumb_size=args.thumbnail_size)
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
