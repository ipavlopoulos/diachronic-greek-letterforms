"""Compact 3-letter augmentation figure: rows = Alpha/Beta/Gamma,
cols = Original | Standard erasure | LF lacuna | Real damage."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parents[1]
PAPER_IMG_DIR = ROOT.parent / "paper" / "imgs"
CL = ROOT / "data/hellchar/cliplets"

def load_gray(path, size=120):
    img = Image.open(path).convert("L")
    img = ImageOps.contain(img, (size, size))
    canvas = Image.new("L", (size, size), 255)
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas

def standard_erasure(img, seed):
    rng = np.random.default_rng(seed); out = img.copy(); d = ImageDraw.Draw(out)
    for _ in range(2):
        w = rng.integers(28, 42); h = rng.integers(24, 40)
        x = rng.integers(8, 120 - w); y = rng.integers(8, 120 - h)
        d.rectangle((x, y, x + w, y + h), fill=245)
    return out

def lacuna_erasure(img, seed):
    out = img.copy(); rng = np.random.default_rng(seed)
    mask = Image.new("L", out.size, 0)
    for _ in range(2):
        cx, cy = rng.integers(30, 90), rng.integers(30, 90)
        rx, ry = rng.integers(14, 26), rng.integers(14, 26)
        ang = rng.integers(0, 180)
        blob = Image.new("L", out.size, 0)
        ImageDraw.Draw(blob).ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=255)
        blob = blob.rotate(ang)
        arr = np.array(blob); noise = rng.normal(0, 1, arr.shape)
        arr = np.where((arr > 0) & (noise > -1.2), 255, arr)
        blob = Image.fromarray(arr.astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.4))
        mask = Image.composite(Image.new("L", out.size, 255), mask, blob)
    oa = np.array(out); ma = np.array(mask); oa[ma > 50] = 245
    return Image.fromarray(oa)

# (letter, clean original, real-damaged instance)
ROWS = [
    ("Alpha", "Alpha_1011_574.jpg", "Alpha_112437_48.jpg"),
    ("Beta",  "Beta_1011_962.jpg",  "Beta_250_996.jpg"),
    ("Gamma", "Gamma_1011_1465.jpg","Gamma_112437_1059.jpg"),
]
COLS = ["Original", "Standard erasure", "LF lacuna", "Real damage"]

fig, axes = plt.subplots(3, 4, figsize=(5.6, 4.3))
for r, (L, clean, dmg) in enumerate(ROWS):
    orig = load_gray(CL / clean)
    cells = [orig, standard_erasure(orig, r + 1), lacuna_erasure(orig, r + 7), load_gray(CL / dmg)]
    for c, im in enumerate(cells):
        ax = axes[r, c]
        ax.imshow(im, cmap="gray", vmin=0, vmax=255); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color("#cccccc"); sp.set_linewidth(0.6)
        if r == 0: ax.set_title(COLS[c], fontsize=8, pad=3)
        if c == 0: ax.set_ylabel(L, fontsize=9, rotation=90, labelpad=4)
fig.subplots_adjust(left=0.06, right=0.995, top=0.93, bottom=0.01, hspace=0.08, wspace=0.06)
out = ROOT / "lf_augmentation_examples3.png"
fig.savefig(out, dpi=300); print("wrote", out)
