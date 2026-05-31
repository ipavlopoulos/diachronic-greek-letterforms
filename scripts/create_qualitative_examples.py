from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parents[1]
PAPER_IMG_DIR = ROOT.parent / "paper" / "imgs"
ARTIFACT_DIR = ROOT / "visual_artifacts"


def load_gray(path, size=120):
    img = Image.open(path).convert("L")
    img = ImageOps.contain(img, (size, size))
    canvas = Image.new("L", (size, size), 255)
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def standard_erasure(img):
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle((38, 24, 78, 58), fill=245)
    draw.rectangle((74, 72, 108, 102), fill=245)
    return out


def lacuna_erasure(img):
    out = img.copy()
    rng = np.random.default_rng(11)
    mask = Image.new("L", out.size, 0)
    draw = ImageDraw.Draw(mask)
    for cx, cy, rx, ry, angle in [(48, 42, 22, 14, 20), (86, 82, 18, 24, -35)]:
        blob = Image.new("L", out.size, 0)
        bdraw = ImageDraw.Draw(blob)
        bdraw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=255)
        arr = np.array(blob)
        noise = rng.normal(0, 1, arr.shape)
        arr = np.where((arr > 0) & (noise > -1.2), 255, arr)
        blob = Image.fromarray(arr.astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.4))
        mask = Image.composite(Image.new("L", out.size, 255), mask, blob)
    out_arr = np.array(out)
    mask_arr = np.array(mask)
    out_arr[mask_arr > 50] = 245
    return Image.fromarray(out_arr)


def show_image(ax, path_or_img, title, subtitle=None):
    img = path_or_img if isinstance(path_or_img, Image.Image) else load_gray(path_or_img)
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#d0d0d0")
        spine.set_linewidth(0.8)
    ax.set_title(title, fontsize=10, pad=5)
    if subtitle:
        ax.text(
            0.5,
            -0.10,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )


def main():
    PAPER_IMG_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    paths = {
        "aug": ROOT / "data/hellchar/cliplets/Alpha_1137_571.jpg",
        "damaged": ROOT / "data/hellchar/cliplets/Omega_18552_12609.jpg",
        "alpha_a": ROOT / "data/hellchar/cliplets/Alpha_5287_267.jpg",
        "lambda": ROOT / "data/hellchar/cliplets/Lambda_1526_5821.jpg",
        "alpha_b": ROOT / "data/hellchar/cliplets/Alpha_3098_449.jpg",
        "phi": ROOT / "data/hellchar/cliplets/Phi_951_11908.jpg",
        "gamma_hell": ROOT / "data/hellchar/cliplets/Gamma_11315_1148.jpg",
        "gamma_palit": ROOT / "data/palitchar/cliplets/Gamma_47181_106.jpg",
        "gamma_med_a": ROOT / "data/medchar/cliplets/Gamma_13030_161.jpg",
        "gamma_med_b": ROOT / "data/medchar/cliplets/Gamma_10009_150.jpg",
        "background": ROOT / "data/palitchar/cliplets/Gamma_220465_104.png",
        "degraded": ROOT / "data/hellchar/cliplets/Beta_250_996.jpg",
        "minuscule": ROOT / "data/medchar/cliplets/Beta_12004_037.jpg",
    }

    original = load_gray(paths["aug"])
    rows = [
        [
            ("Original", original, "Hell-Char Alpha"),
            ("Standard erasure", standard_erasure(original), "rectangular masks"),
            ("LF lacuna erasure", lacuna_erasure(original), "irregular lacunae"),
            ("Real damage", paths["damaged"], "fragmented surface"),
        ],
        [
            ("Alpha", paths["alpha_a"], "cursive wedge"),
            ("Lambda", paths["lambda"], "similar stroke angle"),
            ("Alpha", paths["alpha_b"], "ligatured form"),
            ("Phi", paths["phi"], "vertical stroke overlap"),
        ],
        [
            ("Gamma, Hell-Char", paths["gamma_hell"], "3rd-1st c. BCE"),
            ("Gamma, PaLit-Char", paths["gamma_palit"], "2nd-5th c. CE"),
            ("Gamma, Med-Char", paths["gamma_med_a"], "9th-14th c. CE"),
            ("Gamma, Med-Char", paths["gamma_med_b"], "minuscule variant"),
        ],
        [
            ("Cleaner background", paths["gamma_med_a"], "high contrast"),
            ("Ruled background", paths["background"], "bookhand texture"),
            ("Degradation", paths["degraded"], "ink/support noise"),
            ("Later form", paths["minuscule"], "minuscule drift"),
        ],
    ]

    row_labels = [
        "A. Augmentation and real damage",
        "B. Confusable letterforms",
        "C. Gamma across datasets",
        "D. Visual variability",
    ]

    fig, axes = plt.subplots(4, 4, figsize=(9.2, 10.2), constrained_layout=False)
    for r, row in enumerate(rows):
        axes[r, 0].text(
            0.0,
            1.11,
            row_labels[r],
            transform=axes[r, 0].transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
        for c, (title, item, subtitle) in enumerate(row):
            show_image(axes[r, c], item, title, subtitle)

    fig.subplots_adjust(left=0.04, right=0.99, top=0.94, bottom=0.04, hspace=0.68, wspace=0.20)
    outputs = [
        PAPER_IMG_DIR / "qualitative_visual_examples.png",
        ARTIFACT_DIR / "qualitative_visual_examples.png",
    ]
    for output in outputs:
        fig.savefig(output, dpi=300)
        print(output)


if __name__ == "__main__":
    main()
