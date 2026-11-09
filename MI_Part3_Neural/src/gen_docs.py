"""Generate synthetic scanned-document images for AI Part 2 (perception).

Three document types the health plan's mailroom receives:
  claim_form       dense grid of boxes plus a line-item table (the bill)
  referral         letterhead, paragraphs of text and a signature (doctor sends a member to a specialist)
  enrollment_form  title bar, sections with checkboxes and an ID photo box (a new provider joining the network)
Text is drawn as gray bars, so no real form, logo or person appears. Seeded (seed 42).

Outputs (data/):
  images/*.png      64 x 64 grayscale
  labels.csv        filename, label, set  (set = train / val / test / messy)
The "messy" set is scanned badly on purpose (more tilt, blur, noise, shadows) to test robustness.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
W, H, SIZE = 200, 260, 64
CLASSES = ["claim_form", "referral", "enrollment_form"]


def text_line(d, rng, x, y, max_w, h=4, shade=None):
    """Draw a line of 'words' as gray bars."""
    cx = x
    while cx < x + max_w - 8:
        wlen = int(rng.integers(6, 22))
        if cx + wlen > x + max_w:
            break
        d.rectangle([cx, y, cx + wlen, y + h], fill=int(shade if shade is not None else rng.integers(60, 120)))
        cx += wlen + int(rng.integers(3, 6))


def claim_form(d, rng):
    m = int(rng.integers(8, 16))
    d.rectangle([m, m, W - m, m + 10], fill=40)                               # header bar
    y = m + 18
    cols, rows = int(rng.integers(3, 5)), int(rng.integers(6, 10))           # grid of labeled boxes
    bw, bh = (W - 2 * m) // cols, int(rng.integers(11, 15))
    for r in range(rows):
        for c in range(cols):
            x0 = m + c * bw
            d.rectangle([x0, y + r * bh, x0 + bw, y + (r + 1) * bh], outline=30, width=1)
            if rng.random() < 0.7:
                text_line(d, rng, x0 + 3, y + r * bh + 4, bw - 6, h=3)
    y += rows * bh + 8
    trows, tcols = int(rng.integers(4, 8)), int(rng.integers(5, 7))          # line-item table
    tw = (W - 2 * m) // tcols
    for r in range(trows + 1):
        d.line([m, y + r * 10, W - m, y + r * 10], fill=30, width=1)
    for c in range(tcols + 1):
        d.line([m + c * tw, y, m + c * tw, y + trows * 10], fill=30, width=1)


def referral(d, rng):
    m = int(rng.integers(14, 24))
    d.rectangle([m, m, m + 22, m + 22], fill=int(rng.integers(40, 90)))      # letterhead logo block
    text_line(d, rng, m + 30, m + 3, 110, h=5, shade=40)
    text_line(d, rng, m + 30, m + 13, 80, h=3)
    d.line([m, m + 30, W - m, m + 30], fill=60, width=2)
    y = m + 40
    for _ in range(2):                                                       # To: / From:
        text_line(d, rng, m, y, int(rng.integers(60, 100)), h=4); y += 10
    y += 6
    for _ in range(int(rng.integers(2, 4))):                                 # paragraphs
        for _ in range(int(rng.integers(3, 6))):
            text_line(d, rng, m, y, int(rng.integers(120, W - 2 * m)), h=4); y += 9
        y += 8
    if rng.random() < 0.25:                                                  # sometimes a small table
        for r in range(4):
            d.line([m, y + r * 8, m + 90, y + r * 8], fill=50)
        y += 30
    sx, sy = W - m - 70, min(y + 10, H - 30)                                 # signature squiggle
    pts = [(sx + i * 6, sy + int(6 * np.sin(i * rng.uniform(0.8, 1.6)))) for i in range(12)]
    d.line(pts, fill=20, width=2)
    d.line([sx, sy + 12, sx + 70, sy + 12], fill=60)


def enrollment_form(d, rng):
    m = int(rng.integers(8, 16))
    d.rectangle([m, m, W - m, m + 14], fill=int(rng.integers(30, 70)))      # title bar
    d.rectangle([W - m - 40, m + 22, W - m, m + 70], outline=30, width=2)   # ID photo box
    y = m + 24
    for s in range(int(rng.integers(3, 5))):
        text_line(d, rng, m, y, 70, h=5, shade=40); y += 12                 # section label
        for _ in range(int(rng.integers(2, 4))):
            d.rectangle([m, y, m + 7, y + 7], outline=20, width=1)          # checkbox
            if rng.random() < 0.4:
                d.line([m + 1, y + 4, m + 3, y + 7, m + 7, y], fill=10, width=1)
            text_line(d, rng, m + 12, y + 2, int(rng.integers(50, 110)), h=3); y += 12
        d.line([m, y + 2, W - m, y + 2], fill=40, width=2); y += 10          # section rule


def scan(img, rng, messy):
    """Simulate a scanner: tilt, shift, blur, noise, shading."""
    tilt = rng.uniform(-10, 10) if messy else rng.uniform(-3, 3)
    img = img.rotate(tilt, resample=Image.BILINEAR, fillcolor=255,
                     translate=tuple(int(v) for v in rng.integers(-12 if messy else -5, 13 if messy else 6, 2)))
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(1.0, 2.2) if messy else rng.uniform(0, 0.8)))
    a = np.asarray(img.resize((SIZE, SIZE), Image.BILINEAR), dtype=float)
    if messy:                                                                # uneven light and a shadow band
        grad = np.linspace(rng.uniform(-60, 0), rng.uniform(0, 60), SIZE)
        a = a + (grad[None, :] if rng.random() < 0.5 else grad[:, None])
        if rng.random() < 0.5:
            r0 = int(rng.integers(0, SIZE - 10)); a[r0:r0 + int(rng.integers(4, 10)), :] -= rng.uniform(40, 90)
    a = a * rng.uniform(0.85, 1.0) + rng.normal(0, 22 if messy else 8, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def main(n_per_class=400, n_messy=100):
    rng = np.random.default_rng(SEED)
    out = ROOT / "data" / "images"; out.mkdir(parents=True, exist_ok=True)
    draw = {"claim_form": claim_form, "referral": referral, "enrollment_form": enrollment_form}
    rows = []
    for cls in CLASSES:
        n = n_per_class + n_messy
        sets = ["train"] * int(n_per_class * 0.70) + ["val"] * int(n_per_class * 0.15)
        sets += ["test"] * (n_per_class - len(sets)) + ["messy"] * n_messy
        for i, s in enumerate(sets):
            img = Image.new("L", (W, H), 255)
            draw[cls](ImageDraw.Draw(img), rng)
            name = f"{cls}_{i:04d}.png"
            scan(img, rng, messy=(s == "messy")).save(out / name)
            rows.append({"filename": name, "label": cls, "set": s})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data" / "labels.csv", index=False)
    print(df.groupby(["set", "label"]).size().unstack())


if __name__ == "__main__":
    main()
