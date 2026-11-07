#!/usr/bin/env python3
"""
Write any text in the handwriting style and page layout of a real scan.

Folder layout (all relative to this file):
  scans/        real scans to copy (.jpg/.png/.tif)
  styles/       <scan>.style.yaml      letter size, slant, stroke width, word gaps, gray levels, skew
  structures/   <scan>.structure.yaml  page size, margins, line spacing, line lengths, line tilt
  texts/        .txt files to write, one document per file
  out/          generated images and labels

  python synth.py measure                      # every scan -> one style file + one structure file
  python synth.py generate                     # every texts/*.txt x every style, `variants` images each
  python synth.py generate --style note1 --structure intake_card   # mix one scan's style with another's layout
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# =====================================================================
# Helpers
# =====================================================================
def sample(spec, rng, scale=1.0):
    """12 -> 12;  {mean, sd} -> clipped normal (sd * scale);  [lo, hi] -> uniform."""
    if spec is None:
        return None
    if isinstance(spec, (int, float)):
        return float(spec)
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        return rng.uniform(float(spec[0]), float(spec[1]))
    if isinstance(spec, dict) and "mean" in spec:
        m, sd = float(spec["mean"]), float(spec.get("sd", 0)) * scale
        v = rng.gauss(m, sd) if sd > 0 else m
        return min(max(v, m - 3 * sd), m + 3 * sd)
    raise ValueError(f"Cannot sample from {spec!r}")


def ms(values, min_sd=0.0, nd=1):
    v = np.asarray(values, float)
    if v.size == 0:
        return None
    return {"mean": round(float(np.mean(v)), nd), "sd": round(max(float(np.std(v)), min_sd), nd)}


def load_yaml(p):
    with open(p) as f:
        return yaml.safe_load(f) or {}


def load_config(path):
    path = Path(path).resolve()
    return load_yaml(path), path.parent


# =====================================================================
# 1. Measure a real scan
# =====================================================================
def measure_scan(path, border=25):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise ValueError(f"Cannot read {path}")
    H, W = g.shape
    b = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 18)
    b[:border], b[-border:], b[:, :border], b[:, -border:] = 0, 0, 0, 0
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    # drop huge blobs (photo edges, shadows, table borders)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b)
    for i in range(1, n):
        if st[i, 3] > H * 0.25 or st[i, 2] > W * 0.6:
            b[lab == i] = 0

    # page skew: rotation that makes the row profile sharpest
    best = None
    for a in np.arange(-8, 8.01, 0.25):
        M = cv2.getRotationMatrix2D((W / 2, H / 2), a, 1)
        r = cv2.warpAffine(b, M, (W, H))
        v = np.var(np.diff(r.sum(1).astype(float)))
        if best is None or v > best[0]:
            best = (v, a, r)
    _, skew, r = best

    # line spacing from the autocorrelation of the row profile
    p = r.sum(1).astype(float) / 255
    pc = p - p.mean()
    ac = np.correlate(pc, pc, "full")[len(pc) - 1:]
    lags = np.arange(len(ac))
    valid = (lags > 15) & (lags < H // 2)
    cand = [l for l in lags[valid] if ac[l] >= ac[l - 1] and ac[l] >= ac[min(l + 1, len(ac) - 1)] and ac[l] > 0]
    sp0 = int(cand[0]) if cand else max(20, H // 12)
    win = max(5, sp0 // 3)
    sm = np.convolve(p, np.ones(win) / win, "same")
    half = max(3, int(sp0 * 0.45))
    peaks = [i for i in range(half, H - 3) if sm[i] == sm[max(0, i - half):i + half + 1].max() and sm[i] > sm.max() * 0.2]

    lines = []
    for c in peaks:
        y0, y1 = max(0, c - half), min(H, c + half)
        band = r[y0:y1]
        col = band.sum(0).astype(float)
        if col.sum() == 0:
            continue
        cum = np.cumsum(col) / col.sum()
        x0, x1 = int(np.searchsorted(cum, 0.01)), int(np.searchsorted(cum, 0.99))
        rows = np.convolve(band.sum(1).astype(float), np.ones(3) / 3, "same")
        core = np.nonzero(rows > rows.max() * 0.5)[0]
        occ = band.sum(0) > 0
        lines.append({"y": c, "x0": x0, "x1": x1, "core": int(core.max() - core.min() + 1),
                      "base": y0 + int(core.max()), "occ": occ})
    if not lines:
        raise ValueError(f"No text lines found in {path}")

    xh = float(np.median([l["core"] for l in lines]))
    gaps = []
    for l in lines:
        run = 0
        for v in l["occ"][l["x0"]:l["x1"]]:
            if not v:
                run += 1
            else:
                if 0.25 * xh <= run <= 4 * xh:
                    gaps.append(run)
                run = 0

    # per-line tilt (on the deskewed image)
    tilts = []
    for l in lines:
        band = r[max(0, l["y"] - half):l["y"] + half]
        pts = []
        for x in range(l["x0"], l["x1"] - 40, 40):
            ys = np.nonzero(band[:, x:x + 40])[0]
            if len(ys) > 30:
                pts.append((x + 20, ys.mean()))
        if len(pts) >= 4:
            xs, ys = np.array(pts).T
            tilts.append(-math.degrees(math.atan(np.polyfit(xs, ys, 1)[0])))

    # slant: shear that makes strokes most vertical (+ = leans right)
    best = None
    for a in range(-35, 36):
        t = math.tan(math.radians(a))
        s = cv2.warpAffine(r, np.float32([[1, t, 0], [0, 1, 0]]), (W + H, H))
        v = float((s.sum(0).astype(float) ** 2).sum())
        if best is None or v > best[0]:
            best = (v, a)
    slant = best[1]

    # stroke width: 2 x distance-to-edge at the stroke centre lines, on a global (Otsu) mask
    _, ob = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ob = cv2.bitwise_and(ob, cv2.dilate(b, np.ones((3, 3), np.uint8)))
    dt = cv2.distanceTransform(ob, cv2.DIST_L2, 5)
    ridge = (dt > 0) & (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8)))
    stroke = float(2 * np.median(dt[ridge]) - 1) if ridge.any() else 3.0

    # capture look
    inner = g[border:-border, border:-border]
    ink_mask = cv2.dilate(b, np.ones((7, 7), np.uint8))[border:-border, border:-border] > 0
    paper = inner[~ink_mask]
    mid = ((g > 60) & (g < 190)).mean()
    noise = np.abs(inner.astype(float) - cv2.medianBlur(inner, 5).astype(float))[~ink_mask]

    bases = [l["base"] for l in lines]
    spacing = np.diff(bases) if len(lines) > 1 else [sp0]
    starts = [l["x0"] for l in lines]
    ends = [l["x1"] for l in lines[:-1]] or [lines[0]["x1"]]

    structure = {
        "source": Path(path).name,
        "page": {"width_px": W, "height_px": H},
        "lines_on_scan": len(lines),
        "margins": {
            "top_px": ms([bases[0]], min_sd=xh * 0.3),            # page top -> first baseline
            "bottom_px": ms([H - bases[-1]], min_sd=xh * 0.3),    # last baseline -> page bottom
            "left_px": ms(starts, min_sd=5),                      # where lines start
            "left_drift_px_per_line": round(float(np.polyfit(range(len(lines)), starts, 1)[0]), 2)
                                      if len(lines) > 2 else 0.0,
            "right_px": ms([W - e for e in ends], min_sd=10),     # gap after full lines
        },
        "line_spacing_px": ms(spacing, min_sd=2),                 # baseline to baseline
        "line_length_px": ms([e - s for s, e in zip(starts, ends)], min_sd=10),
        "line_tilt_deg": ms(tilts or [0], min_sd=0.3, nd=2),
    }
    style = {
        "source": Path(path).name,
        "letter_size": {"x_height_px": ms([l["core"] for l in lines], min_sd=1)},
        "slant_deg": {"mean": float(slant), "sd": 2.0},
        "stroke_width_px": {"mean": round(stroke, 2), "sd": round(stroke * 0.1, 2)},
        "word_gap_px": ms(gaps or [xh], min_sd=2),
        "gray_levels": {
            "mode": "binary" if mid < 0.01 else "gray",
            "paper": int(np.median(paper)),
            "ink": int(np.percentile(g[b > 0], 25)),
            "paper_noise": round(float(np.std(noise)), 1),
        },
        "skew_deg": {"mean": round(-float(skew), 2), "sd": 0.5},
    }
    return style, structure


def _clean(d):
    return json.loads(json.dumps(d, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def _write(path, data, header):
    with open(path, "w") as f:
        f.write(header)
        yaml.safe_dump(_clean(data), f, sort_keys=False, width=120)


def cmd_measure(a):
    cfg, root = load_config(a.config)
    sdir = root / cfg["scans_dir"]
    stdir, ctdir = root / cfg["styles_dir"], root / cfg["structures_dir"]
    stdir.mkdir(exist_ok=True)
    ctdir.mkdir(exist_ok=True)
    scans = sorted(p for p in sdir.iterdir() if p.suffix.lower() in IMG_EXT) if sdir.exists() else []
    if a.scans:
        scans = [p for p in scans if p.stem in a.scans or p.name in a.scans]
    if not scans:
        sys.exit(f"No scans found in {sdir}/")
    for s in scans:
        sf, cf = stdir / f"{s.stem}.style.yaml", ctdir / f"{s.stem}.structure.yaml"
        if sf.exists() and cf.exists() and not a.remeasure:
            print(f"{s.name}: style and structure files exist, kept (use --remeasure to overwrite)")
            continue
        style, structure = measure_scan(s, border=a.border)
        _write(sf, style, f"# STYLE measured from {s.name}: how the writing looks.\n"
                          f"# Edit any value. Values are {{mean, sd}}; sd = how much samples vary.\n")
        _write(cf, structure, f"# STRUCTURE measured from {s.name}: where the writing sits on the page.\n"
                              f"# Edit any value. Values are {{mean, sd}}; sd = how much samples vary.\n")
        print(f"{s.name}: {structure['lines_on_scan']} lines, spacing {structure['line_spacing_px']['mean']}px, "
              f"x-height {style['letter_size']['x_height_px']['mean']}px, slant {style['slant_deg']['mean']}°, "
              f"stroke {style['stroke_width_px']['mean']}px\n   -> {sf.relative_to(root)}, {cf.relative_to(root)}")


# =====================================================================
# 2. Text: texts/*.txt -> paragraphs of words
# =====================================================================
def read_text(path, keep_line_breaks=False):
    raw = Path(path).read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    paras = []
    for block in re.split(r"\n\s*\n", raw):
        parts = block.split("\n") if keep_line_breaks else [block]
        for p in parts:
            words = p.split()
            if words:
                paras.append(words)
    return paras


# =====================================================================
# 3. Render
# =====================================================================
class Renderer:
    def __init__(self, cfg):
        self.cfg = cfg
        fdir = Path(cfg.get("fonts_dir", "fonts"))
        self.fonts = []
        for name, w in cfg["handwriting_fonts"].items():
            if not w:
                continue
            p = fdir / name
            if not p.exists():
                raise FileNotFoundError(p)
            f = ImageFont.truetype(str(p), 200)
            hs = []
            for ch in "acemnorsuvwxz":
                l, t, r, b = f.getbbox(ch, anchor="ls")
                if b > t:
                    hs.append(b - t)
            self.fonts.append((str(p), float(w), max(float(np.median(hs)), 1.0) / 200.0))
        self._f, self._g = {}, {}

    def glyph(self, path, size, ch):
        k = (path, size, ch)
        if k not in self._g:
            if (path, size) not in self._f:
                self._f[(path, size)] = ImageFont.truetype(path, size)
            f = self._f[(path, size)]
            adv = f.getlength(ch)
            l, t, r, b = f.getbbox(ch, anchor="ls")
            if r - l <= 0 or b - t <= 0:
                self._g[k] = (None, 0, 0, adv)
            else:
                im = Image.new("L", (r - l + 2, b - t + 2), 0)
                ImageDraw.Draw(im).text((1 - l, 1 - t), ch, font=f, fill=255, anchor="ls")
                self._g[k] = (im, l - 1, t - 1, adv)
        return self._g[k]

    def page_params(self, style, struct, rng):
        v = float(self.cfg.get("variation", 1.0))
        font = rng.choices(self.fonts, weights=[w for _, w, _ in self.fonts])[0]
        xh = max(6.0, sample(style["letter_size"]["x_height_px"], rng, v))
        M = struct["margins"]
        return {
            "font": font, "x_height": xh, "size": max(8, int(round(xh / font[2]))),
            "spacing": max(xh * 1.5, sample(struct["line_spacing_px"], rng, v * 0.3)),
            "top": sample(M["top_px"], rng, v),
            "bottom": max(xh * 0.6, sample(M["bottom_px"], rng, v)),
            "left": sample(M["left_px"], rng, v * 0.5),
            "drift": float(M.get("left_drift_px_per_line", 0.0)),
            "slant": sample(style["slant_deg"], rng, v),
            "stroke": max(1.0, sample(style["stroke_width_px"], rng, v)),
            "W": int(struct["page"]["width_px"]), "H": int(struct["page"]["height_px"]),
        }

    def plan_word(self, w, P, rng):
        cj, fpath = self.cfg["char_jitter"], P["font"][0]
        plans, wx = [], 0.0
        for ch in w:
            size = max(6, int(round(P["size"] * (1 + sample(cj["size_rel"], rng)))))
            rot, dy, sp = sample(cj["rotation_deg"], rng), sample(cj["baseline_px"], rng), sample(cj["spacing_px"], rng)
            plans.append((ch, fpath, size, rot, dy, wx))
            wx += self.glyph(fpath, size, ch)[3] + sp
        return plans, wx

    def layout(self, paras, style, struct, P, rng):
        """Wrap paragraphs into lines using the structure; returns a list of pages (lists of lines)."""
        v = float(self.cfg.get("variation", 1.0))
        M = struct["margins"]
        gap_lines = float(self.cfg.get("paragraph_gap_lines", 0))
        last_base = P["H"] - P["bottom"]
        pages, page, row = [], [], 0.0
        for pi, words in enumerate(paras):
            wi = 0
            while wi < len(words):
                base = P["top"] + row * P["spacing"] + rng.gauss(0, 0.15 * struct["line_spacing_px"]["sd"] * v)
                if page and base > last_base:
                    pages.append(page)
                    if len(pages) >= int(self.cfg.get("max_pages", 10)):
                        return pages, True
                    page, row = [], 0.0
                    base = P["top"] + rng.gauss(0, 0.15 * struct["line_spacing_px"]["sd"] * v)
                x0 = P["left"] + P["drift"] * len(page) + rng.gauss(0, 0.4 * M["left_px"]["sd"] * v)
                right = P["W"] - max(4.0, sample(M["right_px"], rng, v))
                x, cur = x0, []
                while wi < len(words):
                    plans, wx = self.plan_word(words[wi], P, rng)
                    if cur and x + wx > right:
                        break
                    cur.append((words[wi], x, plans))
                    wi += 1
                    x += wx + max(4.0, sample(style["word_gap_px"], rng, v))
                page.append({"words": cur, "text": " ".join(w for w, _, _ in cur), "baseline": base,
                             "tilt": sample(struct["line_tilt_deg"], rng, v), "x0": x0, "paragraph": pi})
                row += 1
            row += gap_lines
        if page:
            pages.append(page)
        return pages, False

    def draw_line(self, line, P, rng, ink):
        cj, W, H = self.cfg["char_jitter"], P["W"], P["H"]
        sh = int(P["spacing"] * 2.6)
        by = int(P["spacing"] * 1.5)
        strip = np.zeros((sh, W), np.float32)
        amp = sample(cj["wobble"]["amplitude_px"], rng)
        lam = sample(cj["wobble"]["wavelength_px"], rng)
        ph = rng.uniform(0, 2 * math.pi)
        for _, wx0, plans in line["words"]:
            for ch, fpath, size, rot, dy, off in plans:
                im, l, t, _ = self.glyph(fpath, size, ch)
                if im is None:
                    continue
                gx = wx0 + off
                yb = by + dy + amp * math.sin(2 * math.pi * gx / lam + ph)
                cx, cy = gx + l + im.width / 2, yb + t + im.height / 2
                gimg = im.rotate(rot, resample=Image.BILINEAR, expand=True) if abs(rot) > 0.2 else im
                a = np.asarray(gimg, np.float32) / 255.0
                x1, y1 = int(round(cx - a.shape[1] / 2)), int(round(cy - a.shape[0] / 2))
                xa, ya, xb, yb2 = max(0, x1), max(0, y1), min(W, x1 + a.shape[1]), min(sh, y1 + a.shape[0])
                if xb > xa and yb2 > ya:
                    np.maximum(strip[ya:yb2, xa:xb], a[ya - y1:yb2 - y1, xa - x1:xb - x1], out=strip[ya:yb2, xa:xb])
        # slant: + leans right (tops move right), sheared about the baseline
        t = math.tan(math.radians(P["slant"]))
        shear = np.array([[1, -t, t * by], [0, 1, 0], [0, 0, 1]])
        rot = np.vstack([cv2.getRotationMatrix2D((line["x0"], by), line["tilt"], 1.0), [0, 0, 1]])
        warped = cv2.warpAffine(strip, (rot @ shear)[:2], (W, sh), flags=cv2.INTER_LINEAR, borderValue=0)
        top = int(round(line["baseline"])) - by
        ya, yb = max(0, top), min(H, top + sh)
        if yb > ya:
            np.maximum(ink[ya:yb], warped[ya - top:yb - top], out=ink[ya:yb])
        ys, xs = np.nonzero(warped > 0.3)
        if len(xs) == 0:
            return None
        return [float(xs.min()), float(ys.min() + top), float(xs.max()), float(ys.max() + top)]

    def elastic(self, ink, rng, nrng):
        e = self.cfg.get("elastic")
        if not e:
            return ink
        alpha, sigma = sample(e["alpha_px"], rng), sample(e["sigma_px"], rng)
        if not alpha:
            return ink
        h, w = ink.shape
        k = max(1, int(round(sigma / 4)))
        fields = []
        for _ in range(2):
            d = cv2.GaussianBlur(nrng.uniform(-1, 1, (h // k + 1, w // k + 1)).astype(np.float32), (0, 0), sigma / k)
            d *= alpha / (np.abs(d).max() + 1e-6)
            fields.append(cv2.resize(d, (w, h)))
        xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        return cv2.remap(ink, xx + fields[0], yy + fields[1], cv2.INTER_LINEAR, borderValue=0)

    @staticmethod
    def match_stroke(ink, target):
        b = (ink > 0.5).astype(np.uint8)
        if not b.any():
            return ink
        cnts, _ = cv2.findContours(b, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cur = 2.0 * b.sum() / max(sum(cv2.arcLength(c, True) for c in cnts), 1.0)
        k = int(round(target - cur))
        el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        if k > 0:
            if k >= 2:
                ink = cv2.dilate(ink, el, iterations=k // 2)
            if k % 2:
                ink = cv2.dilate(ink, np.ones((2, 2), np.uint8))
        elif k < 0:
            k = min(-k, int(cur - 1.5))
            if k >= 2:
                ink = cv2.erode(ink, el, iterations=k // 2)
            if k % 2:
                ink = cv2.erode(ink, np.ones((2, 2), np.uint8))
        return ink

    def capture(self, ink, style, P, rng, nrng, boxes):
        sc, C, W, H = self.cfg["scan"], style["gray_levels"], P["W"], P["H"]
        sig = sample(sc["blur_sigma"], rng)
        if sig:
            ink = cv2.GaussianBlur(ink, (0, 0), sig)
        paper, inkg = C["paper"] / 255.0, C["ink"] / 255.0
        img = paper - np.clip(ink, 0, 1) * (paper - inkg)
        img = img + nrng.normal(0, C.get("paper_noise", 3) / 255.0, img.shape).astype(np.float32)
        skew = sample(style["skew_deg"], rng, float(self.cfg.get("variation", 1.0)))
        M = cv2.getRotationMatrix2D((W / 2, H / 2), skew, 1.0)
        img = cv2.warpAffine(np.clip(img, 0, 1), M, (W, H), flags=cv2.INTER_LINEAR, borderValue=paper)
        polys = []
        for (x0, y0, x1, y1) in boxes:
            pts = np.array([[x0 - 3, y0 - 3], [x1 + 3, y0 - 3], [x1 + 3, y1 + 3], [x0 - 3, y1 + 3]], float)
            pts = np.hstack([pts, np.ones((4, 1))]) @ M.T
            polys.append([[int(np.clip(px, 0, W - 1)), int(np.clip(py, 0, H - 1))] for px, py in pts])
        out = (img * 255).astype(np.uint8)
        if C.get("mode") == "binary":
            _, out = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        rate = sample(sc.get("stroke_break_rate", 0), rng)
        if rate:
            ys, xs = np.nonzero(out < (out.mean() - 40))
            n = int(len(xs) * rate)
            for i in (nrng.choice(len(xs), n, replace=False) if n else []):
                cv2.circle(out, (int(xs[i]), int(ys[i])), int(nrng.integers(1, 3)), int(C["paper"]), -1)
        rate = sample(sc.get("speckle_rate", 0), rng)
        if rate:
            n = int(W * H * rate)
            out[nrng.integers(0, H, n), nrng.integers(0, W, n)] = int(C["ink"])
        q = sc.get("jpeg_quality")
        if q and C.get("mode") != "binary":
            out = cv2.imdecode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(q)])[1], cv2.IMREAD_GRAYSCALE)
        return out, polys, skew


# =====================================================================
# 4. Generate
# =====================================================================
WK = {}


def init_worker(cfg, root, out, rec):
    os.chdir(root)
    WK.update(cfg=cfg, out=Path(out), rec=rec, R=Renderer(cfg))


def make_sample(task):
    sid, text_path, style_path, struct_path, seed = task
    cfg, R, out = WK["cfg"], WK["R"], WK["out"]
    style, struct = load_yaml(style_path), load_yaml(struct_path)
    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    paras = read_text(text_path, cfg.get("keep_line_breaks", False))
    if not paras:
        return {"id": sid, "error": "empty text file"}
    P = R.page_params(style, struct, rng)
    pages, truncated = R.layout(paras, style, struct, P, rng)
    det, rec, files = [], [], []
    for pn, lines in enumerate(pages, 1):
        pid = sid if len(pages) == 1 else f"{sid}_p{pn}"
        ink = np.zeros((P["H"], P["W"]), np.float32)
        kept, boxes = [], []
        for ln in lines:
            bx = R.draw_line(ln, P, rng, ink)
            if bx:
                kept.append(ln)
                boxes.append(bx)
        ink = R.match_stroke(R.elastic(ink, rng, nrng), P["stroke"])
        img, polys, skew = R.capture(ink, style, P, rng, nrng, boxes)
        rel = f"images/{pid}.png"
        Image.fromarray(img).save(out / rel)
        lab_lines = [{"text": l["text"], "paragraph": l["paragraph"], "points": p} for l, p in zip(kept, polys)]
        if WK["rec"]:
            for i, p in enumerate(polys):
                xs, ys = [q[0] for q in p], [q[1] for q in p]
                crel = f"rec/{pid}_{i:02d}.png"
                cv2.imwrite(str(out / crel), img[min(ys):max(ys) + 1, min(xs):max(xs) + 1])
                rec.append((crel, kept[i]["text"]))
        label = {
            "id": pid, "image": rel, "text_file": Path(text_path).name, "page": pn, "pages": len(pages),
            "truncated": truncated, "style_file": Path(style_path).name, "structure_file": Path(struct_path).name,
            "seed": seed, "text": "\n".join(l["text"] for l in kept), "lines": lab_lines,
            "sampled": {"font": Path(P["font"][0]).name, "x_height_px": round(P["x_height"], 1),
                        "line_spacing_px": round(P["spacing"], 1), "slant_deg": round(P["slant"], 1),
                        "stroke_px": round(P["stroke"], 2), "skew_deg": round(skew, 2),
                        "left_margin_px": round(P["left"], 1), "top_margin_px": round(P["top"], 1)},
        }
        with open(out / f"labels/{pid}.json", "w") as f:
            json.dump(label, f, indent=1, ensure_ascii=False)
        det.append(f"{rel}\t" + json.dumps([{"transcription": l["text"], "points": l["points"]} for l in lab_lines],
                                            ensure_ascii=False))
        files.append(rel)
    return {"id": sid, "det": det, "rec": rec, "files": files, "truncated": truncated}


def pick(dirpath, names, suffix):
    files = sorted(dirpath.glob(f"*{suffix}"))
    if names:
        want = {n.replace(suffix, "") for n in names}
        files = [f for f in files if f.name[:-len(suffix)] in want]
    return files


def cmd_generate(a):
    cfg, root = load_config(a.config)
    stdir, ctdir, tdir = root / cfg["styles_dir"], root / cfg["structures_dir"], root / (a.texts or cfg["texts_dir"])
    styles = pick(stdir, a.style, ".style.yaml")
    if not styles:
        sys.exit(f"No style files in {stdir}/. Run: python synth.py measure")
    texts = sorted(tdir.glob("*.txt"))
    if a.only_texts:
        texts = [t for t in texts if t.stem in a.only_texts or t.name in a.only_texts]
    if not texts:
        sys.exit(f"No .txt files in {tdir}/")
    structs = {p.name[:-len(".structure.yaml")]: p for p in ctdir.glob("*.structure.yaml")}
    fixed_struct = None
    if a.structure:
        name = a.structure.replace(".structure.yaml", "")
        if name not in structs:
            sys.exit(f"No structure file {name}.structure.yaml in {ctdir}/")
        fixed_struct = structs[name]
    variants = a.variants or int(cfg.get("variants", 1))
    seed = int(cfg.get("seed", 0))

    tasks = []
    for t in texts:
        for sf in styles:
            sname = sf.name[:-len(".style.yaml")]
            cf = fixed_struct or structs.get(sname)
            if cf is None:
                sys.exit(f"No structure file for style {sname}; pass --structure NAME")
            cname = cf.name[:-len(".structure.yaml")]
            tag = sname if cname == sname else f"{sname}+{cname}"
            for k in range(variants):
                tasks.append((f"{t.stem}__{tag}__v{k + 1}", str(t), str(sf), str(cf), seed * 1_000_003 + len(tasks)))

    out = (root / (a.out or cfg.get("out_dir", "out"))).resolve()
    for d in ("images", "labels") + (("rec",) if a.rec_crops else ()):
        (out / d).mkdir(parents=True, exist_ok=True)
    print(f"{len(texts)} text file(s) x {len(styles)} style(s) x {variants} variant(s) = {len(tasks)} documents, "
          f"{a.workers} workers", flush=True)
    t0, det, rec, pages, warn = time.time(), [], [], 0, []
    with Pool(a.workers, initializer=init_worker, initargs=(cfg, str(root), str(out), a.rec_crops)) as pool:
        for r in pool.imap_unordered(make_sample, tasks):
            if "error" in r:
                warn.append(f"{r['id']}: {r['error']}")
                continue
            det += r["det"]
            rec += r["rec"]
            pages += len(r["files"])
            if r["truncated"]:
                warn.append(f"{r['id']}: text longer than max_pages, rest dropped")
            if a.time_budget and time.time() - t0 > a.time_budget:
                pool.terminate()
                warn.append("time budget reached, stopped early")
                break
    (out / "det_gt.txt").write_text("\n".join(sorted(det)) + "\n", encoding="utf-8")
    if a.rec_crops:
        (out / "rec_gt.txt").write_text("\n".join(f"{p}\t{t}" for p, t in sorted(rec)) + "\n", encoding="utf-8")
    el = time.time() - t0
    print(f"Done: {pages} page image(s) in {el:.1f}s -> {out}")
    for w in warn[:20]:
        print("  warning:", w)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.yaml"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure", help="scan -> styles/<scan>.style.yaml + structures/<scan>.structure.yaml")
    m.add_argument("scans", nargs="*", help="only these scans (default: all in scans/)")
    m.add_argument("--remeasure", action="store_true", help="overwrite existing files")
    m.add_argument("--border", type=int, default=25, help="pixels ignored at the image edges")
    g = sub.add_parser("generate", help="write texts/*.txt in the measured style and structure")
    g.add_argument("--style", nargs="*", help="style names to use (default: all)")
    g.add_argument("--structure", help="use this structure for every style (default: each style's own scan)")
    g.add_argument("--texts", help="folder of .txt files (default: texts_dir in config)")
    g.add_argument("--only-texts", nargs="*", help="only these .txt files")
    g.add_argument("--variants", type=int, help="images per text and style (default: config)")
    g.add_argument("--out", help="output folder (default: out_dir in config)")
    g.add_argument("--workers", type=int, default=os.cpu_count())
    g.add_argument("--time-budget", type=float, help="seconds; stop and keep finished pages")
    g.add_argument("--rec-crops", action="store_true", help="write line crops + rec_gt.txt for PaddleOCR")
    a = ap.parse_args()
    {"measure": cmd_measure, "generate": cmd_generate}[a.cmd](a)


if __name__ == "__main__":
    main()
