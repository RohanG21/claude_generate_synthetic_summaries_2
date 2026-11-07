# Scan-styled handwriting generator

Takes a real scanned document, records its **style** and its **structure** in two separate files,
then writes any text you give it (`.txt` files) in that style and layout.

## Folder layout

```
scan_synth/
  synth.py            the program
  config.yaml         shared settings (fonts, jitter, warping, variants, folders)
  scans/              real scans to copy (.jpg / .png / .tif)
  styles/             <scan>.style.yaml       written by `measure`
  structures/         <scan>.structure.yaml   written by `measure`
  texts/              .txt files to write, one document per file
  fonts/              handwriting fonts (OFL / Apache, from Google Fonts)
  out/                generated images and labels
```

All folders are relative to `synth.py`, so you can run it from anywhere. Folder names can be changed in `config.yaml`.

## Install

```bash
pip install pillow numpy opencv-python-headless pyyaml
```

## 1. Measure a scan

```bash
python synth.py measure              # every scan in scans/
python synth.py measure note1        # just one
python synth.py measure --remeasure  # overwrite files you've edited
```

For `scans/note1.jpg` this writes two files:

**`styles/note1.style.yaml`** describes how the writing looks:
letter size (x-height), slant, stroke width, word gaps, gray levels (paper, ink, noise, gray or
black-and-white), and page skew.

**`structures/note1.structure.yaml`** describes where the writing sits:
page width and height, margins (top distance to the first line, bottom distance after the last line,
left start and its drift, right gap), line spacing, line length, and line tilt.

Each value is stored as `{mean, sd}`. The sd is how much generated pages vary around the mean.
Edit any value that looks wrong; `measure` keeps your edits unless you pass `--remeasure`.

## 2. Add texts

Put `.txt` files in `texts/`. Each file becomes one handwritten document.
- A blank line starts a new paragraph on a new line.
- Single line breaks are joined, unless `keep_line_breaks: true` in `config.yaml`.
- Text longer than one page continues onto extra pages (`_p1`, `_p2`, …), up to `max_pages`.

## 3. Generate

```bash
python synth.py generate                                  # every text x every style x `variants`
python synth.py generate --style note1 --variants 10
python synth.py generate --style note1 --structure card2  # note1's handwriting in card2's layout
python synth.py generate --only-texts bronchitis_note --rec-crops
python synth.py generate --time-budget 600                # stop after 10 minutes
```

Each variant samples new values around the measured ones (font, letter size, slant, spacing,
margins, tilt, skew, noise), so variants of the same text look like different sittings by the same writer.

## Output (in `out/`)

- `images/<text>__<style>__v<n>[_p<page>].png`: same pixel size as the source scan
- `labels/<same name>.json`: exact text, one polygon per line, paragraph numbers, which text, style and
  structure files were used, and every sampled value
- `det_gt.txt`: PaddleOCR detection labels
- `rec/` + `rec_gt.txt`: line crops and PaddleOCR recognition labels (with `--rec-crops`)

## Tips

- `variation` in `config.yaml` scales all spreads (0 = copy each scan's averages exactly).
- Slant is measured automatically and can be off for very messy writing; adjust `slant_deg` in the style file.
- If a scan has dark borders or shadows, raise `--border` when measuring (default ignores 25 px at the edges).
- The fonts are neater than most real handwriting. Turn fonts on or off, and adjust their weights, in `config.yaml`.
