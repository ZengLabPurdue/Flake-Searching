# 20x Flake Extraction Pipeline

Extract individual flakes from microscope images: raw images → contour detection → cropped flakes with masks and color analysis.

**Everything lives in this folder:** Python scripts, settings, original images (`images/x20_images/`), and generated outputs.

---

## Folder Structure

```
20x_extraction/
├── images/
│   ├── x20_images/              # Original 20x microscope images (input)
│   └── filtered_sensitive_overlays/  # Step 1 output (created by pipeline)
├── robust_contours_and_masks_no_gap_close/  # Step 2 output (created by pipeline)
├── flake_extraction_outputs/    # Step 3 output (created by pipeline)
├── run_flake_extraction.py      # One-command runner
├── batch_filtered_sensitive_overlays_2x2.py
├── batch_robust_contours_and_masks.py
├── flake_extraction_pipeline.py
├── batch_filtered_tuner_ui.py
├── contour_tuner_ui.py
├── batch_filtered_settings.json
├── robust_contours_settings.json
├── requirements.txt
└── ... (other Python dependencies)
```

---

## Pipeline Overview

```
RAW IMAGES (images/x20_images/)
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: batch_filtered_sensitive_overlays_2x2.py              │
│  Settings: batch_filtered_settings.json (Batch Filtered Tuner)  │
└─────────────────────────────────────────────────────────────────┘
         │  2×2 binning → preprocessing → edge detection → gap closing
         │  → filter nested → min area filter
         ▼
    images/filtered_sensitive_overlays/
         • {stem}_binned_filtered_contours.png  (shape selection)
         • {stem}_binned_filtered_overlay.png
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: batch_robust_contours_and_masks.py                     │
│  Settings: robust_contours_settings.json (Contour Tuner UI)     │
└─────────────────────────────────────────────────────────────────┘
         │  Same pipeline → shape masks, background masks
         ▼
    robust_contours_and_masks_no_gap_close/{stem}/
         • *_masked_background_only.png, *_mask_shapes.png
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: flake_extraction_pipeline.py                          │
└─────────────────────────────────────────────────────────────────┘
         │  Load contours → HSV filter → bbox crop each shape
         │  → extract_background on crop → K-means color → save
         ▼
    flake_extraction_outputs/{stem}/
         • contour_overlay.png
         • crops/flake_001/crop.png, masked.png, mask.png, color.png, ...
```

---

## How to Run

Run from inside the `20x_extraction` folder (or from repo root with the path):

```bash
cd 20x_extraction
python run_flake_extraction.py
```

Or from repo root:

```bash
python 20x_extraction/run_flake_extraction.py
```

- **Input:** `images/x20_images/` (inside 20x_extraction)
- **Output:** `flake_extraction_outputs/` (inside 20x_extraction)

### Custom input/output

```bash
python run_flake_extraction.py path/to/images -o path/to/output
```

### Run steps individually

From inside `20x_extraction/`:

```bash
# Step 1: Contour overlays (for shape selection)
python batch_filtered_sensitive_overlays_2x2.py images/x20_images -o images/filtered_sensitive_overlays

# Step 2: Robust masks (for background in crops)
python batch_robust_contours_and_masks.py images/x20_images -o robust_contours_and_masks_no_gap_close

# Step 3: Extract flakes
python flake_extraction_pipeline.py images/x20_images -o flake_extraction_outputs
```

---

## How to Change Settings

Settings are stored in JSON files and can be edited via two UIs.

### 1. Batch Filtered 2x2 Overlay Tuner (contour/shape selection)

**Settings file:** `batch_filtered_settings.json`  
**UI:** `python batch_filtered_tuner_ui.py`

Controls which shapes are detected for flake extraction:

- Edge detection method (Canny+H, Canny only, Sobel, Laplacian, etc.)
- Blur sigma, Canny low/high, min area
- Preprocessing (Full / CLAHE only / None)
- Gap closing (bridge factor, force-close factor, line thickness)
- Border morph (margin, corner radius)
- **Min area (contour)** – drop small shapes (default 200 px²)

**Save:** Click **Update Now** in the UI, or edit `batch_filtered_settings.json` directly.

---

### 2. Contour Tuner (robust masks for background)

**Settings file:** `robust_contours_settings.json`  
**UI:** `python contour_tuner_ui.py`

Controls how robust masks are built for background extraction in each crop.

- Same controls as above (edge method, preprocessing, gap closing, etc.)

**Save:** Click **Update Now** in the UI, or edit `robust_contours_settings.json` directly.

---

## Output Structure

```
flake_extraction_outputs/
├── {image_stem}/
│   ├── contour_overlay.png      # Green contours on original
│   ├── summary.txt
│   ├── flake_colors.csv
│   └── crops/
│       ├── flake_001/
│       │   ├── crop.png         # Bbox crop
│       │   ├── overlay.png      # Contour on crop
│       │   ├── masked.png       # Flake pixels only
│       │   ├── mask.png         # Binary mask
│       │   ├── color.png        # Dominant color swatch
│       │   └── background_crop.png  # Background region (when robust masks exist)
│       ├── flake_002/
│       └── ...
```

---

## Dependencies

- Python 3.8+
- OpenCV (`cv2`), NumPy, PIL/Pillow
- Tkinter (for tuning UIs; usually bundled with Python)

---

## Files Reference

| File | Purpose |
|------|---------|
| `run_flake_extraction.py` | One-command pipeline runner |
| `batch_filtered_sensitive_overlays_2x2.py` | Step 1 – contour overlays |
| `batch_robust_contours_and_masks.py` | Step 2 – robust masks |
| `flake_extraction_pipeline.py` | Step 3 – crop & extract flakes |
| `batch_filtered_tuner_ui.py` | Tune Step 1 settings |
| `contour_tuner_ui.py` | Tune Step 2 settings |
| `batch_filtered_settings.json` | Step 1 parameters (auto-saved by UI) |
| `robust_contours_settings.json` | Step 2 parameters (auto-saved by UI) |
