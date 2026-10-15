# Standalone tools

Run these commands from the repository root in PowerShell.

## Region annotation

Run the single Python file:

```powershell
.\.venv\Scripts\python.exe .\tools\annotation_tool.py
```

Label detected regions as good, bad, or unsure; mark missed good flakes in full
images; save/resume projects; and export crop datasets and reference annotations.
The application is a single Python file, with a left control panel and scanner-style
File, View, and Review menus.

Use **File > Add scan**, **Add image folder**, or **Add source images** to extend
an open project. New images use its frozen profile and annotation settings;
existing labels are retained and identical files are skipped. Use a separate
project when a material/substrate combination requires another detection profile.

The main view preserves the full raw image and colors the selected region's actual
pixels, with only its number and no bounding rectangle. **Display all** shows the
other visible regions and their numbers as well. Colors are white for unlabeled,
pure green `(0, 255, 0)` for good, pure red `(255, 0, 0)` for bad, and gray
`(128, 128, 128)` for unsure. **View > Show excluded candidates** controls which
candidates are available. Selection does not enlarge regions or their markers.
Numbers keep a four-pixel gap with simple local spacing; they are never hidden
by collision handling. Use the mouse wheel over either image to zoom and
middle-button drag to pan. The full raw image also supports left-button drag to
pan; click to select a region, or enable drawing mode to mark a missed flake.
The left panel has no outer scrollbar; use the wheel over its controls to scroll.

The **Show mask** checkbox below the cropped preview switches between the untouched
raw crop and its aligned white-on-black binary mask. This affects only the preview.
Exports contain both: each 512×1024 sample stores the 512×512 RGB crop on the left
and its aligned binary mask on the right. Display colors never enter training crops.
Source images retain their original pixels.

Defaults: P:A and other profile geometry filters are off; candidates below
200 µm² are hidden but retained. Images without calibration metadata need an
explicit fallback selected at project creation. Label with **G/B/U**, navigate
regions with **Left/Right**, undo with **Z**, and fit the image with **F**.
Drag out missed-good reference boxes during full-image review; the rectangle appears
only while drawing, and the saved annotation is shown by its `M` number. Select one
from the missed-flake list to inspect or remove it. Mark the entire image reviewed
when finished. Dataset exports include labeled crops, masks, source references,
and full-image reference annotations. Assign train/validation/test splits by sample
group; all views of the same physical sample should stay together.

## 3D contrast and profile viewer

```powershell
.\.venv\Scripts\python.exe .\tools\contrast_viewer.py
```

Choose an image and a scan profile to plot detected regions and profile class
references in signed RGB-contrast space. This viewer also reports pipeline
timings. Use `--help` for image paths, profile selection, thresholds, and plot
export options.

## Detection stages

```powershell
.\.venv\Scripts\python.exe .\tools\flake_detection_stages.py
```

Choose an image and profile to inspect the intermediate detection results.
Use `--navigator` for Previous/Next controls or `--grid` for an overview.

Both viewers import the production pipeline from `app/scanning`.
The standalone profile editor remains in `profile_editor`; launch it with
`profile_editor/run_profile_editor.bat`. Saved profiles remain in `app/Profiles`.

## Retired experiments

The former `Flake Recognition` experiments and `Brody's Work` were removed
from the working tree after verifying that all their files were tracked and
unchanged. Their last complete snapshot is commit
`6ef19b618f576b03e9754840d884b2a66af22b99`.

To recover them into a ZIP without changing the working tree:

```powershell
git archive --format=zip --output=legacy-experiments.zip 6ef19b618f576b03e9754840d884b2a66af22b99 -- "Flake Recognition" "Brody's Work"
```

The current viewers above replace their former locations under
`Flake Recognition`. This cleanup does not retire the app's own legacy
classification code.

Projects using the previous 50 µm² default are upgraded to 200 µm² when opened;
custom cutoffs and all labels are retained. Progress counts the currently listed
regions and reports filtered-out regions separately.
