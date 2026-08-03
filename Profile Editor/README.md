# Standalone Profile Editor

This application contains only the scan-profile creator, loader, and preview
workflow. It does not initialize microscope, camera, stage, map, or scan
hardware.

Run `run_profile_editor.bat`, or from PowerShell:

```powershell
.\.venv\Scripts\python.exe ".\Profile Editor\profile_editor_app.py"
```

Profiles are read from and saved to `App\Profiles`, using the same shared
profile format and implementation as the main scanning application.
For valid Windows folder names, the profile folder now preserves the profile
name exactly, including spaces and parentheses.

Each class can use either:

- **Sampled color** — load a vignette-corrected image, click a reference
  region, and set its color tolerance.
- **Contrast range** — enter inclusive minimum and maximum signed contrast
  values for red, green, and blue. A region matches only when all three
  channels are inside their ranges; no reference image is required.

Use **Preview > Magnification** and **Preview > Resolution** to select the
calibration used for physical measurements and geometry filters.

Profiles can contain any number of optional filters. Size uses the filled-pixel
area in µm²; bounding-box width and height use their independent linear
dimensions. These and perimeter-to-area filters use inclusive bounds and can be
enabled or disabled independently. The neural-network flake filter is shown as
a disabled placeholder until that model is implemented.

After running a preview, use **Previous** and **Next** to inspect each accepted
region. Its class, group, filled area, longest side, bounding-box dimensions,
perimeter, and perimeter-to-area ratio are shown in the panel rather than
written onto the preview image. The full processed image remains visible while
the current region alone retains its class color. Enable **Display everything**
to show all classified regions, and use **Show class legend** to add or remove
the legend without rerunning the profile.

Use **Save preview image** to export the processed preview. Its options dialog
can export the selected region or all accepted regions and independently add
area labels, perimeter-to-area labels, and the class legend.
