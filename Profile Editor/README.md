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

Each class can use either:

- **Sampled color** — load a vignette-corrected image, click a reference
  region, and set its color tolerance.
- **Contrast range** — enter inclusive minimum and maximum signed contrast
  values for red, green, and blue. A region matches only when all three
  channels are inside their ranges; no reference image is required.

Set the preview magnification and resolution in the toolbar before running a
profile preview when the profile uses physical size limits.
