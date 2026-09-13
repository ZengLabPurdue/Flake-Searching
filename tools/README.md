# Standalone tools

Run these commands from the repository root in PowerShell.

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
