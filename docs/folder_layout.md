# Folder names

Python source folders use lowercase names, with underscores between words:

- `app/`: main microscope application; launch with
  `.\.venv\Scripts\python.exe .\app\launch.py` from the repository root.
- `app/database`, `app/hardware`, `app/imaging`, `app/mapping`,
  `app/scanning`, and `app/ui`: application code packages.
- `profile_editor/`: standalone profile editor; double-click
  `profile_editor/run_profile_editor.bat` to open it.
- `microscope_control/`: hardware experiments and examples.
- `tools/`: standalone diagnostics and maintenance scripts.
- `tests/`: automated tests.

User-facing data folders retain their readable names, including `Profiles`,
`Scans`, `Saved Images`, `Flatfields`, `Models`, and `Labeled Data`.
Vendor SDK bundles retain their supplied directory names (`PriorSDK1.9.2`).
The Python environment's own directory structure, such as `.venv/Scripts`,
also stays as supplied.

Use lowercase package names in imports, for example
`from scanning.scan_profile import ScanProfile` when `app` is on the Python
search path. User-facing window titles and labels keep their normal capitalization.
