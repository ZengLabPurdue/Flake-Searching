# Workspace data safety

- `App/Flatfields` contains persistent, user-created microscope calibration
  data. Never delete, recursively clean, move, reset, or empty this directory,
  and never treat its contents as test/build artifacts.
- `App/Calibration Backups/Flatfields` contains recoverable historical
  calibrations and must receive the same protection.
- Vignette-filter creation may replace only its exact active filter file after
  preserving the previous file in the calibration backup directory.
- `App/Saved Images/Vignette Filter` is the disposable intermediate-capture
  directory used while building a filter.
