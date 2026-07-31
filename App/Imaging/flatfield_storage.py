"""Persistent storage helpers for active and historical flatfield calibrations."""

from datetime import datetime
from pathlib import Path
import re
import secrets
import shutil

import cv2

from config import HOME_DIR


FLATFIELD_DIR = HOME_DIR / "Flatfields"
FLATFIELD_BACKUP_DIR = HOME_DIR / "Calibration Backups" / "Flatfields"
VIGNETTE_SUBSTRATES = ("100nm", "285nm_4in", "285nm_8in")
DEFAULT_VIGNETTE_SUBSTRATE = VIGNETTE_SUBSTRATES[0]


def _calibration_key(value, label):
    key = str(value).strip().lower()
    if not key or re.fullmatch(r"[a-z0-9._-]+", key) is None:
        raise ValueError(f"{label} has invalid characters for a calibration filename.")
    return key


def normalize_vignette_substrate(substrate):
    substrate = _calibration_key(substrate, "Substrate")
    if substrate not in VIGNETTE_SUBSTRATES:
        supported = ", ".join(VIGNETTE_SUBSTRATES)
        raise ValueError(
            f"Unknown vignette substrate: {substrate}. "
            f"Choose one of: {supported}."
        )
    return substrate


def vignette_filter_path(
    magnification,
    resolution,
    *,
    substrate=None,
    flatfield_dir=FLATFIELD_DIR,
):
    magnification = _calibration_key(magnification, "Magnification")
    resolution = _calibration_key(resolution, "Resolution")
    filename_parts = ["vignette_filter", magnification]
    if substrate is not None:
        filename_parts.append(normalize_vignette_substrate(substrate))
    filename_parts.append(resolution)
    return Path(flatfield_dir) / f"{'_'.join(filename_parts)}.png"


def save_vignette_filter(
    image_bgr,
    magnification,
    resolution,
    *,
    substrate=None,
    flatfield_dir=FLATFIELD_DIR,
    backup_dir=FLATFIELD_BACKUP_DIR,
    timestamp=None,
):
    """Atomically replace one active filter after backing up its prior version."""
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("The vignette filter must be a three-channel BGR image.")

    save_path = vignette_filter_path(
        magnification,
        resolution,
        substrate=substrate,
        flatfield_dir=flatfield_dir,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)

    backup_path = None
    if save_path.exists():
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        captured_at = timestamp or datetime.now()
        timestamp_text = captured_at.strftime("%Y%m%d_%H%M%S_%f")
        backup_path = (
            backup_dir
            / f"{save_path.stem}_{timestamp_text}{save_path.suffix}"
        )
        shutil.copy2(save_path, backup_path)

    temporary_path = save_path.with_name(
        f".{save_path.stem}_{secrets.token_hex(8)}.tmp{save_path.suffix}"
    )
    try:
        if not cv2.imwrite(str(temporary_path), image_bgr):
            raise OSError(f"Could not save vignette filter to {save_path}")
        temporary_path.replace(save_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return save_path, backup_path
