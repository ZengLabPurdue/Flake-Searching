from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FocusPlane:
    """A locally anchored Z plane fitted from autofocus measurements."""

    origin_x: float
    origin_y: float
    origin_z: float
    slope_x: float
    slope_y: float

    @classmethod
    def from_samples(cls, samples):
        samples = [
            (float(x), float(y), float(z))
            for x, y, z in samples
        ]
        if not samples:
            raise ValueError("At least one autofocus sample is required.")
        if not np.all(np.isfinite(np.asarray(samples, dtype=np.float64))):
            raise ValueError("Autofocus samples must be finite.")

        origin_x, origin_y, origin_z = samples[0]
        if len(samples) == 1:
            return cls(origin_x, origin_y, origin_z, 0.0, 0.0)

        xy_deltas = np.asarray(
            [
                (x - origin_x, y - origin_y)
                for x, y, _z in samples
            ],
            dtype=np.float64,
        )
        z_deltas = np.asarray(
            [z - origin_z for _x, _y, z in samples],
            dtype=np.float64,
        )
        slopes, _residuals, _rank, _singular_values = np.linalg.lstsq(
            xy_deltas,
            z_deltas,
            rcond=None,
        )
        slope_x, slope_y = (float(value) for value in slopes)
        if not np.isfinite(slope_x) or not np.isfinite(slope_y):
            raise ValueError("Could not fit a finite autofocus plane.")
        return cls(
            origin_x,
            origin_y,
            origin_z,
            slope_x,
            slope_y,
        )

    def z_at(self, x, y):
        return (
            self.origin_z
            + self.slope_x * (float(x) - self.origin_x)
            + self.slope_y * (float(y) - self.origin_y)
        )
