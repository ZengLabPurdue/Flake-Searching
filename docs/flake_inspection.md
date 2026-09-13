# Save a selected flake at 100X

Open a completed region-detection scan, select a flake, and click **Image under 100X**. Keep the chip in its original placement and the stage coordinate
reference unchanged. The scan's profile and source images must still exist.

The microscope returns to the predicted position at 10X, autofocuses, and runs
the saved profile on a fresh frame. It checks the nearest region's class and
approximate area before centering a point inside its mask. It stops if no
plausible candidate is within 25 µm, or another plausible candidate is nearly
as close. In that case it displays a frozen 10X frame and waits for you to click
inside the intended flake or return to results. No additional template image is
stored for every flake.

Next it captures a vignette-corrected 20X context image, switches to 100X,
autofocuses, and displays the live camera. The **Save Flake** panel appears below
the results controls:

- **Refocus at 100X** repeats autofocus.
- **Save under 100X** captures a fresh 100X image and exports the 20X/100X pair and
  metadata through the existing database intake process. The 100X image is not
  vignette-corrected, matching the existing 100X capture workflow.
- **Back to results**, **Next**, or **Previous** leaves inspection without an
  additional export. The microscope finishes cleanup and returns to 10X before
  another inspection can start.

Local images are retained in the scan's `Selected Flakes/20x` and
`Selected Flakes/100x` directories. A cancelled inspection may retain its 20X
context image. Database export happens only on confirmation. Repeated clicks
within a successfully saved inspection do not produce duplicate exports.

On first inspection, existing detections gain stable flake IDs and physical
coordinates in `flakes_found.json`. Positions are normalized to a 10X stage
reference. The scan origin is frozen at the center of the saved tile positions;
older scans without tile-position metadata use the detection extent instead.
Chip origins use the original chip-center coordinates recorded in scan folder
names, when available. Each saved inspection also records its refined position
and actual 20X/100X stage positions. These coordinates do not register a chip
that has been removed, rotated, or relocated.

Hardware movement runs on the existing managed scan worker. Manual hardware
controls and new scans stay locked through inspection. Cancellation and app
shutdown wait for that worker to finish; callbacks from an abandoned inspection
cannot enable saving for a different selection.
