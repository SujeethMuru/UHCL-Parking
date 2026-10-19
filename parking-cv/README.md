# Parking CV proof of concept

Run commands from `parking-cv/`. Playback verifies recorded input; the separate
detector draws vehicle boxes on still images. Manually calibrated spaces can show
estimated occupancy. No footage or datasets are included.

## Mark parking spaces and estimate occupancy

## Recorded 360 video occupancy

Copy your full 2:1 360 MP4 to `data/videos/parking-test-Delta-FP-360.mp4`, then run
from `parking-cv/`:

```powershell
.\.venv\Scripts\python.exe src\parking_cv\video_occupancy.py
```

The first run opens a fixed perspective view from the first frame. Mark the actual
bay boundaries, including a bay the moving car enters, and click Save + detect.
This uses the existing zoom/pan editor. The script samples the recording every
second and updates detection boxes and estimated occupancy while it processes.
This is offline sampled analysis, not live or full-frame-rate processing. Playback
may advance faster or slower than the source while computing. A moving vehicle
inside a bay polygon can temporarily count as occupied; this prototype does not
classify stopped versus moving vehicles or smooth flickering predictions.

Video outlines are separate from still-photo outlines. `config/video-spaces.local.json`
records the source fingerprint, yaw, pitch and FOV so outlines cannot silently be
reused on another recording or view. The camera and exported orientation must stay
fixed across the clip. Recalibrate if they change. Only marked bays in this one
perspective view contribute to the counts, not the entire lot.

Default view: yaw 300°, pitch -20°, horizontal FOV 90°. Options:

Two direction presets now keep independent outlines and results:

```powershell
# Working original direction (default, yaw 300):
.\.venv\Scripts\python.exe src\parking_cv\video_occupancy.py --view first
# Opposite direction (yaw 120), first run opens its own bay editor:
.\.venv\Scripts\python.exe src\parking_cv\video_occupancy.py --view second
```

The second preset uses `config/video-spaces-second.local.json` and saves results
under `outputs/video-occupancy/second-view/`. It never replaces first-view outlines.
Mark only actual parking bays, excluding the driving aisle and striped access
areas. Each command processes one direction and reports only that view's marked
bays. This is expanded coverage, not a combined full-lot counter. Both views were
checked with five video samples; the second view still needs user-marked bays
before its occupancy result can be verified. Twenty-three regression checks pass.

```powershell
# Reopen the video-specific editor:
.\.venv\Scripts\python.exe src\parking_cv\video_occupancy.py --edit
# Process saved outlines without opening a window:
.\.venv\Scripts\python.exe src\parking_cv\video_occupancy.py --headless
# Save only the initial view for inspection:
.\.venv\Scripts\python.exe src\parking_cv\video_occupancy.py --prepare
# Change direction and mark a new calibration:
.\.venv\Scripts\python.exe src\parking_cv\video_occupancy.py --yaw 240 --edit
```

Use `--source` for another 360 recording. `--start-seconds`, `--end-seconds`,
`--sample-seconds` and `--max-samples` control the processed interval and sampling.
`--detect-only --headless` runs without outlines and intentionally leaves space
counts blank. Each run saves a separate folder under `outputs/video-occupancy/`:

- `prerecorded-parking-demo.mp4`: annotated sampled preview, labeled as prerecorded.
- `counts.csv`: per-sample timestamps, vehicle count and estimated space states.
- `detections.jsonl`: per-sample boxes and matched bay polygons.
- `first-sample.jpg`, `last-sample.jpg` and `summary.json`.

Close the processing window to stop early. Completed sample records are retained.
The MP4 is encoded at one frame per sample interval so its pace follows source
timestamps, but it omits intervening frames and audio. Open it in a regular video
player, or loop it using `playback.py --source <output-mp4> --loop`.

Verified input: 7680 × 3840, 29.970 fps, 2593 frames, about 86.5 seconds. Five samples
at 0, 20, 40, 60 and 80 seconds decoded and ran detection successfully. The generated
preview reopened successfully; video vehicle counts were 0, 1, 3, 2, 1 in the test
view. This test did not establish parking counts because video bays still require
manual calibration. Twenty-two regression tests cover sampling, projection,
occupancy transitions, storage recovery and editor geometry.

All footage, calibration and generated results remain excluded from Git.

## Still-image occupancy

From `parking-cv/`, run:

```powershell
.\.venv\Scripts\python.exe src\parking_cv\occupancy.py
```

On first run, this opens `yaw_120.jpg` from the newest perspective output folder.
Click the four ground corners of a visible parking bay in clockwise or
counterclockwise order. Each four clicks adds a labeled space. Outline the painted
bay boundaries, not the car. Start with 5–10 clear bays, including the white SUV's
bay. Skip bays whose corners are hidden or cannot be identified. The Undo button
reverts points or edits; Save + detect saves and runs the detector. Closing the
editor cancels unsaved changes. Crossed or overlapping polygons are rejected.

The editor and result viewer resize to fit the screen, including the bottom edge.
Scroll the mouse wheel to zoom at the pointer, or use Zoom + / Zoom minus.
Right-drag to pan. Fit entire image restores the full view. Drawing coordinates
remain attached to the photo while zooming or panning. Adjust corners selects a
space and exposes draggable corner handles. Delete selected removes a bad outline;
Undo restores it. Existing outlines are loaded when you use `--edit`.

Suggest spaces finds experimental candidates from bright painted divider lines,
excluding detected car bodies. Orange dashed outlines are proposals only: click
one to accept it. Escape clears unaccepted proposals and pending clicks. Suggestions
can miss bays or mistake cracks/curbs for paint, especially in night images; review
them before accepting. Existing space outlines are never automatically replaced.
For faster marking, set the bay count, choose Draw row, and click the row's four
outer corners: along the first long edge, then back along the opposite long edge.
The tool generates a perspective-adjusted grid; align it with paint using Adjust
corners. This assumes equally spaced bays on a planar row and needs visual review.

The result shows occupied spaces in red and estimated free spaces in green.
Only marked spaces contribute to counts. The detector matches the lower 60% of
each vehicle box to the strongest-overlapping bay, requiring at least 20% coverage
of that lower region, or a bottom-center point inside the bay as a fallback. Each
vehicle is assigned to at most one space. Cyan rectangles show matching regions.
This image-space approximation can fail with distorted views, occlusion, inaccurate boundaries or
missed detections; green means no matching vehicle was detected, not confirmed
availability. Visually check the first result before using any count in a demo.

Outlines are saved locally in `config/spaces.local.json`. A lossless reference view
is pinned in `outputs/occupancy/reference.png`, so later perspective runs do not
silently change the calibration. The photo content must match the calibration;
this first version is for still-image review, not live or changing frames.
`outputs/occupancy/parking-status.jpg` and `parking-status.json` hold the latest
result. All of these files are ignored by Git. Saving edits preserves the previous
configuration as a dated `config/spaces.*.local.json` backup.

Saving reuses an unchanged reference image, so an open photo viewer does not
require rewriting it. Changed reference images use separate filenames linked
from the configuration. A dated `spaces.recovery.*.local.json` copy preserves new
outlines before replacing the main configuration. If Windows locks an output
picture, results are saved under a dated filename printed in PowerShell. If a
configuration is locked, the program prints the alternate saved path; reopen it
with `--config "config/the-printed-name.local.json"`. Sixteen tests include
unchanged-reference reuse, simulated file locks, and configuration recovery.

Regression check on the supplied ten-space calibration: the SUV is now assigned
to P7, giving 1 occupied and 9 estimated free. Thirteen tests cover occupancy
matching, projection geometry, fit/zoom coordinate mapping, row splitting and
painted-line suggestions. This is functionality verification, not a detection
accuracy benchmark.

```powershell
# Reopen the editor to adjust saved outlines:
.\.venv\Scripts\python.exe src\parking_cv\occupancy.py --edit
# Regenerate results using saved outlines without opening a window:
.\.venv\Scripts\python.exe src\parking_cv\occupancy.py --headless
```

## Detector setup

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-detection.txt
.\.venv\Scripts\python.exe src\parking_cv\detect.py --source data\images
```

The first detection run downloads official YOLO11n weights into `models/`.
Inference runs locally on CPU. The window shows vehicle boxes, confidence values,
and a detected vehicle count. Press any key for the next image, or Q/Escape to quit.
Annotated JPG previews and JSON reports are saved in `outputs/detection/` (ignored
by Git). `--headless` saves results without opening a window. Use `--confidence 0.4`
to require greater confidence. This is a pretrained baseline: distant vehicles and
360-degree distortion can cause misses or false detections. It does not determine
which parking spaces are occupied. Reports give boxes in original image pixels.

Detector API reference: https://docs.ultralytics.com/usage/python/

Verified locally on the supplied 11904 × 5952 parking panorama: YOLO11n at the
default 0.25 threshold detected two cars (confidence 0.90 and 0.73). The annotated
preview was inspected; this is a single-image functionality check, not an accuracy
benchmark. Model weights and generated reports remain excluded from Git.

## Perspective comparison for a 360-degree JPG

From `parking-cv/`, with detection dependencies installed:

```powershell
.\.venv\Scripts\python.exe src\parking_cv\perspective.py
```

This chooses the single JPG/PNG in `data/images/`. If there are several images,
use `--source data/images/your-photo.jpg`. Input must be a full 2:1
equirectangular panorama (not an ordinary wide photo). The script generates six
1024 × 768 perspective views at 60-degree intervals, with 100-degree horizontal
fields of view and a 12-degree downward tilt. It reruns the same detector and
threshold on the panorama and on the views. Results go into a fresh dated folder
under `outputs/perspective/`, including a comparison image and JSON summary.
Use `--headless` to save without opening a window; otherwise press any key to close.

Views overlap: **do not sum their counts as unique vehicles**. Cross-view duplicate
removal and occupancy across views remain future work. On the supplied photo,
the full panorama produced 2 detections; the six views produced 1, 7, 3, 1, 1, 3
respectively. Extra distant candidates need manual review; these results do not
establish improved accuracy. Three geometry tests verify camera directions,
longitude seam handling, and finite/clamped pole coordinates:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## Folder layout

```text
parking-cv/
  src/parking_cv/       Input playback; future detection and occupancy logic
  config/              Shareable example configuration
  data/videos/         Local recordings (ignored)
  data/images/         Local image sequences (ignored)
  models/              Local model weights (ignored)
  outputs/             Generated results (ignored)
  tests/               Future detection and occupancy tests
  requirements.txt
```

## Start

Use Python 3.10 or newer in a virtual environment, then:

```sh
python -m pip install -r requirements.txt
python src/parking_cv/playback.py --source data/videos/parking.mp4 --loop
python src/parking_cv/playback.py --source data/images --loop --fps 2
```

Place your own recording in `data/videos/parking.mp4`, or JPG/PNG images in
`data/images/`. Images play in filename order (use `001.jpg`, `002.jpg`, etc.).
Press Q or Escape to stop. Video uses its recorded FPS unless `--fps` overrides it.
`--max-frames 30 --headless` processes a bounded number of frames without a window.
The path is relative to your current directory. Network streams are intentionally
outside this initial local-footage prototype.

`config/parking.example.json` documents the planned detector/parking-space settings;
the playback command currently uses command-line arguments only. Copy to
`config/local.json` for machine-specific values (ignored). Space coordinates must be
calibrated against the actual camera view before implementing occupancy detection.

## Footage and Git

Everything in `data/` and `models/` is ignored except directory placeholders.
Generated `outputs/` and `runs/` are ignored. Video extensions are also excluded
elsewhere in the repository. Keep still images in `data/images/` so they are ignored.
Never use `git add -f` for footage. Ignore rules do not remove already tracked files.
Before each commit, inspect `git status --short` and `git diff --cached --stat`;
commit source/configuration/documentation only. Share footage outside GitHub.

## Next implementation milestones

1. Verify playback using footage from the intended parking camera.
2. Validate the implemented vehicle detector on more images and camera views.
3. Define parking-space polygons and map detections to spaces.
4. Validate occupied/free counts against manually labeled frames.
5. Connect the verified pipeline to a live stream.

## Rename the GitHub repository

An owner must open the existing repository's **Settings → General**, change
**Repository name** to `UHCL-Parking`, and click **Rename**. This renames the existing
repository and preserves its history; do not create a replacement repository.
After the rename succeeds, run from the repository root:

```sh
git remote set-url origin https://github.com/SujeethMuru/UHCL-Parking.git
git remote -v
```

The local checkout is already named `UHCL-Parking`. Its origin remains the original
URL until the online rename is completed.
