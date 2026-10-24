# Parking computer vision proof of concept

Recorded 360 parking-video analysis for UHCL-Parking. The recommended workflow
uses eight fixed views, a shared map of physical parking spaces, and a full 360
calibration overview. This is sampled offline analysis, not a live availability
service. Footage, weights, calibration maps, and generated results are local only.

## Install on Windows

Use Python 3.10 or newer with Tk support. From the repository root:

```powershell
cd parking-cv
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-detection.txt
```

The detector uses YOLO11n locally on CPU. To download its weights on first setup,
place a JPG in `data/images/` and run:

```powershell
.\.venv\Scripts\python.exe src\parking_cv\detect.py --source data\images --headless
```

Weights are stored in ignored `models/yolo11n.pt`. Existing installations do not
need to repeat setup. No recordings or calibrated maps are supplied by GitHub.

## Prepare a recording

Export a **full 360, 2:1 equirectangular MP4**, keeping camera position and export
orientation fixed throughout the clip. A reframed flat video does not retain the
whole panorama. Copy recordings into ignored `data/videos/`:

- Front lot: `parking-test-Delta-FP-360.mp4`
- Back lot: `parking-test-Delta-BP-360.mp4`

For another filename, use `--source data/videos/your-recording.mp4`. A different
recording requires a new `--config config/your-map.local.json` and calibration;
the source fingerprint prevents silently applying an unrelated map.

## Mark and edit spaces

Run commands from `parking-cv/`:

```powershell
# Front lot gallery and space editor:
.\.venv\Scripts\python.exe src\parking_cv\multi_view.py --setup
# Back lot uses a separate recording and map:
.\.venv\Scripts\python.exe src\parking_cv\multi_view.py --lot back --setup
```

1. Click **Mark view 1** through **Mark view 8**, or click a thumbnail.
2. Choose **Draw space** and click four corners around a painted parking bay in
   boundary order. Exclude driving aisles and striped access areas.
3. Use **Adjust corners** to drag handles, or **Delete selected** / **Undo** to fix
   an outline. **Draw row** divides a row quadrilateral into the chosen bay count.
4. Click **Save + detect** to save and return to the gallery. All saved markings
   also appear on the full 360 panorama below the thumbnails.
5. Choose **Save and process all views** when ready.

Scroll or use **Zoom + / Zoom -** to enlarge details; right-drag to pan.
**Fit entire image** restores the complete image. Closing an editor cancels its
unsaved corner changes. **Suggest spaces** offers experimental painted-line
candidates; click to accept only after visual review. Nighttime paint, cracks,
curbs, and occlusion make automatic suggestions unreliable.

### Edit the same space from a clearer view

Yellow outlines are editable in the current view. Dashed cyan outlines are
spaces already mapped elsewhere. Each physical space has one global ID; do not
redraw cyan outlines as new spaces.

Select its ID under **Edit an existing space here**, then **Edit here / redraw**:

- A fully visible outline moves directly to this editing view, preserving its
  global ID. The move saves immediately and reopens the editor.
- An entry labeled **(redraw)** opens a replacement editor. Draw exactly one new
  four-corner outline and save to replace the old outline with the same ID.
  Closing without saving keeps the old outline.

Save other pending edits before moving a space. Earlier custom directions with
markings remain available through **Edit earlier markings**. Existing legacy
single-view maps are imported only when their source fingerprint matches.

## Process saved markings

```powershell
# Front lot, using saved outlines:
.\.venv\Scripts\python.exe src\parking_cv\multi_view.py
# Back lot:
.\.venv\Scripts\python.exe src\parking_cv\multi_view.py --lot back
# Short check without a window:
.\.venv\Scripts\python.exe src\parking_cv\multi_view.py --headless --max-samples 5
```

Results use the same eight views in four columns and two rows. Legacy directions
with owned spaces may be evaluated internally; unused old directions are skipped.
Close the processing window to stop early and retain completed sample records.
The window advances as samples finish computing, not at the source frame rate.

- **Red:** occupied estimate.
- **Green:** estimated free; a missed vehicle can still produce a false free.
- **Orange:** unknown, incomplete edge outline, or awaiting confirmation.
- **Cyan boxes:** vehicle detections.

The same ID may appear in several overlapping views but contributes once to
totals. Angular vehicle-box overlap groups repeat detections across directions,
then assigns each group to one best matching bay. Geometric duplicate-outline
checks also help prevent repeated spaces; both checks are approximate.

Occupancy requires three consecutive occupied samples or two free samples before
changing. At the default one-second sampling interval this adds roughly 2-3
seconds of confirmation delay and suppresses short detection flicker. Startup
states remain unknown until confirmed. `--sample-seconds` changes the interval,
so the delay is measured in samples, not fixed seconds. The sampled output MP4
follows source time but omits intervening frames and audio.

Labels use antialiased TrueType lettering, dark backgrounds, and offsets for
crowded spaces. Zoom the viewer to inspect distant IDs.

## Local maps and results

- `config/multi-front.local.json`, `config/multi-back.local.json`: saved map,
  source fingerprint, fixed/custom directions, global IDs, and polygons.
- Dated `*.recovery.*.local.json`: recovery snapshots before map replacement.
- `outputs/view-gallery/front/` or `back/`: eight separate reference JPGs,
  annotated JPGs, `360-marked-overview.jpg`, and `eight-views-and-360.jpg`.
- `outputs/multi-view/front/run-.../` or `back/run-.../`: combined sampled MP4,
  `counts.csv`, `observations.jsonl`, first/last preview JPGs, `summary.json`, and
  the run's map snapshot. JSONL records raw and confirmed bay states.

If a configuration is locked, the program prints an alternate path; use
`--config "config/the-printed-name.local.json"` next time. Recovery copies retain
outlines if the main save fails. Keep local maps backed up separately; cloning
GitHub does not restore footage or your markings.

## Other tools

These older commands remain available for isolated experiments:

```powershell
# One direction only; presets: first, second, back:
.\.venv\Scripts\python.exe src\parking_cv\video_occupancy.py --view first --edit
# Still-image occupancy / editor:
.\.venv\Scripts\python.exe src\parking_cv\occupancy.py --edit
# Generate six perspective views from a full 2:1 panorama JPG:
.\.venv\Scripts\python.exe src\parking_cv\perspective.py
# Loop video or a directory of JPG/PNG files:
.\.venv\Scripts\python.exe src\parking_cv\playback.py --source data\videos\parking.mp4 --loop
.\.venv\Scripts\python.exe src\parking_cv\playback.py --source data\images --loop --fps 2
```

Single-view `video_occupancy.py` and still-image occupancy use their own maps and
simpler matching; shared matching and confirmation apply to `multi_view.py`.
The perspective comparison reports raw per-view detections: do not sum them as
unique vehicles. Use `--help` on each script for its supported options.

## Validation and remaining work

Run regression checks:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Current suite: **45 tests**, including projection/seam geometry, editor transforms,
storage recovery, source binding, bay transfers/redraws, eight-view display,
cross-view vehicle assignment, and occupancy confirmation.

Local functionality checks used front and back 7680 x 3840 recordings. The front
map currently has 38 spaces. A targeted 21-27-second check removed the reported
B001/B002 duplicate assignment and held the confirmed total at two while filtering
a one-sample false positive. These checks are not an accuracy benchmark; footage
and maps are excluded from Git, so test results do not reproduce those local
recordings on a fresh clone.

Next steps:

1. Review the complete front and back recordings against manually recorded bay
   states at arrival, parked, departure, and empty moments. Measure false occupied,
   false free, and confirmation delay.
2. Correct boundaries or matching issues found in that review. Edge-cut outlines
   remain unknown until corrected in a clearer view.
3. Distinguish parked cars from vehicles passing through a bay. Current matching
   can count a moving vehicle while it intersects a mapped space.
4. Connect validated, timestamped occupancy to the HawkNav application, including
   an explicit stale/unknown state when updates stop.
5. Add and validate live-stream input after the recorded-video workflow is proven.

No automatic parking-line model, stationary-car classification, application/API
integration, live stream, or confirmed full-lot availability is implemented yet.

## Footage and Git

`data/`, `models/`, `outputs/`, `runs/`, local maps, and the virtual environment are
ignored. Video/model extensions elsewhere are also excluded. Keep still images
inside `data/images/`. Never force-add footage, weights, generated results, or
local maps. Inspect staged names and sizes before committing; share media outside
GitHub. Existing source code and the original research plan are preserved.
