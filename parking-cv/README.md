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

## Watch for camera snapshots

Set the camera to take a photo every minute. Keep it in the same position and
orientation as the parking calibration. Configure your export or import workflow
to place **stitched 2:1 equirectangular JPEGs** in a local folder. The watcher
starts at that folder. A camera writing only to its SD card needs a separate
transfer step before the laptop can see each photo. Raw fisheye captures need
stitching before import.

From `parking-cv`, use the calibration that matches your camera. For the
verified front-lot setup:

```powershell
New-Item -ItemType Directory -Force .\data\camera-inbox
.\.venv\Scripts\python.exe .\src\parking_cv\watch_snapshots.py --inbox .\data\camera-inbox --config .\config\multi-front-02.local.json
```

Copy a stitched JPEG into the folder to try it. If the export workflow supports
it, write to a temporary name and rename to `.jpg` when complete. The watcher
also waits for each file to settle, processes an existing batch on startup,
and skips snapshots whose reports already exist. Run one watcher per inbox.
Use Ctrl+C to stop, or add `--once` to process a batch and exit.

Each completed snapshot has a JSON report in `outputs/snapshots/` with a
snapshot ID, lot, file modification time, analysis time, occupancy counts,
and the status of each calibrated space. The modification time comes from the
imported file and may differ from the camera's capture time. Results are
single-photo estimates saved locally for later integration with HawkNav.

### Send snapshot results to Supabase

Run the watcher on the camera laptop with `--supabase-config` to POST a batch
of spot updates per photo. Each row contains `parking_lot_id`, `parking_spot_id`,
`parking_spot_type`, and `is_available`. Supabase generates `id` and
`updated_at`. The batch includes every calibrated spot, with `is_available`
set to JSON `true` for available, `false` for occupied, and `null` for unknown.
The calibration's accessible spots use the database label `handicap`.

Copy `config/hawknav.example.json` to `config/hawknav.local.json`. It uses
`parking_spot_updates` and lot ID `1`. The watcher uses the number in each
calibrated label as `parking_spot_id`. For example, `B001` becomes `1` and
`B019` becomes `19`.

Set the project URL and enter a key authorized to insert into the table.
Keep credentials in the watcher's environment on the camera laptop.

```powershell
$env:SUPABASE_URL = 'https://YOUR-PROJECT.supabase.co'
$env:SUPABASE_KEY = [System.Net.NetworkCredential]::new('', (Read-Host 'Supabase API key' -AsSecureString)).Password
.\.venv\Scripts\python.exe .\src\parking_cv\watch_snapshots.py --inbox .\data\camera-inbox --config .\config\multi-front-02.local.json --supabase-config .\config\hawknav.local.json
```

For a publishable key with a signed-in role, supply that role's JWT in
`SUPABASE_ACCESS_TOKEN`. For a trusted deployment using a secret key, the
watcher uses the key in the API key header.

The watcher retries uploads that fail to connect or receive a definite
rejection. If a POST may have reached the database and the outcome is unknown,
the report stays in `sending-*` under `outputs/snapshots/` for review while
new snapshots continue. Check Supabase before moving that report to `sent-*`
if it was inserted or back to `pending-*` if it was absent. The generated IDs
give each insert a new identity, so automatically resending an uncertain batch
could duplicate rows. A restart resumes pending uploads even after the source
photos have been removed.

Use a fresh inbox for deployment if you only want new captures. Adding
Supabase delivery also processes photos already in the inbox, including photos
previously analyzed in local-only mode. Run one watcher per inbox and output
directory.

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

For near-camera spaces cut off in the standard views, select the ID under
**Repair a cut-off space** in the gallery and choose **Open centered view**.
This opens a wider, downward-facing perspective centered on that bay. Adjust
its corners and save. The repair keeps the same ID and adds an internal counting
direction; the eight gallery views remain fixed. Closing without saving leaves
the original map unchanged. Neighboring cyan outlines remain references.

This repair option works for both front and back lots. Start the front gallery
with `multi_view.py --setup` and the back gallery with `--lot back --setup`.

### Standard, EV charging, and accessible spaces

For crowded or distant spaces, choose **Identify / label spaces (clear view)**
in the gallery. Pick View 1-8, then click inside a space. The toolbar shows its
global ID and only that space receives an on-image text label, avoiding overlaps.
Choose its type and **Save type**, then click the next space. Each change saves
immediately. **Back to gallery** returns to setup. This view projects the original
panorama at 2048 x 1536 instead of enlarging a small thumbnail. Zoom still cannot
recover detail absent from the camera recording.

In either lot's gallery, choose a space ID under **Parking type**, select its
designation, and click **Save parking type**. The map saves immediately and the
gallery refreshes. Options are Standard, EV charging, Accessible, and Accessible
EV charging. The combined designation is a distinct category so every space
contributes once to the total. Existing untyped spaces default to Standard until
you label them manually. The program does not infer signage or eligibility.

Gallery/video labels use `[EV]`, `[ACC]`, or `[ACC+EV]`. Types survive edits,
transfers, and centered repairs. Video summaries show separate free, occupied,
and unknown counts by type. Each run also writes `category-counts.csv` and adds
`by_type` to JSON totals, while `counts.csv` retains its existing total columns.

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

Bay assignment uses confidence-weighted agreement across perspective views,
with clipped boxes receiving less weight. One distorted view's maximum overlap
cannot override agreement from clearer views. Assignment logs include bay votes.
Matching emphasizes the bottom fifth of vehicle boxes near ground contact,
reducing assignment to the space behind a tall car body. A cross-view vehicle
group cannot contain two detections from the same view, preventing a broad box
from merging two adjacent cars. An unassigned bay substantially covered by a
vehicle box becomes unknown rather than automatically free. Previously confirmed
occupancy can persist for up to eight unknown samples during an occlusion, then
becomes unknown if evidence does not return. This is bounded memory, not proof
that a hidden car remains parked. At slower sample intervals the hold lasts longer.

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

See [challenges, fixes, and decisions](../docs/parking-cv-challenges-and-decisions.md)
for failure cases, their fixes, tunable thresholds, and guidance for revisiting
the implementation when requirements change.

Run regression checks:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Current suite: **53 tests**, including projection/seam geometry, editor transforms,
storage recovery, source binding, bay transfers/redraws, eight-view display,
cross-view vehicle assignment, and occupancy confirmation.

Local functionality checks used front and back 7680 x 3840 recordings. The front
map currently has 38 spaces. A targeted 21-27-second check removed the reported
B001/B002 duplicate assignment and held the confirmed total at two while filtering
a one-sample false positive. These checks are not an accuracy benchmark; footage
and maps are excluded from Git, so test results do not reproduce those local
recordings on a fresh clone.

As of October 30, the user has visually reviewed the recorded-video workflow,
including the second front video after the B024 correction, and reported that
it looks good. Targeted saved-detection replay checks also verified the B006
back-lot assignment and the B024/B025 front-lot correction. Visual acceptance
does not replace a labeled accuracy evaluation.

The second recordings use separate local maps, `multi-front-02.local.json` and
`multi-back-02.local.json`, copied after visual and static-landmark alignment
checks. These copies preserve types and IDs. They are not included in GitHub,
and there is no general automatic transfer command yet. Once a verified map
exists locally, run, for example:

```powershell
.\.venv\Scripts\python.exe src\parking_cv\multi_view.py --lot front --source data\videos\parking-test-Delta-FP-360-02.mp4 --config config\multi-front-02.local.json
.\.venv\Scripts\python.exe src\parking_cv\multi_view.py --lot back --source data\videos\parking-test-Delta-BP-360-02.mp4 --config config\multi-back-02.local.json
```

This remains a working baseline. Revisit the matching rules, categories, timing,
and interface when requirements change, preserving existing IDs and calibrations.

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
