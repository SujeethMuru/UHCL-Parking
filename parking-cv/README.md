# Parking CV proof of concept

Run commands from `parking-cv/`. This first component verifies recorded input and
looping playback. It does **not** detect vehicles or report parking availability yet.
No footage, model weights, or datasets are included or downloaded automatically.

## Structure

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
2. Add vehicle detection using locally supplied weights.
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
