# Parking CV: challenges, fixes, and decisions

Last reviewed: October 30, 2026.

This document records the current recorded-video proof of concept. It is a
working baseline that we expect to revisit as the team changes requirements.
It does not declare the detector production-ready or the app interface final.

## Current baseline

- Full 2:1 equirectangular MP4 input, with separate front/back calibration maps.
- Eight fixed gallery views and a full panorama overview.
- Global space IDs, centered repair views, and transfers/redraws that preserve IDs.
- Standard, EV charging, accessible, and accessible EV charging categories.
- Grouping of repeated vehicle detections, agreement across views, and temporal
  confirmation before an occupancy state changes.
- CPU inference, sampled preview videos, count CSVs, category CSVs, and detailed
  JSONL observations. No live camera or application/backend connection yet.
- 53 automated regression tests at this revision.

Local calibration currently includes 38 front-lot spaces and 49 back-lot spaces.
These are counts of mapped spaces, not independently verified lot capacities.
Recordings, maps, weights, and generated results remain outside GitHub.

## Challenges and fixes

| Challenge | Cause or finding | Current response | Remaining limitation |
| --- | --- | --- | --- |
| Camera files appeared duplicated and imports complained about LRV masters | Raw camera formats and preview files can share a visible filename when extensions are hidden. Original files had also been renamed. | Use separately named full 360 MP4 exports for the CV program. Preserve original camera files when possible. | The program does not decode raw INSV/INSP files or reconstruct missing camera master files. |
| Only part of the 360 scene appeared | A perspective view intentionally covers one direction of the panorama. | Provide eight overlapping fixed views plus a panorama overview. | The gallery is a static calibration view, not an interactive spherical player. |
| Editor clipped the image or made corner placement difficult | Display size and fixed-size image windows limited access to boundaries. | Fit-to-window rendering, cursor-centered zoom, right-drag pan, adjustable corners, undo, and row drawing. | A space whose full painted boundaries are hidden still needs judgment. |
| Turning and adding directions became confusing | Saved views protected their outlines from angle changes. Repeated additions reached the direction limit. | Replace the main setup flow with eight fixed thumbnails. Retain older directions internally and keep them editable. | Extra repair directions add internal processing cost. |
| Old incomplete outlines could not be edited from a clearer view | Each physical bay had one owner view. Offscreen corners prevented direct transfer. | Allow an explicit transfer for complete outlines or a one-space redraw under the same ID. | A replacement that overlaps another saved bay fails validation. Users must correct conflicting boundaries rather than create a duplicate. |
| Near-camera bays remained cut off in all standard views | A fixed 90-degree view and downward tilt did not cover the entire ground region. | Add a wider centered repair view, tilted toward the selected bay. Save only after the user confirms its corners. | The old outline's direction guides centering, so a badly misplaced outline may need further repair. |
| Markings appeared in thumbnails but disappeared in the editor | The gallery projected all spaces, while the editor initially showed only owned spaces. Partial polygons also disappeared when any corner left the view. | Show dashed cyan references from other views and clip visible polygon portions at view edges. | Space counts still require a usable full calibration, not merely a clipped visual reference. |
| IDs overlapped and zooming the gallery looked grainy | Labels were rasterized into small thumbnails. Zooming cannot recover discarded pixels. | Add a 2048 x 1536 identification view with click-to-select. Only the selected ID appears, and parking type saves directly there. Video labels use TrueType text, dark backgrounds, and collision offsets. | Native source resolution and distortion still limit distant detail. |
| One car filled two spaces in different views | Independent assignments treated repeated detections as separate vehicles. | Group angular vehicle boxes before choosing one bay for each group. | Overlap-based grouping is approximate. It is not a learned identity tracker. |
| Two nearby cars could become one group | A broad cross-view box could indirectly connect two detections from the same view. | Forbid a group from containing two detections from one view and merge with only the best eligible group. | A detector that produces only one box for two hidden cars still provides incomplete evidence. |
| Back-lot white car filled B005 while parked in B006 | The car body visually extended over the bay behind the wheels. A large lower-body footprint biased matching. | Match the bottom fifth of the box nearer ground contact. The 100-second check assigns B006 rather than B005. | Bounding boxes do not describe tire contact or full vehicle geometry exactly. |
| Front-lot B024 appeared occupied beside the parked car | A clipped or distorted perspective box's maximum overlap overruled clearer views. | Use confidence-weighted votes across views, downweight clipped boxes, and reduce the single anchor bonus. | Views share camera evidence and are not statistically independent. |
| Occupancy flickered when detections briefly disappeared | A single frame could change availability. | Require three occupied or two free samples before changing the stable state. | Confirmation adds delay and can temporarily retain a departing car's state. |
| A white car hid the black car and its space became free | Missing detections were treated as empty despite foreground occlusion. | Mark substantially obscured, unassigned bays unknown. Hold previously confirmed occupancy for up to eight unknown samples, then report unknown. | Long occlusion cannot be resolved with certainty from this single viewpoint. |
| Different recordings rejected the same calibration | Source fingerprints intentionally prevent silent map reuse. | Visually compare the recordings and check static landmark alignment near the start, middle, and end. Copy the map, categories, and IDs to a separate source-bound configuration only when aligned. | This transfer was performed locally, not through a general automatic calibration-transfer command. Camera translation or changed export orientation may require adjustment. |
| Windows failed to save an output/reference or locked a file | An existing viewer or process could hold the destination open. | Atomic writes, unchanged-reference reuse, recovery copies, and alternate dated filenames. | Users must reopen the printed alternate config when the main config stays locked. |
| Video processing looked slow | CPU detection runs on many perspective images for each sampled timestamp. | Skip unused legacy directions and allow a larger sample interval for quick previews. | Increasing the interval misses short events and increases confirmation delay in source time. GPU optimization remains separate future work. |

## Evidence and its limits

The user visually reviewed the first front recording and both back recordings
through the iterative fixes. After the final B024 fix, the user watched the
second front recording through and reported that it looked good on October 30.
This is useful acceptance feedback, but it is not a measured accuracy score.

Targeted local replay checks used saved detection logs:

- The second back recording at 100 seconds assigns the white car to B006 rather
  than B005. The later consensus change retains that assignment.
- The revised back-lot logic can retain two separately detected cars in the
  checked 25-40-second period instead of merging them.
- Across 24 checked samples between 25 and 50 seconds in the second front
  recording, B024 never reports occupied and B025 reports occupied throughout.
- Calibration transfer checks found thousands of static landmark matches across
  the field of view. The median displacement was zero at 1920-pixel panorama
  width, with 95th-percentile displacement about two pixels in the tested frames.
  This supports reuse for these recordings, not arbitrary recordings.

Automated tests cover synthetic matching cases, projection/seam geometry,
map identity, category totals, transfer/redraw behavior, storage recovery, source
binding, gallery composition, and temporal filtering. They do not contain the
private recordings and do not prove detector accuracy on unseen footage.

## Tunable decisions

These values are current implementation choices, not permanent requirements.
Most are code defaults rather than settings exposed in the UI.

| Decision | Current behavior | Revisit when |
| --- | --- | --- |
| Sampling interval | One source second by default | Required update rate or processing budget changes |
| Occupied/free confirmation | Three occupied samples, two free samples | Desired delay or error tolerance changes |
| Occupancy hold during uncertainty | Up to eight unknown samples | Staleness policy, camera frame rate, or tracking approach changes |
| Grouping | Angular box overlap threshold 0.6, at most one detection per view per group | Adjacent cars merge, repeat detections split, or a tracker/segmenter is introduced |
| Bay evidence | Bottom fifth of box, overlap/anchor candidate gating, weighted cross-view vote | Camera angle, camera height, boundary quality, or vehicle shape changes |
| Clipped-box vote | 0.2 of the normal confidence weight | New views or edge detections change assignment quality |
| Occlusion | At least 45% of bay angular mask covered by a vehicle box | Persistent false unknown or false free states appear |
| Categories | Four mutually exclusive types, including accessible EV charging | Team needs separate accessibility/charging flags, restrictions, or additional types |
| Map identity | Source fingerprint plus space IDs within a lot | A permanent camera setup or backend mapping becomes available |

## Planned work and integration boundaries

1. Agree with Philip on the data contract and integration approach. Include lot
   ID, space ID, parking type, state, observation time, and prerecorded/live source.
   Existing CSV/JSONL outputs are debugging artifacts, not a finalized API.
2. Define how HawkNav handles unknown states, delayed updates, and missing updates.
   Never turn a missing update into a free parking space.
3. Preserve separate standard, EV, and accessible availability. If the app needs
   overlapping filters, define how accessible EV spaces contribute without
   double-counting the physical capacity.
4. Add a reproducible labeled evaluation with false occupied/free rates,
   unknown rate, and transition delay. Use a fixed test set before tuning.
5. Distinguish parked cars from vehicles passing through a bay if required.
6. Test live input when a camera is available and live operation enters scope.
7. Benchmark CPU/GPU execution separately for Heterogeneous Computing. Do not
   present proposed GPU speedup as a measured result.

## How to revisit this work

For a requirement change, first record the requested behavior, affected lots,
space categories, update rate, and acceptance examples. Then:

- Preserve the old source-bound map and a recovery copy before migration.
- Retain physical bay IDs and types when moving or repairing geometry. Do not
  reset established accessible/EV labels to Standard.
- Treat map fields and output fields as potential contracts. Add explicit schema
  versions/migrations before an app integration depends on incompatible changes.
- Keep raw evidence alongside confirmed state and record sampling/thresholds so
  a run can be explained later.
- Add a focused regression for a demonstrated failure, then replay representative
  front and back examples. Improving one camera angle can regress another.
- Update this document with the new decision, reason, evidence, and limitations.
  Update the README's commands and test count when behavior changes.
- Review staged files and sizes before pushing. Keep media, local maps, weights,
  personal machine paths, and generated results outside the source commit.

The repository is the shared code baseline. Local maps and exported recordings
must be handed off or backed up separately when another team member needs them.
