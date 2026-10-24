# UHCL-Parking

Parking computer-vision proof of concept for the UHCL senior project. Start with
recorded 360 footage; live camera integration comes later.

## Current parking prototype

- Eight fixed editing views and a full 360 overview with saved parking-space IDs.
- Zoomable polygon editor, with moving/redrawing spaces from a clearer view.
- Combined video occupancy, cross-view vehicle grouping, and confirmation to
  reduce duplicate assignments and flickering states.
- Readable labels, sampled preview MP4s, timestamped counts, and observation logs.
- Separate front/back lot maps and recovery copies, kept local alongside footage.

From `parking-cv/`, after installing the detection dependencies and adding local
footage:

```powershell
.\.venv\Scripts\python.exe src\parking_cv\multi_view.py --setup
# Later runs with saved markings:
.\.venv\Scripts\python.exe src\parking_cv\multi_view.py
```

The current suite has 45 regression tests. Local recordings have been exercised,
but full-video occupancy accuracy still needs measured validation. The next step
is to check arrivals/departures and false occupied/free results on both lots,
then integrate verified, timestamped results into HawkNav. Live streaming and
stationary-car classification remain future work.

See [parking-CV setup](parking-cv/README.md) for the isolated prototype and footage rules.
The original project plan is preserved below for reference.

## Original indoor-navigation research plan

Computer-vision research and prototype development for the UHCL indoor-navigation senior project, with GPU-performance experiments that can also support CENG 6536 Heterogeneous Computing.

## Goals

- Explore computer-vision techniques useful for indoor navigation and visual localization.
- Prototype image and video processing with OpenCV.
- Evaluate YOLO where object or landmark detection is useful.
- Compare CPU and NVIDIA CUDA GPU execution for relevant workloads.
- Measure latency, FPS, scalability, and data-movement overhead.
- Develop components that can later integrate with the main UHCL indoor-navigation application.

## Current hardware / acceleration environment

Development and benchmarking currently target an NVIDIA GeForce RTX 3060 Ti (8 GB) with CUDA 12.6 under WSL2 Ubuntu.

## Planned progression

1. Validate OpenCV image and video processing.
2. Explore visual features and landmark matching for indoor localization.
3. Test YOLO inference where object/landmark detection adds value.
4. Establish repeatable CPU and GPU benchmarks.
5. Evaluate memory-transfer and end-to-end latency, not only kernel/inference time.
6. Integrate useful vision functionality into the UHCL indoor-navigation project.

## Repository structure

```text
src/
  opencv/       OpenCV prototypes
  yolo/         YOLO experiments
benchmarks/     CPU/GPU timing and scalability experiments
tests/          Automated tests
data/
  images/       Local test images (large/raw datasets should not be committed)
  videos/       Local test videos (large/raw datasets should not be committed)
docs/           Design notes, experiment notes, and integration documentation
models/         Model documentation/placeholders; large weights should not be committed
```

## Heterogeneous Computing connection

The Senior Project uses the computer-vision functionality itself. The Heterogeneous Computing project can use the same workload to study CPU vs. GPU performance, CUDA acceleration, memory/data-movement costs, and scalability. This keeps application development and performance research connected without maintaining duplicate implementations.

## Status

The parking proof of concept above is the active implementation. The original
indoor-navigation plan and experiments remain preserved for reference.
