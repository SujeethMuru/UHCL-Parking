# UHCL-Parking

Parking computer-vision proof of concept for the UHCL senior project. Start with
recorded footage or a looping image sequence; live camera integration comes later.

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

Initial repository setup. OpenCV validation is the first implementation milestone.
