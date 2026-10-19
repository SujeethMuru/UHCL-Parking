"""Play local parking footage or an image sequence; no detection is performed."""

import argparse
import math
from pathlib import Path
import time


def frames(source, loop, cv2):
    if source.is_dir():
        images = sorted(p for p in source.iterdir()
                        if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'})
        if not images:
            raise ValueError(f'No images in {source}')
        while True:
            for path in images:
                frame = cv2.imread(str(path))
                if frame is None:
                    raise ValueError(f'Cannot decode image: {path}')
                yield frame, 2.0
            if not loop:
                return
    else:
        while True:
            capture = cv2.VideoCapture(str(source))
            try:
                if not capture.isOpened():
                    raise ValueError(f'Cannot open video: {source}')
                fps = capture.get(cv2.CAP_PROP_FPS)
                if not math.isfinite(fps) or fps <= 0:
                    fps = 30.0
                count = 0
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    count += 1
                    yield frame, fps
                if not count:
                    raise ValueError(f'No decodable frames: {source}')
            finally:
                capture.release()
            if not loop:
                return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--loop', action='store_true')
    parser.add_argument('--fps', type=float)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--max-frames', type=int)
    args = parser.parse_args()
    if not args.source.exists():
        parser.error(f'Local source does not exist: {args.source}')
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error('--fps must be a finite positive number')
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error('--max-frames must be positive')
    if args.headless and args.loop and args.max_frames is None:
        parser.error('Headless looping requires --max-frames')
    try:
        import cv2
    except ImportError:
        parser.error('Install dependencies: python -m pip install -r requirements.txt')
    stream = frames(args.source, args.loop, cv2)
    count = 0
    try:
        for frame, recorded_fps in stream:
            start = time.monotonic()
            count += 1
            if not args.headless:
                # Fit large camera panoramas on screen without changing source data.
                height, width = frame.shape[:2]
                scale = min(1.0, 1200 / width, 700 / height)
                preview = frame
                if scale < 1.0:
                    preview = cv2.resize(frame, (max(1, round(width * scale)),
                                                 max(1, round(height * scale))),
                                         interpolation=cv2.INTER_AREA)
                cv2.imshow('UHCL-Parking | playback only | Q to quit', preview)
                delay = max(1, round(1000 / (args.fps or recorded_fps)
                                     - (time.monotonic() - start) * 1000))
                if cv2.waitKey(delay) & 0xFF in (ord('q'), 27):
                    break
            if args.max_frames is not None and count >= args.max_frames:
                break
    except (ValueError, cv2.error) as exc:
        parser.exit(1, f'{exc}\n')
    finally:
        stream.close()
        if not args.headless:
            cv2.destroyAllWindows()
    print(f'Processed {count} frames (playback only; no occupancy predictions).')


if __name__ == '__main__':
    main()
