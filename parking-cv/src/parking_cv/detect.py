"""Detect vehicles in local JPG/PNG images and save a preview and JSON report."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/images'))
    parser.add_argument('--model', type=Path, default=Path('models/yolo11n.pt'))
    parser.add_argument('--output', type=Path, default=Path('outputs/detection'))
    parser.add_argument('--confidence', type=float, default=0.25)
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.confidence) or not 0 < args.confidence <= 1:
        parser.error('--confidence must be between 0 (exclusive) and 1')
    extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = (sorted(p for p in args.source.iterdir()
                     if p.is_file() and p.suffix.lower() in extensions)
              if args.source.is_dir() else [args.source])
    if not images or any(not p.is_file() or p.suffix.lower() not in extensions
                         for p in images):
        parser.error('Supply an existing image or a directory of JPG/PNG/BMP images')
    import cv2
    from ultralytics import YOLO

    args.model.parent.mkdir(parents=True, exist_ok=True)
    if not args.model.exists():
        print(f'Downloading official pretrained model to {args.model} on first run.')
    model = YOLO(str(args.model))
    vehicle_ids = [int(key) for key, name in model.names.items()
                   if name in {'car', 'motorcycle', 'bus', 'truck'}]
    if not vehicle_ids:
        parser.error('Model has no supported vehicle classes')
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        for index, path in enumerate(images, 1):
            frame = cv2.imread(str(path))
            if frame is None:
                parser.exit(1, f'Cannot decode {path}\n')
            height, width = frame.shape[:2]
            # Preserve more detail than the screen preview while bounding memory.
            scale = min(1.0, 2560 / width, 1440 / height)
            resized = cv2.resize(frame, (round(width * scale), round(height * scale)),
                                 interpolation=cv2.INTER_AREA) if scale < 1 else frame
            result = model.predict(resized, imgsz=1280, conf=args.confidence,
                                   classes=vehicle_ids, device='cpu', verbose=False)[0]
            detections = []
            sx, sy = width / resized.shape[1], height / resized.shape[0]
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({'class': model.names[int(box.cls.item())],
                                   'confidence': round(float(box.conf.item()), 4),
                                   'box_xyxy': [round(x1*sx), round(y1*sy),
                                                round(x2*sx), round(y2*sy)]})
            counts = dict(Counter(item['class'] for item in detections))
            annotated = result.plot()
            h, w = annotated.shape[:2]
            preview_scale = min(1.0, 1200/w, 700/h)
            preview = cv2.resize(annotated, (round(w*preview_scale),
                                            round(h*preview_scale)))
            cv2.rectangle(preview, (0, 0), (preview.shape[1], 42), (25, 25, 25), -1)
            cv2.putText(preview, f'Vehicles: {len(detections)} | Detection only - no space counts',
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            name = f'{index:03d}_{path.stem}'
            if not cv2.imwrite(str(args.output / f'{name}.jpg'), preview):
                raise RuntimeError('Could not save annotated preview')
            report = {'source': str(path), 'model': str(args.model),
                      'image_size': [width, height], 'confidence_threshold': args.confidence,
                      'vehicle_count': len(detections), 'counts_by_class': counts,
                      'detections': detections,
                      'note': 'Boxes use original image pixels. No occupancy inference.'}
            (args.output / f'{name}.json').write_text(json.dumps(report, indent=2))
            print(f'{path.name}: {len(detections)} vehicles {counts}')
            if not args.headless:
                cv2.imshow('UHCL-Parking | vehicle detection | any key: next, Q: quit', preview)
                if cv2.waitKey(0) & 0xFF in (ord('q'), 27):
                    break
    finally:
        if not args.headless:
            cv2.destroyAllWindows()
    print(f'Results saved locally in {args.output.resolve()}')


if __name__ == '__main__':
    main()
