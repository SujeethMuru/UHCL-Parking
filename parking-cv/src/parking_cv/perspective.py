"""Compare full-panorama detection with six overlapping perspective views."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np


def projection_maps(panorama_width, panorama_height, yaw, pitch=-12,
                    fov=100, width=1024, height=768):
    """Map a pinhole camera onto a full 2:1 equirectangular panorama.

    Yaw zero faces the panorama center. Positive pitch looks up. Longitude wraps
    across the seam; latitude is clamped at the poles rather than wrapped.
    """
    focal = width / (2 * np.tan(np.deg2rad(fov) / 2))
    x, y = np.meshgrid((np.arange(width) + 0.5 - width/2) / focal,
                       -(np.arange(height) + 0.5 - height/2) / focal)
    z = np.ones_like(x)
    p, a = np.deg2rad(pitch), np.deg2rad(yaw)
    yp, zp = y*np.cos(p) + z*np.sin(p), z*np.cos(p) - y*np.sin(p)
    xp, zz = x*np.cos(a) + zp*np.sin(a), zp*np.cos(a) - x*np.sin(a)
    longitude = np.arctan2(xp, zz)
    latitude = np.arctan2(yp, np.sqrt(xp*xp + zz*zz))
    # Padding makes bilinear sampling continuous across the longitude seam.
    map_x = ((longitude / (2*np.pi) + 0.5) * panorama_width - 0.5) % panorama_width + 1
    map_y = np.clip((0.5 - latitude/np.pi)*panorama_height - 0.5,
                    0, panorama_height-1)
    return map_x.astype(np.float32), map_y.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/images'))
    parser.add_argument('--output', type=Path, default=Path('outputs/perspective'))
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    paths = sorted(p for p in args.source.iterdir() if p.suffix.lower() in
                   {'.jpg', '.jpeg', '.png'}) if args.source.is_dir() else [args.source]
    if len(paths) != 1:
        parser.error('Choose exactly one panorama with --source path/to/photo.jpg')
    frame = cv2.imread(str(paths[0]))
    if frame is None:
        parser.error('Cannot decode source image')
    h, w = frame.shape[:2]
    if abs(w/h - 2) > 0.05:
        parser.error('Expected a full 360-degree 2:1 equirectangular panorama')
    # A separate run directory prevents stale views from entering the comparison.
    from datetime import datetime
    run = args.output / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    views = run / 'views'
    views.mkdir(parents=True)
    padded = np.concatenate((frame[:, -1:], frame, frame[:, :1]), axis=1)
    for yaw in range(0, 360, 60):
        mx, my = projection_maps(w, h, yaw)
        view = cv2.remap(padded, mx, my, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
        if not cv2.imwrite(str(views / f'yaw_{yaw:03d}.jpg'), view):
            raise RuntimeError('Could not save perspective view')
    detector = Path(__file__).with_name('detect.py')
    for source, output in [(paths[0], run/'baseline'), (views, run/'detections')]:
        subprocess.run([sys.executable, str(detector), '--source', str(source),
                        '--output', str(output), '--headless'], check=True)
    reports = [json.loads(p.read_text()) for p in sorted((run/'detections').glob('*.json'))]
    baseline = json.loads(next((run/'baseline').glob('*.json')).read_text())
    tiles = []
    for path, report in zip(sorted((run/'detections').glob('*.jpg')), reports):
        tile = cv2.resize(cv2.imread(str(path)), (600, 450))
        cv2.rectangle(tile, (0, 0), (600, 38), (25, 25, 25), -1)
        label = f"{Path(report['source']).stem}: {report['vehicle_count']} detections"
        cv2.putText(tile, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
        tiles.append(tile)
    sheet = np.concatenate([np.concatenate(tiles[i:i+3], axis=1)
                            for i in (0, 3)], axis=0)
    cv2.imwrite(str(run/'comparison.jpg'), sheet)
    summary = {'source': str(paths[0]), 'baseline_vehicle_count': baseline['vehicle_count'],
               'views': [{'yaw': i*60, 'detections': r['vehicle_count']} for i,r in enumerate(reports)],
               'projection': {'horizontal_fov':100, 'pitch':-12, 'size':[1024,768]},
               'note':'Views overlap. Counts must NOT be summed as unique vehicles. No occupancy inference.'}
    (run/'comparison.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f'Comparison saved to {run.resolve()}')
    if not args.headless:
        cv2.imshow('Perspective comparison | overlapping views | any key to close',
                   cv2.resize(sheet, (1200,600)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
