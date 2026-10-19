"""Mark parking-space quadrilaterals and estimate occupancy on a still view."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from local_storage import save_calibration,save_image,write_artifact


def validate_spaces(spaces):
    if not spaces:
        raise ValueError('Mark at least one parking space before saving.')
    polygons = []
    for points in spaces:
        polygon = np.asarray(points, dtype=np.float32)
        if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
            raise ValueError('Each space needs four finite corner coordinates.')
        if (polygon < 0).any() or (polygon > 1).any():
            raise ValueError('Space coordinates must be normalized between 0 and 1.')
        if not cv2.isContourConvex(polygon) or cv2.contourArea(polygon) < 0.0001:
            raise ValueError('Click four corners around the boundary, without crossing lines.')
        for previous in polygons:
            area, _ = cv2.intersectConvexConvex(polygon, previous)
            if area > 0.00001:
                raise ValueError('Space outlines overlap. Undo and redraw their boundaries.')
        polygons.append(polygon)


def classify_spaces(spaces, detections, width, height):
    """Match each car to its strongest lower-body overlap, with anchor fallback.

    Using an area rather than one bottom pixel tolerates bounding boxes extending
    beyond bay boundaries. It remains an image-space occupancy estimate.
    """
    validate_spaces(spaces)
    matches = [[] for _ in spaces]
    for index, detection in enumerate(detections):
        x1, y1, x2, y2 = detection['box_xyxy']
        if x2 <= x1 or y2 <= y1:
            continue
        point = ((x1+x2)/2 / width, y2 / height)
        foot = np.array([[x1/width,(y1+.4*(y2-y1))/height],
                         [x2/width,(y1+.4*(y2-y1))/height],
                         [x2/width,y2/height],[x1/width,y2/height]],np.float32)
        foot_area = cv2.contourArea(foot)
        candidates = []
        for slot, points in enumerate(spaces):
            polygon = np.asarray(points,np.float32)
            overlap,_ = cv2.intersectConvexConvex(polygon,foot)
            fraction = overlap / foot_area
            anchored = cv2.pointPolygonTest(polygon,point,False) >= 0
            if fraction >= .2 or anchored:
                candidates.append((fraction + .05*anchored, -slot))
        if candidates:
            _,negative_slot = max(candidates)
            matches[-negative_slot].append(index)
    return [{'id': f'P{i+1}', 'status': 'occupied' if ids else 'free',
             'matched_detection_indices': ids, 'corners_normalized': spaces[i]}
            for i, ids in enumerate(matches)]


def edit_spaces(frame, spaces, detections=None):
    from parking_editor import open_editor
    return open_editor(frame, spaces, validate_spaces, detections)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--config', type=Path, default=Path('config/spaces.local.json'))
    parser.add_argument('--edit', action='store_true')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    output = Path('outputs/occupancy')
    saved = json.loads(args.config.read_text()) if args.config.exists() else None
    source = args.source
    if source is None:
        if saved:
            source = Path(saved.get('reference_image',str(output/'reference.png')))
        else:
            candidates = sorted(Path('outputs/perspective').glob('*/views/yaw_120.jpg'))
            if not candidates:
                parser.error('Run perspective.py first, or pass --source path/to/view.jpg')
            source = candidates[-1]
    frame = cv2.imread(str(source))
    if frame is None:
        parser.error(f'Cannot load {source}. Pass --source to select a view.')
    digest = hashlib.sha256(frame.tobytes()).hexdigest()
    spaces = saved['spaces'] if saved and saved.get('image_sha256') == digest else []
    if saved and saved.get('image_sha256') != digest and not args.edit:
        parser.error('View changed. Use --edit to mark spaces for the new image.')
    if args.headless and (args.edit or not spaces):
        parser.error('Mark spaces interactively first; omit --headless.')
    weights = Path('models/yolo11n.pt')
    if not weights.exists():
        parser.error('Run detect.py first to download the vehicle model.')
    from ultralytics import YOLO
    model = YOLO(str(weights))
    vehicle_ids = [i for i,n in model.names.items() if n in {'car','motorcycle','bus','truck'}]
    print('Detecting vehicles locally...')
    result = model.predict(frame, imgsz=1280, conf=.25, classes=vehicle_ids,
                           device='cpu', verbose=False)[0]
    detections = [{'class':model.names[int(b.cls.item())],
                   'confidence':float(b.conf.item()),'box_xyxy':b.xyxy[0].tolist()}
                  for b in result.boxes]
    if args.edit or not spaces:
        if args.headless:
            parser.error('Mark spaces interactively first; omit --headless.')
        spaces = edit_spaces(frame, spaces, detections)
        if spaces is None:
            print('Cancelled; configuration unchanged.')
            return
        validate_spaces(spaces)
        args.config=save_calibration(frame,spaces,args.config,output)
        print(f'Space outlines saved: {args.config.resolve()}')
    validate_spaces(spaces)
    h,w = frame.shape[:2]
    statuses = classify_spaces(spaces,detections,w,h)
    occupied = sum(s['status']=='occupied' for s in statuses)
    preview = frame.copy()
    tint = frame.copy()
    for slot in statuses:
        polygon = (np.asarray(slot['corners_normalized'])*[w,h]).astype(np.int32)
        color = (0,0,255) if slot['status']=='occupied' else (0,190,0)
        cv2.fillPoly(tint,[polygon],color)
        cv2.polylines(preview,[polygon],True,color,3)
        cv2.putText(preview,f"{slot['id']} {slot['status'].upper()}",
                    tuple(polygon.mean(axis=0).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,
                    .55,color,2)
    preview = cv2.addWeighted(preview,.8,tint,.2,0)
    for d in detections:
        x1,y1,x2,y2=map(int,d['box_xyxy'])
        cv2.rectangle(preview,(x1,y1),(x2,y2),(255,200,0),1)
        # Show the lower-body matching region as well as the bottom anchor.
        lower=int(y1+.4*(y2-y1))
        cv2.rectangle(preview,(x1,lower),(x2,y2),(255,200,0),1)
        cv2.circle(preview,((x1+x2)//2,y2),5,(255,200,0),-1)
    preview = cv2.copyMakeBorder(preview,65,0,0,0,cv2.BORDER_CONSTANT)
    cv2.putText(preview,f'Marked spaces: {len(spaces)} | Occupied: {occupied} | Free: {len(spaces)-occupied}',
                (10,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
    cv2.putText(preview,'Estimates from detections only. Review green spaces; missed cars can look free.',
                (10,49),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
    output.mkdir(parents=True,exist_ok=True)
    image_path=save_image(output/'parking-status.jpg',preview)
    report = {'marked_spaces':len(spaces),'occupied':occupied,'free':len(spaces)-occupied,
              'spaces':statuses,'detections':detections,'image_sha256':digest,
              'method':'Best lower-body overlap (>=20%) or bottom-center fallback; estimated occupancy.'}
    report['preview_image']=str(image_path.resolve())
    write_artifact(image_path.with_suffix('.json'),json.dumps(report,indent=2).encode('utf-8'))
    print(f'{occupied} occupied, {len(spaces)-occupied} estimated free. Results: {output.resolve()}')
    print(f'Picture saved: {image_path.resolve()}')
    if not args.headless:
        from parking_editor import show_result
        show_result(preview)


if __name__ == '__main__':
    main()
