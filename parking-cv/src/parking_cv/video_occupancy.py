"""Sample a recorded 360 MP4, project a fixed view, and estimate bay occupancy."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from local_storage import save_calibration,save_image,stamp,write_artifact
from occupancy import classify_spaces,validate_spaces
from perspective import projection_maps


def view_defaults(view):
    """Keep independent calibration and outputs for the opposite camera views."""
    if view=='first':
        return 300,Path('config/video-spaces.local.json'),Path('outputs/video-occupancy')
    if view=='second':
        return 120,Path('config/video-spaces-second.local.json'),Path('outputs/video-occupancy/second-view')
    if view=='back':
        return 60,Path('config/video-spaces-back.local.json'),Path('outputs/video-occupancy/back-parking')
    raise ValueError('Unknown view')


def default_source(view):
    name='parking-test-Delta-BP-360.mp4' if view=='back' else 'parking-test-Delta-FP-360.mp4'
    return Path('data/videos')/name


def source_signature(path):
    """Bind calibration to this recording without hashing gigabytes every run."""
    size=path.stat().st_size
    digest=hashlib.sha256()
    with path.open('rb') as file:
        digest.update(file.read(1024*1024))
        file.seek(max(0,size-1024*1024))
        digest.update(file.read(1024*1024))
    return {'size':size,'edge_sha256':digest.hexdigest()}


def sample_indices(fps,frame_count,interval,start=0,end=None,limit=None):
    if not math.isfinite(fps) or fps<=0 or frame_count<=0:
        raise ValueError('Video metadata is invalid.')
    if not math.isfinite(interval) or interval<=0 or not math.isfinite(start) or start<0:
        raise ValueError('Sample interval must be positive; start must be nonnegative.')
    if interval<1/fps:
        raise ValueError('Sample interval cannot be shorter than one source frame.')
    duration=frame_count/fps
    end=duration if end is None else end
    if not math.isfinite(end) or not start<end<=duration+.001:
        raise ValueError('Choose start < end within the recording duration.')
    if limit is not None and limit<1:
        raise ValueError('Sample limit must be positive.')
    previous=-1
    count=0
    sample_number=0
    t=start
    while t<min(end,duration):
        index=round(t*fps)
        if index>=frame_count:
            break
        if index!=previous:
            yield index
            previous=index
            count+=1
            if limit is not None and count>=limit:
                break
        sample_number+=1
        t=start+sample_number*interval


def make_projector(width,height,yaw,pitch,fov):
    x,y=projection_maps(width,height,yaw,pitch=pitch,fov=fov)
    # projection_maps includes one-pixel padding; sample original frames directly.
    x-=1
    def project(frame):
        return cv2.remap(frame,x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP)
    return project


def read_frame(cap,index):
    cap.set(cv2.CAP_PROP_POS_FRAMES,index)
    ok,frame=cap.read()
    if not ok:
        raise ValueError(f'Could not decode video frame {index}.')
    return frame


def annotate(view,detections,spaces,seconds,source_name):
    h,w=view.shape[:2]
    states=classify_spaces(spaces,detections,w,h) if spaces else []
    occupied=sum(s['status']=='occupied' for s in states)
    overlay=view.copy()
    for state in states:
        polygon=(np.asarray(state['corners_normalized'])*[w,h]).astype(np.int32)
        color=(0,0,255) if state['status']=='occupied' else (0,200,0)
        cv2.polylines(overlay,[polygon],True,color,3)
        cv2.putText(overlay,state['id'],tuple(polygon.mean(axis=0).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,.65,color,2)
    for detection in detections:
        x1,y1,x2,y2=map(int,detection['box_xyxy'])
        cv2.rectangle(overlay,(x1,y1),(x2,y2),(255,200,0),2)
    result=cv2.copyMakeBorder(overlay,78,0,0,0,cv2.BORDER_CONSTANT)
    cv2.putText(result,f'PRERECORDED DEMO | {source_name} | Time {seconds:.1f}s | Vehicles: {len(detections)}',
                (12,25),cv2.FONT_HERSHEY_SIMPLEX,.64,(255,255,255),2)
    text=(f'Marked bays: {len(spaces)} | Occupied: {occupied} | Estimated free: {len(spaces)-occupied}'
          if spaces else 'Detection preview only. Mark parking bays to enable occupancy counts.')
    cv2.putText(result,text,(12,50),cv2.FONT_HERSHEY_SIMPLEX,.59,(255,255,255),1)
    cv2.putText(result,'Sampled frames. Green means no matching vehicle detected; review estimates.',
                (12,69),cv2.FONT_HERSHEY_SIMPLEX,.46,(180,210,210),1)
    return result,states


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--view',choices=['first','second','back'],default='first',help='Choose an independently calibrated parking direction and recording.')
    parser.add_argument('--config',type=Path)
    parser.add_argument('--yaw',type=float)
    parser.add_argument('--pitch',type=float,default=-20)
    parser.add_argument('--fov',type=float,default=90)
    parser.add_argument('--sample-seconds',type=float,default=1)
    parser.add_argument('--start-seconds',type=float,default=0)
    parser.add_argument('--end-seconds',type=float)
    parser.add_argument('--max-samples',type=int)
    parser.add_argument('--prepare',action='store_true',help='Save the view to mark, without processing.')
    parser.add_argument('--edit',action='store_true',help='Edit video-specific parking-space outlines.')
    parser.add_argument('--detect-only',action='store_true',help='Preview vehicle boxes without occupancy.')
    parser.add_argument('--headless',action='store_true')
    args=parser.parse_args()
    args.source=default_source(args.view) if args.source is None else args.source
    default_yaw,default_config,base=view_defaults(args.view)
    args.yaw=default_yaw if args.yaw is None else args.yaw
    args.config=default_config if args.config is None else args.config
    if not args.source.is_file():
        parser.error(f'Video not found: {args.source}')
    if not all(math.isfinite(n) for n in (args.yaw,args.pitch,args.fov)) or not -80<=args.pitch<=80 or not 30<=args.fov<=120:
        parser.error('Use finite angles, pitch -80 to 80 and FOV 30 to 120.')
    cap=cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        parser.error('Cannot open video.')
    viewer=None
    writer=None
    try:
        fps=cap.get(cv2.CAP_PROP_FPS)
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if height<=0 or abs(width/height-2)>.05:
            parser.error('Expected a 2:1 full 360 panorama. Use the -360.mp4 export.')
        try:
            indices=list(sample_indices(fps,count,args.sample_seconds,args.start_seconds,args.end_seconds,args.max_samples))
        except ValueError as exc:
            parser.error(str(exc))
        if not indices:
            parser.error('No frames in requested interval.')
        project=make_projector(width,height,args.yaw,args.pitch,args.fov)
        reference=project(read_frame(cap,0))
        metadata={'source_signature':source_signature(args.source),'yaw':args.yaw,
                  'pitch':args.pitch,'fov':args.fov,'size':[1024,768]}
        base.mkdir(parents=True,exist_ok=True)
        if args.prepare:
            prepared=save_image(base/'view-to-mark.jpg',reference)
            print(f'Video: {width}x{height}, {fps:.3f} fps, {count/fps:.1f}s')
            print(f'{args.view.capitalize()} view saved: {prepared.resolve()}')
            return
        saved=json.loads(args.config.read_text()) if args.config.exists() else None
        matches=saved is not None and saved.get('video')==metadata
        spaces=saved['spaces'] if matches else []
        if saved and not matches and not args.edit and not args.detect_only:
            parser.error('Video or view settings changed. Use --edit to calibrate this view.')
        weights=Path('models/yolo11n.pt')
        if not weights.exists():
            parser.error('Run detect.py first to obtain the model.')
        from ultralytics import YOLO
        model=YOLO(str(weights))
        vehicle_ids=[i for i,name in model.names.items() if name in {'car','truck','bus','motorcycle'}]
        def detect(view):
            result=model.predict(view,imgsz=1280,conf=.25,classes=vehicle_ids,device='cpu',verbose=False)[0]
            return [{'class':model.names[int(b.cls.item())], 'confidence':float(b.conf.item()),
                     'box_xyxy':b.xyxy[0].tolist()} for b in result.boxes]
        if not args.detect_only and (args.edit or not spaces):
            if args.headless:
                parser.error('Run once without --headless to mark bays for this video.')
            from parking_editor import open_editor
            spaces=open_editor(reference,spaces,validate_spaces,detect(reference))
            if spaces is None:
                print('Cancelled; saved calibration unchanged.')
                return
            args.config=save_calibration(reference,spaces,args.config,base,metadata=metadata)
            print(f'Video outlines saved: {args.config.resolve()}')
        if args.detect_only:
            spaces=[]
        else:
            validate_spaces(spaces)
        run=base/f'run-{stamp()}'
        run.mkdir()
        movie=run/'prerecorded-parking-demo.mp4'
        writer=cv2.VideoWriter(str(movie),cv2.VideoWriter_fourcc(*'mp4v'),
                               1/args.sample_seconds,(1024,846))
        if not writer.isOpened():
            raise RuntimeError('Cannot create output MP4. Choose a writable local folder.')
        if not args.headless:
            from parking_editor import ParkingWindow
            viewer=ParkingWindow(reference)
            viewer.root.title('UHCL Parking | processing prerecorded video')
            viewer.message.set('Processing sampled video locally | Scroll to zoom | Right-drag to pan | Close to stop')
            viewer.root.update()
        processed=0
        first_occupied=None
        last_occupied=None
        start=time.monotonic()
        with (run/'counts.csv').open('w',newline='') as csv_file, (run/'detections.jsonl').open('w') as json_file:
            fields=['source_frame','video_seconds','vehicles','occupied','estimated_free']+[f'P{i+1}' for i in range(len(spaces))]
            csv_writer=csv.DictWriter(csv_file,fieldnames=fields);csv_writer.writeheader()
            for index in indices:
                if viewer is not None:
                    try:
                        viewer.root.update()
                        if not viewer.root.winfo_exists():
                            break
                    except Exception as exc:
                        from tkinter import TclError
                        if isinstance(exc,TclError):
                            break
                        raise
                view=project(read_frame(cap,index))
                detections=detect(view)
                annotated,states=annotate(view,detections,spaces,index/fps,args.view+' view')
                occupied=sum(s['status']=='occupied' for s in states) if spaces else None
                row={'source_frame':index,'video_seconds':round(index/fps,3),'vehicles':len(detections),
                     'occupied':occupied,'estimated_free':len(spaces)-occupied if spaces else None}
                row.update({s['id']:s['status'] for s in states})
                csv_writer.writerow(row);csv_file.flush()
                json_file.write(json.dumps({'frame':index,'seconds':index/fps,'detections':detections,'spaces':states})+'\n');json_file.flush()
                writer.write(annotated)
                if processed==0:
                    first_occupied=occupied
                    save_image(run/'first-sample.jpg',annotated)
                processed+=1
                last_occupied=occupied
                if processed%5==0 or processed==len(indices):
                    print(f'{processed}/{len(indices)} samples | video {index/fps:.1f}s | vehicles {len(detections)} | occupied {occupied}',flush=True)
                if viewer is not None:
                    from PIL import Image
                    viewer.image=Image.fromarray(cv2.cvtColor(annotated,cv2.COLOR_BGR2RGB))
                    if viewer.h!=annotated.shape[0]:
                        viewer.h,viewer.w=annotated.shape[:2]
                        viewer.view.height,viewer.view.width=viewer.h,viewer.w
                        viewer.fit()
                    viewer.render()
        if processed:
            save_image(run/'last-sample.jpg',annotated)
        writer.release();writer=None
        summary={'source':str(args.source.resolve()),'source_duration_seconds':count/fps,
                 'source_fps':fps,'source_size':[width,height],'view':metadata,'view_name':args.view,
                 'sample_seconds':args.sample_seconds,'samples_processed':processed,
                 'requested_samples':len(indices),'processing_seconds':round(time.monotonic()-start,2),
                 'marked_spaces':len(spaces),'first_occupied':first_occupied,'last_occupied':last_occupied,
                 'mode':'occupancy' if spaces else 'detection_only',
                 'note':'Prerecorded sampled demonstration, no live data. Estimates can miss vehicles. One view only.',
                 'preview_video':str(movie.resolve())}
        write_artifact(run/'summary.json',json.dumps(summary,indent=2).encode())
        print(f'Finished. Results: {run.resolve()}',flush=True)
        print('MP4 is a sampled preview, not a full-frame-rate export.',flush=True)
        if viewer is not None:
            try:
                if viewer.root.winfo_exists():
                    viewer.message.set(f'Finished. Results saved in {run.name}. Close this window to return to PowerShell.')
                    viewer.root.mainloop()
            except Exception as exc:
                from tkinter import TclError
                if not isinstance(exc,TclError):
                    raise
    finally:
        cap.release()
        if writer is not None:
            writer.release()


if __name__=='__main__':
    main()
