"""Shared physical bay map and sampled occupancy across selected 360 directions."""
from copy import deepcopy
import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

from local_storage import save_image,stamp,write_artifact
from occupancy import classify_spaces,validate_spaces
from parking_map import register_spaces,validate_map,combined_counts,cut_off,transfer_space,replace_space,reproject
from video_occupancy import default_source,source_signature,view_defaults,make_projector,read_frame,sample_indices


def initial_map(lot,signature):
    keys=['first','second'] if lot=='front' else ['back']
    mapping={'version':1,'lot':lot,'source_signature':signature,'views':[],'spaces':[]}
    for key in keys:
        yaw,config,_=view_defaults(key)
        view={'id':key,'yaw':yaw,'pitch':-20,'fov':90}
        saved=json.loads(config.read_text()) if config.exists() else None
        if saved and saved.get('video',{}).get('source_signature')==signature:
            for field in ('yaw','pitch','fov'):
                view[field]=saved['video'][field]
            mapping['views'].append(view)
            mapping,skipped=register_spaces(mapping,key,saved['spaces'])
            print(f'Imported {len(saved["spaces"])-len(skipped)} saved bays from {key}.',flush=True)
        else:
            mapping['views'].append(view)
    validate_map(mapping)
    return mapping


def persist(path,mapping):
    validate_map(mapping)
    data=json.dumps(mapping,indent=2).encode('utf-8')
    recovery=path.with_name(f'{path.stem.replace(".local", "")}.recovery.{stamp()}.local.json')
    write_artifact(recovery,data)
    actual=write_artifact(path,data)
    if actual!=path:
        print(f'Map was locked; use --config "{actual}" next time.',flush=True)
    return actual


def infer(model,frame):
    classes=[i for i,name in model.names.items() if name in {'car','truck','bus','motorcycle'}]
    result=model.predict(frame,imgsz=1280,conf=.25,classes=classes,device='cpu',verbose=False)[0]
    return [{'box_xyxy':b.xyxy[0].tolist(),'confidence':float(b.conf.item()),
             'class':model.names[int(b.cls.item())]} for b in result.boxes]


def setup(panorama,mapping,config,model):
    from view_gallery import select_action,editor_references
    from parking_editor import open_editor
    h,w=panorama.shape[:2]
    pending_view=None
    while True:
        action=('mark',mapping,pending_view) if pending_view else select_action(panorama,mapping)
        pending_view=None
        if action is None:
            return None,config
        kind,mapping,key=action
        if kind=='process':
            return mapping,persist(config,mapping)
        view=next(v for v in mapping['views'] if v['id']==key)
        frame=make_projector(w,h,view['yaw'],view['pitch'],view['fov'])(panorama)
        owned=[s for s in mapping['spaces'] if s['owner']==key]
        polygons=open_editor(frame,[s['corners'] for s in owned],validate_spaces,infer(model,frame),
                             reference_spaces=editor_references(mapping,view))
        if polygons is None:
            continue
        if isinstance(polygons,dict) and 'redraw' in polygons:
            space_id=polygons['redraw']

            def validate_replacement(outlines):
                if len(outlines)!=1:
                    raise ValueError(f'Draw exactly one outline to replace {space_id}.')
                replace_space(mapping,space_id,key,outlines[0])

            replacement=open_editor(frame,[],validate_replacement,infer(model,frame),
                                    reference_spaces=editor_references(mapping,view),
                                    allow_transfer=False,
                                    title=f'UHCL Parking | Redraw {space_id} here: click four corners, then save')
            if replacement is not None:
                mapping=replace_space(mapping,space_id,key,replacement[0])
                config=persist(config,mapping)
                print(f'Replaced outline for {space_id}; same ID retained.',flush=True)
            pending_view=key
            continue
        if isinstance(polygons,dict) and 'transfer' in polygons:
            try:
                mapping=transfer_space(mapping,polygons['transfer'],key)
                config=persist(config,mapping)
                print(f'Moved {polygons["transfer"]} to {key}; same bay ID retained.',flush=True)
            except ValueError as exc:
                print(f'Could not move space: {exc}',flush=True)
            pending_view=key
            continue
        try:
            mapping,skipped=register_spaces(mapping,key,polygons)
        except ValueError as exc:
            print(f'Could not update map: {exc}. Existing map retained.',flush=True)
            continue
        config=persist(config,mapping)
        print(f'Saved {len(mapping["spaces"])} unique mapped bays. Already owned elsewhere: {skipped or "none"}.',flush=True)


def process_sample(mapping,frames,detections,seconds,display_views=None,state_filter=None):
    from occupancy_state import global_observations
    from readable_labels import labels
    raw_observations,assignments=global_observations(mapping,detections)
    observations=state_filter.update(raw_observations) if state_filter else raw_observations
    tiles=[]
    view_reports={}
    # A physical bay is evaluated only in its owner view, even if visible elsewhere.
    for view in mapping['views']:
        key=view['id']
        if key not in frames:
            continue
        owned=[s for s in mapping['spaces'] if s['owner']==key]
        boxes=detections.get(key)
        records=[]
        for i,slot in enumerate(owned):
            status=observations[slot['id']]
            records.append({'id':slot['id'],'status':status,'raw_status':raw_observations[slot['id']]})
        view_reports[key]={'detections':boxes,'owned_bays':records,'vehicle_assignments':assignments}
    owners={v['id']:v for v in mapping['views']}
    shown=display_views if display_views is not None else mapping['views']
    for number,view in enumerate(shown):
        key=view['id']
        if key not in frames:
            continue
        image=frames[key].copy()
        visible=0
        text_labels=[]
        for slot in mapping['spaces']:
            points=reproject(slot['corners'],owners[slot['owner']],view,allow_partial=True)
            if points is None:
                continue
            visible+=1
            status=observations.get(slot['id'],'unknown')
            polygon=(np.asarray(points)*[1024,768]).astype(np.int32)
            color={'occupied':(0,0,255),'free':(0,210,0),'unknown':(0,180,255)}[status]
            cv2.polylines(image,[polygon],True,color,3,cv2.LINE_AA)
            center=polygon.mean(axis=0)*.625
            text_labels.append((slot['id'],center[0],center[1]+34,color))
        for box in detections.get(key) or []:
            x1,y1,x2,y2=map(int,box['box_xyxy'])
            cv2.rectangle(image,(x1,y1),(x2,y2),(255,200,0),2)
        tile=cv2.resize(image,(640,480))
        tile=cv2.copyMakeBorder(tile,34,0,0,0,cv2.BORDER_CONSTANT)
        tile=labels(tile,text_labels,size=22)
        name=f'View {number+1}' if display_views is not None else key
        text=f'{name} | {view["yaw"]:.0f} deg | {visible} visible bays'
        tile=labels(tile,[(text,10,3,(255,255,255))],size=22,background=False)
        tiles.append(tile)
    total=combined_counts(mapping,observations)
    if not tiles:
        raise ValueError('No views available for this sample.')
    columns=4 if display_views is not None else 2
    while len(tiles)%columns:
        tiles.append(np.zeros_like(tiles[0]))
    grid=np.concatenate([np.concatenate(tiles[i:i+columns],axis=1) for i in range(0,len(tiles),columns)],axis=0)
    grid=cv2.copyMakeBorder(grid,90,0,0,0,cv2.BORDER_CONSTANT)
    grid=labels(grid,[
        (f'PRERECORDED DEMO | Time {seconds:.1f}s | {len(shown)} displayed directions',14,0,(255,255,255)),
        (f'Mapped {total["mapped"]} | Occupied {total["occupied"]} | Estimated free {total["estimated_free"]} | Unknown {total["unknown"]}',14,28,(255,255,255)),
        ('Red: occupied | Green: estimated free | Orange: unknown / confirming | Each car assigned once',14,58,(210,210,210))],size=24,background=False)
    return grid,total,view_reports


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lot',choices=['front','back'],default='front')
    parser.add_argument('--source',type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--setup',action='store_true',help='Choose directions and edit their saved bay map.')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--sample-seconds',type=float,default=1)
    parser.add_argument('--max-samples',type=int)
    args=parser.parse_args()
    source=args.source or default_source('first' if args.lot=='front' else 'back')
    config=args.config or Path(f'config/multi-{args.lot}.local.json')
    if args.setup and args.headless:
        parser.error('--setup needs an interactive window.')
    if not source.is_file():
        parser.error(f'Video not found: {source}')
    signature=source_signature(source)
    created=not config.exists()
    try:
        mapping=json.loads(config.read_text()) if not created else initial_map(args.lot,signature)
        validate_map(mapping)
    except (ValueError,KeyError) as exc:
        parser.error(f'Invalid parking map: {exc}')
    if mapping.get('lot')!=args.lot or mapping.get('source_signature')!=signature:
        parser.error('This map belongs to another lot or recording. Choose a new --config path and recalibrate.')
    cap=cv2.VideoCapture(str(source))
    writer=None
    viewer=None
    try:
        fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w,h=int(cap.get(3)),int(cap.get(4))
        if not cap.isOpened() or h<=0 or abs(w/h-2)>.05:
            parser.error('Expected a decodable full 2:1 360 MP4.')
        try:
            indices=list(sample_indices(fps,count,args.sample_seconds,limit=args.max_samples))
        except ValueError as exc:
            parser.error(str(exc))
        weights=Path('models/yolo11n.pt')
        if not weights.exists():
            parser.error('Run detect.py first to obtain the model weights.')
        from ultralytics import YOLO
        model=YOLO(str(weights))
        if args.setup or (created and not args.headless) or not mapping['spaces']:
            if args.headless:
                parser.error('No calibrated bays. Run with --setup first.')
            mapping,config=setup(read_frame(cap,0),mapping,config,model)
            if mapping is None:
                print('Setup closed. Previously saved maps are unchanged.')
                return
        elif created:
            config=persist(config,mapping)
        from view_gallery import gallery_views
        display_views=gallery_views(mapping)
        owners={s['owner'] for s in mapping['spaces']}
        visible_ids={v['id'] for v in display_views}
        views=[v for v in mapping['views'] if v['id'] in owners or v['id'] in visible_ids]
        projectors={v['id']:make_projector(w,h,v['yaw'],v['pitch'],v['fov']) for v in views}
        run=Path(f'outputs/multi-view/{args.lot}/run-{stamp()}');run.mkdir(parents=True)
        movie=run/'combined-prerecorded-demo.mp4'
        processed=0
        from occupancy_state import OccupancyFilter
        state_filter=OccupancyFilter()
        print(f'Processing {len(views)} directions, {len(mapping["spaces"])} mapped bay IDs.',flush=True)
        with (run/'counts.csv').open('w',newline='') as file,(run/'observations.jsonl').open('w') as details:
            csv_writer=csv.DictWriter(file,fieldnames=['video_seconds','mapped','occupied','estimated_free','unknown'])
            csv_writer.writeheader()
            for index in indices:
                if viewer is not None:
                    from tkinter import TclError
                    try:
                        viewer.root.update()
                        if not viewer.root.winfo_exists():
                            break
                    except TclError:
                        break
                panorama=read_frame(cap,index)
                frames={key:project(panorama) for key,project in projectors.items()}
                boxes={key:infer(model,frame) for key,frame in frames.items()}
                grid,total,observations=process_sample(mapping,frames,boxes,index/fps,display_views=display_views,state_filter=state_filter)
                if writer is None:
                    writer=cv2.VideoWriter(str(movie),cv2.VideoWriter_fourcc(*'mp4v'),1/args.sample_seconds,(grid.shape[1],grid.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError('Cannot create combined preview video.')
                writer.write(grid)
                csv_writer.writerow({'video_seconds':round(index/fps,3),**{k:v for k,v in total.items() if k!='spaces'}})
                file.flush()
                details.write(json.dumps({'seconds':index/fps,'counts':total,'views':observations})+'\n');details.flush()
                if processed==0:
                    save_image(run/'first-sample.jpg',grid)
                processed+=1
                if processed%5==0 or processed==len(indices):
                    print(f'{processed}/{len(indices)} samples | {index/fps:.1f}s | {total["occupied"]} occupied, {total["estimated_free"]} estimated free, {total["unknown"]} unknown',flush=True)
                if not args.headless:
                    if viewer is None:
                        from parking_editor import ParkingWindow
                        viewer=ParkingWindow(grid)
                        viewer.root.title('UHCL Parking | combined prerecorded views')
                    else:
                        from PIL import Image
                        viewer.image=Image.fromarray(cv2.cvtColor(grid,cv2.COLOR_BGR2RGB))
                        viewer.render()
            if processed:
                save_image(run/'last-sample.jpg',grid)
        if writer is not None:
            writer.release();writer=None
        summary={'source':str(source.resolve()),'map':str(config.resolve()),'views':views,'display_views':display_views,
                 'samples_processed':processed,'requested_samples':len(indices),'sample_seconds':args.sample_seconds,
                 'mapped_bays':len(mapping['spaces']),'partial':processed<len(indices),
                 'confirmation_samples':{'occupied':3,'free':2},
                 'note':'One owner per mapped bay ID. Panorama overlap check is approximate and requires review. Not full-lot or live availability.'}
        write_artifact(run/'summary.json',json.dumps(summary,indent=2).encode())
        write_artifact(run/'parking-map.local.json',json.dumps(mapping,indent=2).encode())
        print(f'Results saved: {run.resolve()}',flush=True)
        if viewer is not None:
            from tkinter import TclError
            try:
                if viewer.root.winfo_exists():
                    viewer.message.set('Finished. Close to exit. Scroll to zoom or right-drag to pan.')
                    viewer.root.mainloop()
            except TclError:
                pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()


if __name__=='__main__':
    main()
