"""Eight fixed editing views and a shared, annotated full-panorama overview."""
from copy import deepcopy
from pathlib import Path
from tkinter import ttk, messagebox

import cv2
import numpy as np

from local_storage import save_image
from parking_editor import ParkingWindow
from parking_map import reproject, sphere_mask
from video_occupancy import make_projector


def gallery_views(mapping):
    """Add fixed directions without deleting or moving any existing calibration."""
    selected=[]
    for i in range(8):
        yaw=(300+45*i)%360
        match=next((v for v in mapping['views']
                    if abs((v['yaw']-yaw+180)%360-180)<.01
                    and abs(v['pitch']+20)<.01 and abs(v['fov']-90)<.01),None)
        if match is None:
            key=f'gallery-{i+1}'
            while any(v['id']==key for v in mapping['views']):
                key+='-new'
            match={'id':key,'yaw':yaw,'pitch':-20,'fov':90}
            mapping['views'].append(match)
        selected.append(match)
    return selected


def draw_bays(frame,mapping,target):
    owners={v['id']:v for v in mapping['views']}
    for slot in mapping['spaces']:
        points=reproject(slot['corners'],owners[slot['owner']],target,allow_partial=True)
        if points is None:
            continue
        pixels=(np.asarray(points)*[1024,768]).astype(np.int32)
        color=(0,255,255) if slot['owner']==target['id'] else (255,255,0)
        cv2.polylines(frame,[pixels],True,color,2)
        cv2.putText(frame,slot['id'],tuple(pixels.mean(axis=0).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
    return frame


def editor_references(mapping,target):
    """Read-only projections, never included in editable polygons or counts."""
    owners={v['id']:v for v in mapping['views']}
    fixed=gallery_views(deepcopy(mapping))
    names={v['id']:f'View {i+1}' for i,v in enumerate(fixed)}
    references=[]
    for slot in mapping['spaces']:
        if slot['owner']==target['id']:
            continue
        points=reproject(slot['corners'],owners[slot['owner']],target,allow_partial=True)
        name=names.get(slot['owner'],slot['owner'])
        full=reproject(slot['corners'],owners[slot['owner']],target)
        from parking_map import cut_off
        transferable=full is not None and not cut_off(full)
        suffix='' if transferable else ' (redraw)'
        references.append({'corners':points,'label':f'{slot["id"]} - {name}{suffix}',
                           'id':slot['id'],'transferable':transferable})
    return references


def panorama_overlay(panorama,mapping):
    image=cv2.resize(panorama,(1280,640))
    owners={v['id']:v for v in mapping['views']}
    for slot in mapping['spaces']:
        mask=sphere_mask(slot['corners'],owners[slot['owner']],1280,640)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image,contours,-1,(0,255,255),2)
        # A seam-crossing bay may have two visible pieces; label both.
        for contour in contours:
            x,y,w,h=cv2.boundingRect(contour)
            cv2.putText(image,slot['id'],(x+w//2,y+h//2),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1)
    return image


def build_gallery(panorama,mapping,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    views=gallery_views(mapping)
    h,w=panorama.shape[:2]
    tiles=[]
    for i,view in enumerate(views):
        clean=make_projector(w,h,view['yaw'],view['pitch'],view['fov'])(panorama)
        save_image(output/f'view-{i+1}-reference.jpg',clean)
        annotated=draw_bays(clean.copy(),mapping,view)
        save_image(output/f'view-{i+1}-marked.jpg',annotated)
        tile=cv2.resize(annotated,(320,240))
        cv2.rectangle(tile,(0,0),(320,28),(0,0,0),-1)
        cv2.putText(tile,f'View {i+1} | {view["yaw"]:.0f} degrees',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
        tiles.append(tile)
    grid=np.concatenate([np.concatenate(tiles[:4],axis=1),np.concatenate(tiles[4:],axis=1)],axis=0)
    overview=panorama_overlay(panorama,mapping)
    save_image(output/'360-marked-overview.jpg',overview)
    bar=np.zeros((32,1280,3),np.uint8)
    cv2.putText(bar,'Full 360 panorama | All saved space IDs | Yellow/cyan outlines are calibration, not occupancy',(10,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
    atlas=np.concatenate((grid,bar,overview),axis=0)
    save_image(output/'eight-views-and-360.jpg',atlas)
    return views,atlas


def select_action(panorama,mapping):
    mapping=deepcopy(mapping)
    output=Path('outputs/view-gallery')/mapping.get('lot','front')
    views,atlas=build_gallery(panorama,mapping,output)
    window=ParkingWindow(atlas)
    window.root.title('UHCL Parking | Eight views and full 360 overview')
    result=None
    panel=ttk.Frame(window.root,padding=4)
    panel.pack(fill='x',before=window.canvas)

    def edit(view):
        nonlocal result
        result=('mark',mapping,view['id'])
        window.root.destroy()

    for i,view in enumerate(views):
        ttk.Button(panel,text=f'Mark view {i+1}',command=lambda v=view:edit(v)).grid(row=i//4,column=i%4,padx=3,pady=2,sticky='ew')
    for col in range(4):
        panel.columnconfigure(col,weight=1)

    def process():
        nonlocal result
        if not mapping['spaces']:
            messagebox.showinfo('Mark a space first','Choose Mark view 1 through 8 and save at least one parking space.',parent=window.root)
            return
        result=('process',mapping,None)
        window.root.destroy()

    # Keep earlier hand-marked directions editable even when they are not one
    # of the eight fixed angles. Their markings remain visible in the overview.
    extras=[v for v in mapping['views'] if v not in views and any(s['owner']==v['id'] for s in mapping['spaces'])]
    if extras:
        from tkinter import StringVar
        choice=StringVar(value=extras[0]['id'])
        row=ttk.Frame(window.root,padding=4);row.pack(fill='x',before=window.canvas)
        ttk.Label(row,text='Earlier markings:').pack(side='left')
        ttk.Combobox(row,textvariable=choice,values=[v['id'] for v in extras],state='readonly',width=15).pack(side='left')
        ttk.Button(row,text='Edit earlier markings',command=lambda:edit(next(v for v in extras if v['id']==choice.get()))).pack(side='left',padx=4)
    ttk.Button(panel,text='Save and process all views',command=process).grid(row=2,column=0,columnspan=4,pady=4)

    def clicked(event):
        x,y=window.view.to_image(event.x,event.y)
        if 0<=x<1280 and 0<=y<480:
            edit(views[int(y//240)*4+int(x//320)])

    window.canvas.bind('<ButtonPress-1>',clicked)
    window.message.set('Click a thumbnail or Mark view 1–8. Save + detect returns here. Below: all markings on the full 360 image. Scroll to zoom; right-drag to pan.')
    window.root.mainloop()
    return result
