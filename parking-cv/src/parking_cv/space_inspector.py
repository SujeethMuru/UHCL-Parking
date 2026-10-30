"""Identify spaces by clicking high-resolution views, without crowded ID labels."""
from copy import deepcopy
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image
from parking_editor import ParkingWindow,ViewTransform
from parking_map import reproject,SPACE_TYPES,space_label
from perspective import projection_maps
from view_gallery import gallery_views


def inspect_spaces(panorama,mapping,save_category):
    mapping=deepcopy(mapping)
    views=gallery_views(mapping)
    owners={v['id']:v for v in mapping['views']}
    h,w=panorama.shape[:2]
    window=ParkingWindow(np.zeros((1536,2048,3),np.uint8))
    window.root.title(f'UHCL Parking | {mapping.get("lot","front").title()} lot | Identify and label spaces')
    selected=None
    polygons=[]
    bar=ttk.Frame(window.root,padding=6)
    bar.pack(fill='x',before=window.canvas)
    view_choice=tk.StringVar(value='View 1')
    view_picker=ttk.Combobox(bar,textvariable=view_choice,values=[f'View {i+1}' for i in range(8)],state='readonly',width=9)
    view_picker.pack(side='left',padx=4)
    identity=tk.StringVar(value='Click a parking space in the image')
    ttk.Label(bar,textvariable=identity,font=('Arial',14,'bold')).pack(side='left',padx=8)
    type_choice=tk.StringVar(value='Standard')
    ttk.Combobox(bar,textvariable=type_choice,values=list(SPACE_TYPES.values()),state='readonly',width=24).pack(side='left')
    base_render=window.render

    def render():
        base_render()
        for slot,points in polygons:
            coords=window.screen(points)
            color={'standard':'#d7dce0','ev':'#c77dff','accessible':'#40c9ff','accessible_ev':'#ffb84d'}[slot.get('space_type','standard')]
            chosen=slot['id']==selected
            window.canvas.create_polygon(*[n for p in coords for n in p],fill='',outline='#fff34d' if chosen else color,width=4 if chosen else 2)
            if chosen:
                x,y=np.mean(coords,axis=0)
                x=max(65,min(x,window.canvas.winfo_width()-85));y=max(20,min(y,window.canvas.winfo_height()-25))
                label=window.canvas.create_text(x,y,text=space_label(slot),fill='white',font=('Arial',17,'bold'))
                bounds=window.canvas.bbox(label)
                bg=window.canvas.create_rectangle(bounds[0]-5,bounds[1]-4,bounds[2]+5,bounds[3]+4,fill='#111820',outline='#fff34d',width=2)
                window.canvas.tag_raise(label,bg)

    window.render=render

    def load(event=None):
        nonlocal selected,polygons
        selected=None
        view=views[int(view_choice.get().split()[-1])-1]
        mx,my=projection_maps(w,h,view['yaw'],pitch=view['pitch'],fov=view['fov'],width=2048,height=1536)
        frame=cv2.remap(panorama,mx-1,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP)
        window.image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        window.w,window.h=2048,1536
        window.view=ViewTransform(2048,1536)
        polygons=[]
        for slot in mapping['spaces']:
            points=reproject(slot['corners'],owners[slot['owner']],view,allow_partial=True)
            if points:
                polygons.append((slot,points))
        identity.set('Click a space to identify it')
        window.fit()

    def click(event):
        nonlocal selected
        x,y=window.view.to_image(event.x,event.y)
        point=(x/window.w,y/window.h)
        hits=[(cv2.contourArea(np.asarray(points,np.float32)),slot)
              for slot,points in polygons if cv2.pointPolygonTest(np.asarray(points,np.float32),point,False)>=0]
        if not hits:
            return
        slot=min(hits,key=lambda item:item[0])[1]
        selected=slot['id']
        type_choice.set(SPACE_TYPES[slot.get('space_type','standard')])
        identity.set(f'Selected: {selected}')
        window.message.set(f'{selected}: {type_choice.get()}. Choose a type and Save type. Scroll to zoom; right-drag to pan.')
        render()

    def save():
        nonlocal mapping
        if selected is None:
            window.message.set('Click a parking space first.')
            return
        kind=next(k for k,v in SPACE_TYPES.items() if v==type_choice.get())
        mapping=save_category(selected,kind)
        for slot,_ in polygons:
            if slot['id']==selected:
                slot['space_type']=kind
        window.message.set(f'Saved {selected} as {SPACE_TYPES[kind]}. Click the next space to continue.')
        render()

    ttk.Button(bar,text='Save type',command=save).pack(side='left',padx=5)
    ttk.Button(bar,text='Back to gallery',command=window.root.destroy).pack(side='right')
    window.canvas.bind('<ButtonPress-1>',click)
    view_picker.bind('<<ComboboxSelected>>',load)
    window.message.set('Click a space to reveal its ID. Only the selected ID appears, so labels cannot overlap. Scroll to zoom; right-drag to pan.')
    window.root.after(100,load)
    window.root.mainloop()
