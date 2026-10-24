"""Choose saved directions in a 360 recording and open their space editor."""
from copy import deepcopy
import tkinter as tk
from tkinter import ttk,messagebox

import cv2
import numpy as np
from PIL import Image

from parking_editor import ParkingWindow
from parking_map import reproject,validate_view,cut_off
from video_occupancy import make_projector


def matching_direction(views,value):
    return next((v for v in views
                 if abs((v['yaw']-value['yaw']+180)%360-180)<.01
                 and abs(v['pitch']-value['pitch'])<.01
                 and abs(v['fov']-value['fov'])<.01),None)


def select_action(panorama,mapping):
    mapping=deepcopy(mapping)
    h,w=panorama.shape[:2]
    current=mapping['views'][0]
    initial=make_projector(w,h,current['yaw'],current['pitch'],current['fov'])(panorama)
    window=ParkingWindow(initial)
    window.root.title('UHCL Parking | 360 directions and shared parking map')
    result=None
    panel=ttk.Frame(window.root,padding=6)
    panel.pack(fill='x',before=window.canvas)
    selected=tk.StringVar(value=current['id'])
    yaw=tk.DoubleVar(value=current['yaw']);pitch=tk.DoubleVar(value=current['pitch']);fov=tk.DoubleVar(value=current['fov'])
    chooser=ttk.Combobox(panel,textvariable=selected,values=[v['id'] for v in mapping['views']],state='readonly',width=12)
    chooser.pack(side='left',padx=3)
    for label,var,low,high in [('Direction',yaw,-360,360),('Tilt',pitch,-80,80),('Field of view',fov,30,120)]:
        ttk.Label(panel,text=label).pack(side='left',padx=3)
        ttk.Spinbox(panel,textvariable=var,from_=low,to=high,increment=5,width=5).pack(side='left')
    controls=ttk.Frame(window.root,padding=6)
    controls.pack(fill='x',before=window.canvas)
    explanation=ttk.Label(window.root,text='Yellow: owned here. Orange: touches edge, counted unknown. Cyan: owned elsewhere. Use another direction for complete bay coverage.',padding=6,wraplength=1100)
    explanation.pack(fill='x',before=window.canvas)

    def angles():
        value={'id':selected.get(),'yaw':yaw.get(),'pitch':pitch.get(),'fov':fov.get()}
        validate_view(value)
        return value

    def preview():
        try:
            view=angles()
        except (ValueError,tk.TclError) as exc:
            messagebox.showerror('Invalid direction',str(exc),parent=window.root)
            return
        frame=make_projector(w,h,view['yaw'],view['pitch'],view['fov'])(panorama)
        views={v['id']:v for v in mapping['views']}
        for slot in mapping['spaces']:
            corners=reproject(slot['corners'],views[slot['owner']],view,allow_partial=True)
            if corners is None:
                continue
            pixels=(np.asarray(corners)*[1024,768]).astype(np.int32)
            color=(0,255,255) if slot['owner']==view['id'] else (255,255,0)
            if slot['owner']==view['id'] and cut_off(slot['corners']):
                color=(0,180,255)
            cv2.polylines(frame,[pixels],True,color,2)
            cv2.putText(frame,slot['id'],tuple(pixels.mean(axis=0).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
        window.image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        window.fit()
        window.message.set(f'{len(mapping["views"])} saved directions | {len(mapping["spaces"])} mapped bay IDs | First frame for setup, not video playback')

    def load(event=None):
        view=next(v for v in mapping['views'] if v['id']==selected.get())
        yaw.set(view['yaw']);pitch.set(view['pitch']);fov.set(view['fov'])
        preview()

    def turn(amount):
        try:
            yaw.set((yaw.get()+amount)%360)
            preview()
        except tk.TclError:
            pass

    def add_direction():
        try:
            value=angles()
        except (ValueError,tk.TclError) as exc:
            messagebox.showerror('Invalid direction',str(exc),parent=window.root)
            return
        existing=matching_direction(mapping['views'],value)
        if existing is not None:
            selected.set(existing['id'])
            load()
            window.message.set(f'This direction already exists as {existing["id"]}. Click Mark spaces to edit it.')
            return
        if len(mapping['views'])>=8:
            messagebox.showinfo('View limit','Eight directions already exist. Select an unused direction from the dropdown, turn to the desired angle, then click Mark spaces. Or use Remove unused direction to free a slot.',parent=window.root)
            return
        counter=1
        while any(v['id']==f'direction-{counter}' for v in mapping['views']):
            counter+=1
        value['id']=f'direction-{counter}'
        mapping['views'].append(value)
        chooser['values']=[v['id'] for v in mapping['views']]
        selected.set(value['id'])
        preview()

    def remove_unused():
        key=selected.get()
        if any(s['owner']==key for s in mapping['spaces']):
            messagebox.showinfo('Direction has marked spaces','This direction contains saved spaces and will be kept. Select a direction without markings to remove it.',parent=window.root)
            return
        if len(mapping['views'])<=1:
            return
        mapping['views']=[v for v in mapping['views'] if v['id']!=key]
        chooser['values']=[v['id'] for v in mapping['views']]
        selected.set(mapping['views'][0]['id'])
        load()

    def mark():
        nonlocal result
        try:
            value=angles()
        except (ValueError,tk.TclError) as exc:
            messagebox.showerror('Invalid direction',str(exc),parent=window.root)
            return
        original=next(v for v in mapping['views'] if v['id']==selected.get())
        if value!=original and any(s['owner']==original['id'] for s in mapping['spaces']):
            messagebox.showinfo('Preserve saved outlines','These angles differ from the saved view. Click Add this direction to keep its existing outlines.',parent=window.root)
            return
        original.update(value)
        result=('mark',mapping,selected.get())
        window.root.destroy()

    def process():
        nonlocal result
        if not mapping['spaces']:
            messagebox.showinfo('Mark bays first','Choose a direction and click Mark spaces before processing.',parent=window.root)
            return
        result=('process',mapping,None)
        window.root.destroy()

    for label,fn in [('Turn left',lambda:turn(-15)),('Turn right',lambda:turn(15)),
                     ('Preview angles',preview),('Add this direction',add_direction),
                     ('Mark spaces',mark),('Save and process all views',process)]:
        ttk.Button(controls,text=label,command=fn).pack(side='left',padx=2)
    ttk.Button(panel,text='Remove unused direction',command=remove_unused).pack(side='left',padx=8)
    chooser.bind('<<ComboboxSelected>>',load)
    window.root.after(100,preview)
    window.root.mainloop()
    return result
