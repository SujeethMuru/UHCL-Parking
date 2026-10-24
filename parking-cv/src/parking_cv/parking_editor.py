"""Resizable parking editor and result viewer with cursor-centered zoom and pan."""
from copy import deepcopy
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from space_suggestions import suggest_spaces, split_row


class ViewTransform:
    def __init__(self,width,height):
        self.width,self.height=width,height
        self.scale=1.
        self.x=self.y=0.

    def fit(self,cw,ch):
        self.scale=min(cw/self.width,ch/self.height)*.97
        self.x=(cw-self.width*self.scale)/2
        self.y=(ch-self.height*self.scale)/2

    def to_image(self,x,y):
        return ((x-self.x)/self.scale,(y-self.y)/self.scale)

    def zoom(self,factor,x,y):
        ix,iy=self.to_image(x,y)
        self.scale=max(.05,min(12,self.scale*factor))
        self.x=x-ix*self.scale
        self.y=y-iy*self.scale


class ParkingWindow:
    def __init__(self,frame,spaces=None,validate=None,detections=None,reference_spaces=None):
        self.root=tk.Tk()
        self.root.title('UHCL Parking — space editor' if validate else 'UHCL Parking — results')
        sw,sh=self.root.winfo_screenwidth(),self.root.winfo_screenheight()
        self.root.geometry(f'{min(1250,int(sw*.88))}x{min(900,int(sh*.84))}+30+30')
        self.image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        self.frame=frame
        self.h,self.w=frame.shape[:2]
        self.view=ViewTransform(self.w,self.h)
        self.spaces=deepcopy(spaces or [])
        self.reference_spaces=deepcopy(reference_spaces or [])
        self.validate=validate
        self.detections=detections or []
        self.result=None
        self.history=[]
        self.pending=[]
        self.proposals=[]
        self.selected=None
        self.mode='draw'
        self.drag_vertex=None
        self.pan_origin=None
        self.fitted=False
        self.at_fit=True
        bar=ttk.Frame(self.root,padding=6)
        bar.pack(fill='x')
        for label,fn in [('Fit entire image',self.fit),('Zoom +',lambda:self.zoom(1.3)),
                         ('Zoom −',lambda:self.zoom(1/1.3))]:
            ttk.Button(bar,text=label,command=fn).pack(side='left',padx=2)
        if validate:
            ttk.Button(bar,text='Save + detect',command=self.save).pack(side='right')
            controls=ttk.Frame(self.root,padding=(6,0,6,6))
            controls.pack(fill='x')
            for label,fn in [('Draw space',lambda:self.set_mode('draw')),
                             ('Adjust corners',lambda:self.set_mode('adjust')),
                             ('Suggest spaces',self.suggest),
                             ('Draw row',lambda:self.set_mode('row'))]:
                ttk.Button(controls,text=label,command=fn).pack(side='left',padx=2)
            self.row_count=tk.IntVar(value=5)
            ttk.Spinbox(controls,from_=2,to=30,textvariable=self.row_count,width=3).pack(side='left')
            ttk.Label(controls,text='bays in row').pack(side='left',padx=4)
            ttk.Button(controls,text='Undo',command=self.undo).pack(side='left',padx=2)
            ttk.Button(controls,text='Delete selected',command=self.delete).pack(side='left',padx=2)
        self.message=tk.StringVar(value='Scroll: zoom | Right-drag: pan | Fit: show entire image')
        ttk.Label(self.root,textvariable=self.message,padding=6,wraplength=1000).pack(side='bottom',fill='x')
        self.canvas=tk.Canvas(self.root,bg='#15181d',highlightthickness=0)
        self.canvas.pack(fill='both',expand=True)
        self.canvas.bind('<Configure>',self.resize)
        self.canvas.bind('<MouseWheel>',self.wheel)
        self.canvas.bind('<Button-4>',lambda e:self.wheel(e,1))
        self.canvas.bind('<Button-5>',lambda e:self.wheel(e,-1))
        self.canvas.bind('<ButtonPress-3>',self.pan_start)
        self.canvas.bind('<B3-Motion>',self.pan)
        self.canvas.bind('<ButtonRelease-3>',lambda e:setattr(self,'pan_origin',None))
        self.canvas.bind('<ButtonPress-1>',self.click)
        self.canvas.bind('<B1-Motion>',self.move_vertex)
        self.canvas.bind('<ButtonRelease-1>',self.release_vertex)
        self.root.bind('<KeyPress-f>',lambda e:self.fit())
        self.root.bind('<KeyPress-u>',lambda e:self.undo())
        self.root.bind('<Delete>',lambda e:self.delete())
        self.root.bind('<Escape>',lambda e:self.cancel_pending())
        if validate:
            self.root.bind('<KeyPress-s>',lambda e:self.save())
        self.root.protocol('WM_DELETE_WINDOW',self.root.destroy)

    def resize(self,event):
        if not self.fitted or self.at_fit:
            self.view.fit(event.width,event.height)
            self.fitted=True
        else:
            # Keep the current center during resize; Fit restores the complete frame.
            self.view.x+=(event.width-getattr(self,'cw',event.width))/2
            self.view.y+=(event.height-getattr(self,'ch',event.height))/2
        self.cw,self.ch=event.width,event.height
        self.render()

    def fit(self):
        self.at_fit=True
        self.view.fit(self.canvas.winfo_width(),self.canvas.winfo_height())
        self.render()

    def zoom(self,factor,x=None,y=None):
        self.at_fit=False
        self.view.zoom(factor,x if x is not None else self.canvas.winfo_width()/2,
                       y if y is not None else self.canvas.winfo_height()/2)
        self.render()

    def wheel(self,event,direction=None):
        self.zoom(1.2 if (direction or event.delta)>0 else 1/1.2,event.x,event.y)

    def pan_start(self,event):
        self.pan_origin=(event.x,event.y,self.view.x,self.view.y)

    def pan(self,event):
        if self.pan_origin:
            self.at_fit=False
            x,y,ox,oy=self.pan_origin
            self.view.x=ox+event.x-x
            self.view.y=oy+event.y-y
            self.render()

    def screen(self,points):
        return [(p[0]*self.w*self.view.scale+self.view.x,
                 p[1]*self.h*self.view.scale+self.view.y) for p in points]

    def render(self):
        cw,ch=max(1,self.canvas.winfo_width()),max(1,self.canvas.winfo_height())
        v=self.view
        # Render only the visible viewport, even at high zoom levels.
        visible=self.image.transform((cw,ch),Image.Transform.AFFINE,
                    (1/v.scale,0,-v.x/v.scale,0,1/v.scale,-v.y/v.scale),
                    Image.Resampling.BILINEAR,fillcolor='#15181d')
        self.photo=ImageTk.PhotoImage(visible)
        self.canvas.delete('all')
        self.canvas.create_image(0,0,image=self.photo,anchor='nw')
        for reference in self.reference_spaces:
            if reference['corners'] is None:
                continue
            coords=self.screen(reference['corners'])
            self.canvas.create_polygon(*[n for xy in coords for n in xy],
                                       fill='',outline='#00dfff',dash=(6,3),width=2)
            cx,cy=np.mean(coords,axis=0)
            self.canvas.create_text(cx,cy,text=reference['label'],fill='#00dfff',font=('Arial',10,'bold'))
        for polygon in self.proposals:
            coords=self.screen(polygon)
            self.canvas.create_polygon(*[n for xy in coords for n in xy],
                                       fill='',outline='#ffad33',dash=(6,4),width=2)
        for i,polygon in enumerate(self.spaces):
            coords=self.screen(polygon)
            color='#00e5ff' if i==self.selected else '#fff34d'
            self.canvas.create_polygon(*[n for xy in coords for n in xy],fill='',outline=color,width=2)
            cx,cy=np.mean(coords,axis=0)
            self.canvas.create_text(cx,cy,text=f'P{i+1}',fill=color,font=('Arial',12,'bold'))
            if self.mode=='adjust' and i==self.selected:
                for x,y in coords:
                    self.canvas.create_oval(x-5,y-5,x+5,y+5,fill=color)
        if self.pending:
            coords=self.screen(self.pending)
            if len(coords)>1:
                self.canvas.create_line(*[n for xy in coords for n in xy],fill='#ff55df',width=2)
            for x,y in coords:
                self.canvas.create_oval(x-4,y-4,x+4,y+4,fill='#ff55df')

    def set_mode(self,mode):
        self.mode=mode
        self.pending=[]
        self.message.set({'draw':'Click four corners around one bay. Scroll to zoom; right-drag to pan.',
                          'adjust':'Click a space, then drag its corner handles. Delete removes selected space.',
                          'row':'Click row outer corners: along its long edge, then back along the opposite edge.'}[mode])
        self.render()

    def checkpoint(self):
        self.history.append(deepcopy(self.spaces))

    def add(self,polygons):
        try:
            self.validate(self.spaces+polygons)
        except ValueError as exc:
            self.message.set(str(exc))
            return False
        self.checkpoint()
        self.spaces.extend(deepcopy(polygons))
        self.selected=len(self.spaces)-1
        return True

    def click(self,event):
        self.canvas.focus_set()
        if not self.validate:
            return
        ix,iy=self.view.to_image(event.x,event.y)
        if not (0<=ix<self.w and 0<=iy<self.h):
            return
        point=(ix/self.w,iy/self.h)
        for i,polygon in enumerate(self.proposals):
            if cv2.pointPolygonTest(np.asarray(polygon,np.float32),point,False)>=0:
                if self.add([polygon]):
                    self.proposals.pop(i)
                    self.message.set('Suggestion accepted. Adjust corners if needed; Save + detect when ready.')
                self.render()
                return
        if self.mode=='adjust':
            if self.selected is not None:
                for j,(x,y) in enumerate(self.screen(self.spaces[self.selected])):
                    if (x-event.x)**2+(y-event.y)**2<=100:
                        self.checkpoint()
                        self.drag_vertex=(self.selected,j)
                        return
            self.selected=next((i for i,p in enumerate(self.spaces)
                               if cv2.pointPolygonTest(np.asarray(p,np.float32),point,False)>=0),None)
            self.render()
            return
        self.pending.append(list(point))
        if len(self.pending)==4:
            if self.mode=='row':
                try:
                    count=int(self.row_count.get())
                    if not 2<=count<=30:
                        raise ValueError('Row count must be 2 to 30.')
                    self.validate([self.pending])
                    cells=split_row(self.pending,count)
                    if self.add(cells):
                        self.message.set(f'Added {count} bays. Adjust corners to the painted lines.')
                except (ValueError,tk.TclError) as exc:
                    self.message.set(str(exc))
            elif self.add([self.pending]):
                self.message.set(f'{len(self.spaces)} spaces marked. Keep drawing or Save + detect.')
            self.pending=[]
        self.render()

    def move_vertex(self,event):
        if self.drag_vertex:
            i,j=self.drag_vertex
            x,y=self.view.to_image(event.x,event.y)
            self.spaces[i][j]=[max(0,min(1,x/self.w)),max(0,min(1,y/self.h))]
            self.render()

    def release_vertex(self,event):
        if self.drag_vertex:
            try:
                self.validate(self.spaces)
            except ValueError as exc:
                self.spaces=self.history.pop()
                self.message.set(f'Move rejected: {exc}')
            self.drag_vertex=None
            self.render()

    def suggest(self):
        self.proposals=[]
        for proposal in suggest_spaces(self.frame,self.detections):
            try:
                self.validate(self.spaces+[proposal])
                self.proposals.append(proposal)
            except ValueError:
                continue
        self.pending=[]
        self.message.set(f'{len(self.proposals)} experimental suggestions (orange). Click to accept; they are not counted until accepted. Use Draw row for missed bays.')
        self.render()

    def cancel_pending(self):
        self.pending=[]
        self.proposals=[]
        self.render()

    def undo(self):
        if self.pending:
            self.pending.pop()
        elif self.history:
            self.spaces=self.history.pop()
            self.selected=None
        self.render()

    def delete(self):
        if self.validate and self.selected is not None:
            self.checkpoint()
            self.spaces.pop(self.selected)
            self.selected=None
            self.render()

    def save(self):
        if not self.validate:
            return
        if self.pending:
            self.message.set('Finish the pending outline or press Escape first.')
            return
        try:
            self.validate(self.spaces)
        except ValueError as exc:
            self.message.set(str(exc))
            return
        self.result=deepcopy(self.spaces)
        self.root.destroy()


def open_editor(frame,spaces,validate,detections=None,reference_spaces=None,allow_transfer=True,title=None):
    window=ParkingWindow(frame,spaces,validate,detections,reference_spaces)
    if title:
        window.root.title(title)
    if reference_spaces:
        window.message.set('Yellow: editable here. Cyan: saved elsewhere. Choose a cyan space below and Move editing here to adjust it in this view without making a duplicate.')
        transferable=[r for r in reference_spaces if r.get('id')] if allow_transfer else []
        if transferable:
            choice=tk.StringVar(value=transferable[0]['label'])
            row=ttk.Frame(window.root,padding=5)
            row.pack(fill='x',before=window.canvas)
            ttk.Label(row,text='Edit an existing space here:').pack(side='left')
            ttk.Combobox(row,textvariable=choice,values=[r['label'] for r in transferable],
                         state='readonly',width=24).pack(side='left',padx=4)

            def transfer():
                if window.spaces!=list(spaces) or window.pending:
                    window.message.set('Save your current edits first, then reopen this view to move a cyan space.')
                    return
                reference=next(r for r in transferable if r['label']==choice.get())
                operation='transfer' if reference.get('transferable') else 'redraw'
                window.result={operation:reference['id']}
                window.root.destroy()

            ttk.Button(row,text='Edit here / redraw',command=transfer).pack(side='left')
            ttk.Label(row,text='Keeps its ID. Incomplete outlines: draw four new corners.').pack(side='left',padx=6)
    window.root.mainloop()
    return window.result


def show_result(frame):
    window=ParkingWindow(frame)
    window.root.mainloop()
