"""Antialiased TrueType text drawn at final output resolution."""
from functools import lru_cache
from pathlib import Path
import cv2
from PIL import Image,ImageDraw,ImageFont


@lru_cache(maxsize=8)
def font(size):
    for name in ('C:/Windows/Fonts/segoeuib.ttf','DejaVuSans-Bold.ttf'):
        try:
            return ImageFont.truetype(name,size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def labels(image,items,size=22,background=True):
    canvas=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    draw=ImageDraw.Draw(canvas)
    face=font(size)
    placed=[]
    for text,x,y,color in items:
        x=max(3,min(int(x),image.shape[1]-int(draw.textlength(text,font=face))-6))
        y=max(2,min(int(y),image.shape[0]-size-8))
        rgb=tuple(reversed(color))
        if background:
            origin=(x,y)
            chosen=None
            for offset in (0,size+7,-size-7,2*(size+7),-2*(size+7),3*(size+7)):
                candidate_y=max(34,min(y+offset,image.shape[0]-size-8))
                bounds=draw.textbbox((x,candidate_y),text,font=face)
                rect=(bounds[0]-3,bounds[1]-2,bounds[2]+3,bounds[3]+2)
                if not any(rect[0]<r[2]+2 and rect[2]>r[0]-2 and rect[1]<r[3]+2 and rect[3]>r[1]-2 for r in placed):
                    chosen=(candidate_y,rect)
                    break
            if chosen is None:
                continue
            y,rect=chosen
            placed.append(rect)
            if abs(y-origin[1])>size:
                draw.line((origin[0],origin[1],x,y+size//2),fill=rgb,width=1)
            draw.rectangle(rect,fill=(12,15,18),outline=rgb)
        draw.text((x,y),text,font=face,fill=(255,255,255) if background else rgb)
    import numpy as np
    return cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2BGR)
