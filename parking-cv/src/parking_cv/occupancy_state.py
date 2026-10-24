"""Shared vehicle assignment across overlapping views and temporal confirmation."""
import cv2
import numpy as np

from parking_map import sphere_mask,overlap_fraction,reproject,cut_off


def global_observations(mapping,detections):
    views={v['id']:v for v in mapping['views']}
    groups=[]
    for key,boxes in detections.items():
        if key not in views:
            continue
        for box in boxes or []:
            x1,y1,x2,y2=box['box_xyxy']
            if x2<=x1 or y2<=y1:
                continue
            polygon=np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])/[1024,768]
            mask=sphere_mask(polygon,views[key])
            # The same car appears in multiple perspective images. Group its
            # angular boxes before assigning any bay, rather than per owner.
            matches=[g for g in groups if any(k!=key and overlap_fraction(mask,m)>=.6 for k,_,m in g)]
            group=[(key,box,mask)]
            for previous in matches:
                group.extend(previous)
            groups=[g for g in groups if not any(g is previous for previous in matches)]
            groups.append(group)
    states={s['id']:('unknown' if cut_off(s['corners']) or detections.get(s['owner']) is None else 'free')
            for s in mapping['spaces']}
    assignments=[]
    for group in groups:
        scores={}
        for key,box,_ in group:
            x1,y1,x2,y2=box['box_xyxy']
            foot=np.array([[x1,y1+.4*(y2-y1)],[x2,y1+.4*(y2-y1)],
                           [x2,y2],[x1,y2]],np.float32)/[1024,768]
            foot=foot.astype(np.float32)
            area=cv2.contourArea(foot)
            anchor=((x1+x2)/2048,y2/768)
            for slot in mapping['spaces']:
                if states[slot['id']]=='unknown':
                    continue
                points=reproject(slot['corners'],views[slot['owner']],views[key],allow_partial=True)
                if points is None:
                    continue
                poly=np.asarray(points,np.float32)
                fraction=cv2.intersectConvexConvex(poly,foot)[0]/max(area,1e-9)
                anchored=cv2.pointPolygonTest(poly,anchor,False)>=0
                if fraction>=.2 or anchored:
                    scores[slot['id']]=max(scores.get(slot['id'],0),fraction+.05*anchored)
        chosen=max(scores,key=scores.get) if scores else None
        if chosen:
            states[chosen]='occupied'
        assignments.append({'bay':chosen,'views':[key for key,_,_ in group]})
    return states,assignments


class OccupancyFilter:
    """Require three occupied / two free samples; unavailable evidence is unknown."""
    def __init__(self,occupied_samples=3,free_samples=2):
        self.occupied_samples=occupied_samples
        self.free_samples=free_samples
        self.history={}

    def update(self,observations):
        result={}
        for key,state in observations.items():
            previous,count,stable=self.history.get(key,(None,0,'unknown'))
            if state=='unknown':
                self.history[key]=(None,0,'unknown')
                result[key]='unknown'
                continue
            count=count+1 if previous==state else 1
            threshold=self.occupied_samples if state=='occupied' else self.free_samples
            if count>=threshold:
                stable=state
            self.history[key]=(state,count,stable)
            result[key]=stable
        return result
