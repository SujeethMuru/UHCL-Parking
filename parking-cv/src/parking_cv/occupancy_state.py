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
            # Never collapse two detections from the same view into one car,
            # including indirectly through a larger box in another view.
            eligible=[(max((overlap_fraction(mask,m) for _,_,m in g),default=0),g)
                      for g in groups if all(k!=key for k,_,_ in g)]
            best=max(eligible,key=lambda item:item[0]) if eligible else (0,None)
            matches=[best[1]] if best[0]>=.6 else []
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
        votes={}
        for key,box,_ in group:
            x1,y1,x2,y2=box['box_xyxy']
            # The tall body projects over the bay behind the wheels. Use the
            # bottom fifth near ground contact, rather than the lower 60%.
            foot=np.array([[x1,y1+.8*(y2-y1)],[x2,y1+.8*(y2-y1)],
                           [x2,y2],[x1,y2]],np.float32)/[1024,768]
            foot=foot.astype(np.float32)
            area=cv2.contourArea(foot)
            anchor=((x1+x2)/2048,y2/768)
            local_scores={}
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
                    local_scores[slot['id']]=fraction+.1*anchored
            # A clipped box's lower edge is unreliable. Let intact views vote
            # on the bay instead of allowing one distorted view's maximum win.
            weight=float(box.get('confidence',1))*(.2 if x1<=2 or y1<=2 or x2>=1022 or y2>=766 else 1)
            for bay,score in local_scores.items():
                scores[bay]=scores.get(bay,0)+weight*score
            if local_scores:
                winner=max(local_scores,key=local_scores.get)
                votes[winner]=votes.get(winner,0)+weight
        chosen=max(votes,key=lambda bay:(votes[bay],scores[bay])) if votes else None
        if chosen:
            states[chosen]='occupied'
        assignments.append({'bay':chosen,'views':[key for key,_,_ in group],'bay_votes':votes})
    # A foreground vehicle can hide a previously parked car. No detection is
    # not evidence of an empty bay when its ground region is obscured.
    vehicle_masks=[mask for group in groups for _,_,mask in group]
    for slot in mapping['spaces']:
        if states[slot['id']]!='free':
            continue
        bay=sphere_mask(slot['corners'],views[slot['owner']])
        area=max(1,np.count_nonzero(bay))
        if any(np.count_nonzero(bay & mask)/area>=.45 for mask in vehicle_masks):
            states[slot['id']]='unknown'
    return states,assignments


class OccupancyFilter:
    """Require three occupied / two free samples; unavailable evidence is unknown."""
    def __init__(self,occupied_samples=3,free_samples=2,occlusion_hold_samples=8):
        self.occupied_samples=occupied_samples
        self.free_samples=free_samples
        self.history={}
        self.occlusion_hold_samples=occlusion_hold_samples
        self.uncertain={}

    def update(self,observations):
        result={}
        for key,state in observations.items():
            previous,count,stable=self.history.get(key,(None,0,'unknown'))
            if state=='unknown':
                self.uncertain[key]=self.uncertain.get(key,0)+1
                held='occupied' if stable=='occupied' and self.uncertain[key]<=self.occlusion_hold_samples else 'unknown'
                self.history[key]=(None,0,held)
                result[key]=held
                continue
            self.uncertain[key]=0
            count=count+1 if previous==state else 1
            threshold=self.occupied_samples if state=='occupied' else self.free_samples
            if count>=threshold:
                stable=state
            self.history[key]=(state,count,stable)
            result[key]=stable
        return result
