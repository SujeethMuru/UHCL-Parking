"""One catalog of physical bays, each counted by exactly one owner view."""
from copy import deepcopy
import math

import cv2
import numpy as np

from occupancy import validate_spaces

SPACE_TYPES={'standard':'Standard','ev':'EV charging','accessible':'Accessible',
             'accessible_ev':'Accessible EV charging'}


def set_space_type(mapping,space_id,space_type):
    if space_type not in SPACE_TYPES:
        raise ValueError('Choose a supported parking-space type.')
    proposed=deepcopy(mapping)
    slot=next(s for s in proposed['spaces'] if s['id']==space_id)
    slot['space_type']=space_type
    validate_map(proposed)
    return proposed


def space_label(slot):
    suffix={'standard':'','ev':' [EV]','accessible':' [ACC]','accessible_ev':' [ACC+EV]'}
    return slot['id']+suffix[slot.get('space_type','standard')]


def validate_view(view):
    yaw,pitch,fov=(view[k] for k in ('yaw','pitch','fov'))
    if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (yaw,pitch,fov)):
        raise ValueError('View angles must be finite numbers.')
    if not -80<=pitch<=80 or not 30<=fov<=120:
        raise ValueError('Use pitch -80 to 80 and field of view 30 to 120.')


def rays(points,view):
    """Normalized view coordinates to world directions in the 360 panorama."""
    p=np.asarray(points,dtype=float)
    tangent=np.tan(np.deg2rad(view['fov'])/2)
    x=(p[:,0]-.5)*2*tangent
    y=(.5-p[:,1])*2*tangent*.75
    z=np.ones_like(x)
    pitch,yaw=np.deg2rad([view['pitch'],view['yaw']])
    yp=y*np.cos(pitch)+z*np.sin(pitch)
    zp=z*np.cos(pitch)-y*np.sin(pitch)
    result=np.stack((x*np.cos(yaw)+zp*np.sin(yaw),yp,
                     zp*np.cos(yaw)-x*np.sin(yaw)),axis=1)
    return result/np.linalg.norm(result,axis=1,keepdims=True)


def reproject(points,owner,target,allow_partial=False):
    world=rays(points,owner)
    yaw,pitch=np.deg2rad([target['yaw'],target['pitch']])
    x=world[:,0]*np.cos(yaw)-world[:,2]*np.sin(yaw)
    zp=world[:,2]*np.cos(yaw)+world[:,0]*np.sin(yaw)
    y=world[:,1]*np.cos(pitch)-zp*np.sin(pitch)
    z=zp*np.cos(pitch)+world[:,1]*np.sin(pitch)
    if (z<=0).any():
        return None
    tangent=np.tan(np.deg2rad(target['fov'])/2)
    result=np.stack((.5+x/z/(2*tangent),.5-y/z/(1.5*tangent)),axis=1)
    if not allow_partial and ((result<0).any() or (result>1).any()):
        return None
    if allow_partial:
        # Clip against the view instead of hiding the entire bay at an edge.
        result=cv2.intersectConvexConvex(np.asarray(result,np.float32),
                                        np.array([[0,0],[1,0],[1,1],[0,1]],np.float32))[1]
        if result is None:
            return None
        result=result.reshape(-1,2)
    return result.tolist()


def sphere_mask(points,view,width=1024,height=512):
    # Sample straight image edges before projection so curved spherical boundaries
    # and the panorama seam do not turn into a huge false polygon.
    p=np.asarray(points)
    boundary=np.concatenate([p[i]+np.arange(16)[:,None]/16*(p[(i+1)%4]-p[i]) for i in range(4)])
    direction=rays(boundary,view)
    longitude=np.unwrap(np.arctan2(direction[:,0],direction[:,2]))
    latitude=np.arcsin(np.clip(direction[:,1],-1,1))
    pixels=np.stack(((longitude/(2*np.pi)+.5)*width,(.5-latitude/np.pi)*height),axis=1)
    mask=np.zeros((height,width),np.uint8)
    for shift in (-width,0,width):
        poly=np.round(pixels+[shift,0]).astype(np.int32)
        cv2.fillPoly(mask,[poly],1)
    return mask


def overlap_fraction(first,second):
    return np.count_nonzero(first & second)/max(1,min(np.count_nonzero(first),np.count_nonzero(second)))


def validate_map(mapping):
    views={v['id']:v for v in mapping['views']}
    if len(views)!=len(mapping['views']) or not views:
        raise ValueError('View IDs must be unique and at least one direction is required.')
    for view in views.values():
        validate_view(view)
    ids=[s['id'] for s in mapping['spaces']]
    if len(set(ids))!=len(ids):
        raise ValueError('Each physical bay must have one unique ID and one owner.')
    for slot in mapping['spaces']:
        if slot.get('space_type','standard') not in SPACE_TYPES:
            raise ValueError('Invalid parking-space type.')
        if slot['owner'] not in views:
            raise ValueError('Parking space refers to a missing owner view.')
    for key in views:
        polygons=[s['corners'] for s in mapping['spaces'] if s['owner']==key]
        if polygons:
            validate_spaces(polygons)
    masks=[]
    for slot in mapping['spaces']:
        mask=sphere_mask(slot['corners'],views[slot['owner']])
        for previous_id,previous in masks:
            if overlap_fraction(mask,previous)>=.45:
                raise ValueError(f'Bays {previous_id} and {slot["id"]} overlap strongly in the panorama. Review boundaries.')
        masks.append((slot['id'],mask))


def register_spaces(mapping,view_id,polygons):
    """Replace this owner's outlines; skip bays already mapped in another view.

    Angular overlap is a geometric safeguard, not a learned bay identity system.
    Users must still review that distinct IDs correspond to distinct real bays.
    """
    if polygons:
        validate_spaces(polygons)
    proposed=deepcopy(mapping)
    view=next(v for v in mapping['views'] if v['id']==view_id)
    views={v['id']:v for v in mapping['views']}
    old=[s for s in mapping['spaces'] if s['owner']==view_id]
    other=[s for s in mapping['spaces'] if s['owner']!=view_id]
    other_masks=[(s['id'],sphere_mask(s['corners'],views[s['owner']])) for s in other]
    old_masks=[(s,sphere_mask(s['corners'],view)) for s in old]
    skipped=[]
    accepted=[]
    used=set()
    next_id=max([int(s['id'][1:]) for s in mapping['spaces'] if s['id'].startswith('B') and s['id'][1:].isdigit()]+[0])+1
    for polygon in polygons:
        mask=sphere_mask(polygon,view)
        duplicate=next((key for key,m in other_masks if overlap_fraction(mask,m)>=.45),None)
        if duplicate:
            prior=next(s for s in other if s['id']==duplicate)
            if cut_off(prior['corners']) and not cut_off(polygon):
                # A complete outline in an adjacent direction replaces an edge
                # fragment while retaining the same physical bay identity.
                other=[s for s in other if s['id']!=duplicate]
                other_masks=[(key,m) for key,m in other_masks if key!=duplicate]
                accepted.append({**deepcopy(prior),'owner':view_id,'corners':deepcopy(polygon)})
                used.add(duplicate)
            else:
                skipped.append(duplicate)
            continue
        ranked=[(overlap_fraction(mask,m),s) for s,m in old_masks if s['id'] not in used]
        score,prior=max(ranked,key=lambda item:item[0]) if ranked else (0,None)
        key=prior['id'] if score>=.5 else f'B{next_id:03d}'
        if score<.5:
            next_id+=1
        used.add(key)
        metadata=deepcopy(prior) if score>=.5 else {}
        accepted.append({**metadata,'id':key,'owner':view_id,'corners':deepcopy(polygon)})
        accepted[-1].setdefault('space_type','standard')
    proposed['spaces']=other+accepted
    validate_map(proposed)
    return proposed,skipped


def combined_counts(mapping,observations):
    """Count catalog IDs once. Missing/edge-cut observations remain unknown."""
    states={slot['id']:observations.get(slot['id'],'unknown') for slot in mapping['spaces']}
    if len(states)!=len(mapping['spaces']):
        raise ValueError('Duplicate physical bay IDs.')
    if any(s not in {'free','occupied','unknown'} for s in states.values()):
        raise ValueError('Invalid occupancy state.')
    by_type={}
    for kind in SPACE_TYPES:
        members=[s for s in mapping['spaces'] if s.get('space_type','standard')==kind]
        by_type[kind]={'mapped':len(members),'occupied':sum(states[s['id']]=='occupied' for s in members),
                       'estimated_free':sum(states[s['id']]=='free' for s in members),
                       'unknown':sum(states[s['id']]=='unknown' for s in members)}
    return {'by_type':by_type,'mapped':len(states),'occupied':sum(s=='occupied' for s in states.values()),
            'estimated_free':sum(s=='free' for s in states.values()),
            'unknown':sum(s=='unknown' for s in states.values()),'spaces':states}


def transfer_space(mapping,space_id,target_id):
    """Move one existing bay to a fully visible view, preserving its global ID."""
    proposed=deepcopy(mapping)
    views={v['id']:v for v in proposed['views']}
    slot=next(s for s in proposed['spaces'] if s['id']==space_id)
    points=reproject(slot['corners'],views[slot['owner']],views[target_id])
    if points is None or cut_off(points):
        raise ValueError('Choose a view showing the whole outline before moving it.')
    slot.update(owner=target_id,corners=points)
    validate_map(proposed)
    return proposed


def replace_space(mapping,space_id,target_id,corners):
    """Explicit user redraw replaces exactly one bay; all other IDs are retained."""
    validate_spaces([corners])
    proposed=deepcopy(mapping)
    slot=next(s for s in proposed['spaces'] if s['id']==space_id)
    slot.update(owner=target_id,corners=deepcopy(corners))
    validate_map(proposed)
    return proposed


def focused_view(mapping,space_id):
    """Center a temporary repair view on the bay's world direction."""
    slot=next(s for s in mapping['spaces'] if s['id']==space_id)
    owner=next(v for v in mapping['views'] if v['id']==slot['owner'])
    direction=rays(slot['corners'],owner).mean(axis=0)
    direction/=np.linalg.norm(direction)
    key=f'repair-{space_id}'
    while any(v['id']==key for v in mapping['views']):
        key+='-new'
    return {'id':key,'yaw':float(np.degrees(np.arctan2(direction[0],direction[2]))%360),
            'pitch':float(np.clip(np.degrees(np.arcsin(direction[1])),-80,80)),
            'fov':110}


def cut_off(corners,margin=.005):
    p=np.asarray(corners)
    return bool((p<=margin).any() or (p>=1-margin).any())
