"""Experimental painted-line suggestions; user review is required."""
import cv2
import numpy as np


def split_row(corners, count):
    """Split an outer row quadrilateral into perspective-adjusted equal bays.

    Corners go along one long edge, then back along the opposite long edge.
    """
    unit = np.array([[0,0],[1,0],[1,1],[0,1]],np.float32)
    matrix = cv2.getPerspectiveTransform(unit,np.asarray(corners,np.float32))
    cells=[]
    for i in range(count):
        points = np.array([[[i/count,0],[(i+1)/count,0],
                            [(i+1)/count,1],[i/count,1]]],np.float32)
        cells.append(cv2.perspectiveTransform(points,matrix)[0].tolist())
    return cells


def suggest_spaces(frame,detections=None):
    """Pair nearby painted divider segments below the horizon.

    This deliberately returns candidates, never confirmed parking spaces. Faded
    paint, vehicle occlusion and curb lines can yield misses and false proposals.
    """
    h,w=frame.shape[:2]
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # Local contrast helps separate pale paint from the light asphalt.
    contrast=cv2.subtract(gray,cv2.GaussianBlur(gray,(0,0),5))
    mask=((hsv[:,:,1]<65)&(hsv[:,:,2]>180)&(contrast>15)).astype(np.uint8)*255
    mask[:int(h*.43)]=0
    # Car body edges are not painted bay dividers. Exclude detected vehicles.
    for detection in detections or []:
        x1,y1,x2,y2=map(int,detection['box_xyxy'])
        mask[max(0,y1-5):min(h,y2+6),max(0,x1-5):min(w,x2+6)]=0
    lines=cv2.HoughLinesP(mask,1,np.pi/360,threshold=15,
                          minLineLength=max(28,int(w*.035)),maxLineGap=16)
    if lines is None:
        return []
    segments=[]
    for x1,y1,x2,y2 in lines[:,0]:
        p,q=np.array([x1,y1],float),np.array([x2,y2],float)
        if p[1]>q[1]:
            p,q=q,p
        delta=q-p
        if delta[1]<20 or np.linalg.norm(delta)<35:
            continue
        if abs(delta[0])/max(delta[1],1)>5:
            continue
        # Merge duplicate edges/collinear fragments of a painted stripe.
        merged=False
        for k,(a,b) in enumerate(segments):
            direction=b-a
            offset=p-a
            distance=abs(direction[0]*offset[1]-direction[1]*offset[0])/np.linalg.norm(direction)
            cosine=np.dot(direction,delta)/(np.linalg.norm(direction)*np.linalg.norm(delta))
            if distance<7 and cosine>.985:
                ends=[a,b,p,q]
                segments[k]=(min(ends,key=lambda e:e[1]),max(ends,key=lambda e:e[1]))
                merged=True
                break
        if not merged:
            segments.append((p,q))
    proposals=[]
    # Near-parallel segments with an overlapping vertical range define one bay.
    for i,(a,b) in enumerate(segments):
        for c,d in segments[i+1:]:
            top=max(a[1],c[1]); bottom=min(b[1],d[1])
            if bottom-top<max(30,h*.05):
                continue
            da,dc=b-a,d-c
            cosine=np.dot(da,dc)/(np.linalg.norm(da)*np.linalg.norm(dc))
            if cosine<.8:
                continue
            ax=lambda y: a[0]+(y-a[1])*da[0]/da[1]
            cx=lambda y: c[0]+(y-c[1])*dc[0]/dc[1]
            left,right=(ax,cx) if ax(top)<cx(top) else (cx,ax)
            gap=(right(top)-left(top)+right(bottom)-left(bottom))/2
            if not w*.035<gap<w*.4 or not .35<(bottom-top)/gap<6:
                continue
            poly=np.array([[left(top),top],[right(top),top],
                           [right(bottom),bottom],[left(bottom),bottom]],np.float32)
            if cv2.contourArea(poly)<w*h*.0008 or not cv2.isContourConvex(poly):
                continue
            poly/=np.array([w,h],np.float32)
            if (poly<0).any() or (poly>1).any():
                continue
            # Suppress near-duplicate suggestions. Final overlap checks are in the editor.
            duplicate=False
            for other in proposals:
                overlap,_=cv2.intersectConvexConvex(poly,np.asarray(other,np.float32))
                smaller=min(cv2.contourArea(poly),cv2.contourArea(np.asarray(other,np.float32)))
                if overlap/max(smaller,1e-9)>.5:
                    duplicate=True
                    break
            if not duplicate:
                proposals.append(poly.tolist())
    return proposals[:40]
