"""Fixed source-anchor crops and class/projection gates, without target relabelling."""
import cv2,numpy as np
from field_full_survey import ROOT,TR,OUT,CONFIG,read,save
from identify_baseline_targets import marker_centres
from visualize_field_baseline import project,draw

def main():
    cv2.setNumThreads(1);cfg=read(CONFIG);folder=OUT/'detection_reviews';folder.mkdir(exist_ok=True);rows=[]
    for row in cfg['clips']:
        c=row['clip'];md=read(OUT/'markers'/f'{c}_markers.json');centers=marker_centres(OUT/'markers'/f'{c}_markers.json')[0]
        if md.get('seed'):
            f=md['seed']['frame'];x0,y0,x1,y1=md['seed']['box'];point=((x0+x1)/2,(y0+y1)/2)
        else:
            f=sorted(centers)[len(centers)//2];point=centers[f]
        im=cv2.imread(str(TR/'data/camera-data'/c/'frames_all'/f'{f:03d}.jpg'));left=max(0,min(im.shape[1]-700,int(point[0]-350)));top=max(0,min(im.shape[0]-460,int(point[1]-230)));tiles=[]
        def tile(pic,label):
            t=np.full((500,700,3),245,np.uint8);t[40:]=pic[top:top+460,left:left+700];cv2.putText(t,label,(8,25),0,.57,(20,20,20),1);return t
        ref=im.copy();cv2.circle(ref,tuple(map(int,point)),5,(255,0,255),2);tiles.append(tile(ref,f'{c} source f{f}; identity anchor'))
        for key in row['models']:
            tag=cfg['models'][key]['tag'];dd=TR/'outputs/object_detection/camera-data'/f'{c}_{tag}';cal=read(dd/'calibration_used.json');K=np.array(cal['K']);M=np.array(cal['lidar2cam']);overlay=im.copy();allmatch=[]
            for b in read(dd/f'{f:03d}_pred.json'):
                uv=project(b,K,M)
                if uv is None:continue
                margin=cv2.pointPolygonTest(cv2.convexHull(uv.astype(np.float32)),tuple(map(float,point)),True)
                if np.linalg.norm(uv.mean(0)-point)<360 or margin>=0:
                    draw(overlay,b,K,M,(0,220,220) if b['class_id']==0 else (0,100,255),f'{b["class_name"]} {b["score"]:.2f}')
                if margin>=0:allmatch.append(dict(class_name=b['class_name'],score=b['score'],margin_px=float(margin)))
            cv2.circle(overlay,tuple(map(int,point)),5,(255,0,255),2);tiles.append(tile(overlay,key+' raw boxes; yellow=car, orange=other'))
            counts=dict(raw_any=0,raw_car=0,road_any=0,road_car=0);classes={}
            for frame,pt in centers.items():
                for stage,src in [('raw',dd),('road',dd.with_name(dd.name+'_road'))]:
                    names=[]
                    for b in read(src/f'{frame:03d}_pred.json'):
                        uv=project(b,K,M)
                        if uv is None:continue
                        if cv2.pointPolygonTest(cv2.convexHull(uv.astype(np.float32)),tuple(map(float,pt)),True)>=0:names.append(b['class_name'])
                    counts[stage+'_any']+=bool(names);counts[stage+'_car']+='car' in names
                    if stage=='raw':
                        for name in set(names):classes[name]=classes.get(name,0)+1
            rows.append(dict(clip=c,model=key,anchor_frame=f,marker_frames=len(centers),anchor_overlapping_boxes=allmatch,raw_overlapping_class_frames=classes,**counts))
        cv2.imwrite(str(folder/f'{c}.jpg'),np.hstack(tiles))
    save(OUT/'projection_gate_diagnostics.json',dict(rows=rows,note='Containment only, no size/uniqueness/motion gate. Large or neighbouring boxes can overlap the marker. Counts are diagnostic proxies, not independently labelled detection recall or class truth. Fixed source-anchor frames used for all model crops.'))

if __name__=='__main__':main()
