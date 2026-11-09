"""Contact sheets of observed Todd targets; no identity or geometry edits."""
import cv2
import numpy as np
import pandas as pd
import todd_experiment_data as D
from visualize_field_baseline import draw,project


def main():
    records={r['clip']:r for r in D.Q.read(D.ROOT/'experiments/field_review_applied_20261102/decisions_source.json')['records']}
    dest=D.OUT/'image_review';dest.mkdir(exist_ok=True)
    manifest={};pages=[];rows=[]
    for clip in D.CLIPS:
        r=records[clip];folder=D.SOURCE/'clips'/clip;track=D.Q.csv(folder/'target_trajectory.csv')
        calpath=D.ROOT/'trajectory/outputs/object_detection/camera-data'/f'{clip}_{r["tag"]}'/'calibration_used.json'
        dims=D.Q.read(folder/'dimensions.json');cal=D.Q.read(calpath);K=np.array(cal['K']);M=np.array(cal['lidar2cam'])
        for p in [calpath,folder/'dimensions.json']:manifest[str(p.relative_to(D.ROOT))]=D.Q.sha(p)
        tiles=[]
        for idx in [len(track)//2,len(track)-1]:
            state=track.iloc[idx].to_dict();frame=int(state['frame']);p=D.ROOT/'trajectory/data/camera-data'/clip/'frames_all'/f'{frame:03d}.jpg'
            manifest[str(p.relative_to(D.ROOT))]=D.Q.sha(p);im=cv2.imread(str(p));assert im is not None
            dim=dims[str(state['track_id'])];b=dict(state,l=dim['length_m'],w=dim['width_m'],h=dim['height_m'])
            draw(im,b,K,M,(0,220,255),'SELF '+str(state['track_id']));uv=project(b,K,M)
            center=np.mean(uv,axis=0) if uv is not None else np.array([im.shape[1]/2,im.shape[0]/2])
            x=int(np.clip(center[0]-320,0,im.shape[1]-640));y=int(np.clip(center[1]-180,0,im.shape[0]-360))
            tile=im[y:y+360,x:x+640].copy();tile[:28]=0
            cv2.putText(tile,f'{clip} f{frame} t={state["t_s"]:.2f}s',(8,20),0,.55,(255,255,255),1)
            tiles.append(tile)
        panel=np.hstack(tiles);cv2.imwrite(str(dest/f'{clip}.jpg'),panel);rows.append(panel)
        if len(rows)==4 or clip==D.CLIPS[-1]:
            name=f'contact_{len(pages)+1:02d}.jpg';cv2.imwrite(str(dest/name),np.vstack(rows));pages.append(name);rows=[]
    D.Q.save(dest/'input_manifest.json',manifest)
    D.Q.save(dest/'review.json',dict(cases=13,frames_per_case=2,selection='Midpoint observed row and last observed row; not selected by classification error',
        image_review='pending',pages=pages,note='Sparse visual check of existing reviewed exports; not dense ground-truth detection or geometry validation'))
    print('Saved',len(pages),'contact sheets for 13 Todd recordings')


if __name__=='__main__':main()
