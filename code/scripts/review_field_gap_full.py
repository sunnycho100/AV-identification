"""Source-image review of all target-repaired recordings; no data edits."""
import cv2
import numpy as np
import pandas as pd
from run_field_gap_full import OUT
from field_full_survey import ROOT, TR, CONFIG, read, save
from visualize_field_baseline import draw, project


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cfg=read(CONFIG); audit=pd.read_csv(OUT/'target_gap_audit.csv')
    gaps=audit[(audit.variant=='linear_1p0s') & audit.accepted]
    folder=OUT/'source_reviews';folder.mkdir(exist_ok=True)
    figure_folder=OUT/'figures';figure_folder.mkdir(exist_ok=True)
    selected=[]
    # Largest bounded internal hole in each repaired recording, independent of label.
    for clip,gg in gaps.groupby('clip',sort=True):
        gap=gg.sort_values(['anchor_gap_s','left_frame'],ascending=[False,True]).iloc[0]
        tag=cfg['models'][gap.model]['tag'];src=ROOT/'experiments/field_full_survey_20261101/pairs'/f'{clip}_{tag}'
        result=read(OUT/'runs/linear_1p0s'/clip/'result.json')
        old=pd.read_csv(src/'target_trajectory.csv');new=pd.read_csv(OUT/'runs/linear_1p0s'/clip/'target_trajectory.csv')
        dims=read(src/'dimensions.json'); evaluation=read(ROOT/'experiments/field_full_survey_20261101/runs'/f'{clip}_{gap.model}'/'evaluation.json')
        dim=dims[str(evaluation['identified_target'])]
        cal=read(TR/'outputs/object_detection/camera-data'/f'{clip}_{tag}'/'calibration_used.json');K=np.array(cal['K']);M=np.array(cal['lidar2cam'])
        between=new[(new.frame>gap.left_frame)&(new.frame<gap.right_frame)]
        middle=int(between.iloc[np.argmin(abs(between.t_s-(gap.start_s+gap.end_s)/2))].frame)
        frames=[int(gap.left_frame),middle,int(gap.right_frame)];boxes=[];points=[]
        for f in frames:
            s=new.loc[new.frame==f].iloc[0].to_dict()
            b=dict(s,l=dim['length_m'],w=dim['width_m'],h=dim['height_m']);boxes.append(b);points.extend(project(b,K,M))
        center=np.mean(points,axis=0);tiles=[]
        for j,(f,b) in enumerate(zip(frames,boxes)):
            im=cv2.imread(str(TR/'data/camera-data'/clip/'frames_all'/f'{f:03d}.jpg'))
            left=max(0,min(im.shape[1]-700,int(center[0]-350)));top=max(0,min(im.shape[0]-450,int(center[1]-225)))
            draw(im,b,K,M,(0,170,255) if j==1 else (0,230,80),'estimated' if j==1 else 'observed')
            tile=np.full((490,700,3),250,np.uint8);tile[40:]=im[top:top+450,left:left+700]
            cv2.putText(tile,f'{clip} f{f} t={b["t_s"]:.3f}',(10,25),0,.62,(20,20,20),1);tiles.append(tile)
        cv2.imwrite(str(folder/f'{clip}.jpg'),np.hstack(tiles))
        selected.append(dict(clip=clip,left_frame=frames[0],middle_frame=middle,right_frame=frames[2],gap_s=gap.anchor_gap_s))
        fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True,constrained_layout=True)
        for ax,col in zip(axes,['x','y','speed_mps']):
            ax.scatter(old.t_s,old[col],s=6,label='Existing detector-supported states',color='#2878a5')
            fill=new[new.is_imputed];ax.scatter(fill.t_s,fill[col],s=10,label='Bounded linear estimates',color='#dc7328')
            for g in gg.itertuples():ax.axvspan(g.start_s,g.end_s,color='#dc7328',alpha=.07)
            ax.set_ylabel(col+' (m/s)' if col=='speed_mps' else col+' (m)');ax.grid(alpha=.2)
        axes[0].legend(fontsize=8);axes[-1].set_xlabel('Source video time (s)')
        fig.suptitle(clip+': observed points preserved; interpolation is not measured behavior')
        fig.savefig(figure_folder/f'{clip}.png',dpi=150);plt.close(fig)
    save(OUT/'source_review_manifest.json',dict(records=selected,selection='Largest accepted target gap per repaired recording; endpoint/midpoint source frames, not exhaustive visual validation'))
    print('REVIEW_SHEETS',len(selected))


if __name__=='__main__':main()
