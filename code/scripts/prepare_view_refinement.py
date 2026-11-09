"""Reproduce preparation and collect calibration for all additional pilot clips."""
import pandas as pd
from prepare_baseline_videos import ROOT,prepare,calibrate,read

if __name__=='__main__':
    cfg=read(ROOT/'configs/view_refinement.json');follow=read(ROOT/'configs/seminole_following.json');out=ROOT/'experiments'/cfg['experiment']
    selected=dict(clips=sum(cfg['train'].values(),[])+['AV_S_WE_4','HV_S_WE_3']+[r['clip'] for r in follow['excerpts']],height_m=15.)
    cat=pd.read_csv(ROOT/'catalogs/field_clips.csv').set_index('clip_id');prepare(selected,cat,out);calibrate(selected,cat,out)
