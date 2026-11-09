"""Decode and calibrate the recording-disjoint adaptation training clips.

Run with .venv-classifier (the AnyCalib runtime), before adapt_new_views.py.
"""
import json
import pandas as pd
from prepare_baseline_videos import ROOT, prepare, calibrate

if __name__=='__main__':
    cfg=json.loads((ROOT/'configs/view_adaptation.json').read_text(encoding='utf-8'))
    selected=dict(clips=sum(cfg['train'].values(),[]),height_m=15.0)
    out=ROOT/'experiments'/cfg['experiment'];out.mkdir(parents=True,exist_ok=True)
    catalog=pd.read_csv(ROOT/'catalogs/field_clips.csv').set_index('clip_id')
    prepare(selected,catalog,out)
    calibrate(selected,catalog,out)
