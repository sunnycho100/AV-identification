"""Prepare temporal pseudo-labels and run a small held-out-recording adaptation."""
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
cfg=json.loads((ROOT/'configs/field_pilot.json').read_text());tr=ROOT/'trajectory';exp=ROOT/'experiments'/cfg['experiment']
env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',NUMBA_DISABLE_JIT='1',TORCH_HOME=str(ROOT/'.runtime/torch-cache'))
for clip in cfg['train']+cfg['validation']:
    cmd=[sys.executable,'scripts/finetune/make_pseudo_labels.py','--clip',clip,'--det-dir',str(tr/'outputs/object_detection/camera-data'/f'{clip}_pilot_base'),'--tracks-json',str(tr/'outputs/trajectories'/f'{clip}_pilot_base'/'tracks.json'),'--out',str(tr/'outputs/finetune/pilot_labels'/clip)]
    subprocess.run(cmd,cwd=tr,env=env,check=True)
f=cfg['finetune']
cmd=[sys.executable,'-u','scripts/finetune/train_finetune.py','--epochs',str(f['epochs']),'--lr',str(f['lr']),'--device','cuda','--ckpt',cfg['baseline_checkpoint'],'--labels','pilot_labels','--train-clips',*cfg['train'],'--val-clips',*cfg['validation'],'--limit',str(f['train_frames_per_clip']),'--val-limit',str(f['validation_frames_per_clip']),'--uniform-sample','--out','outputs/finetune/pilot_head']
(exp/'finetune_command.json').write_text(json.dumps(dict(command=cmd,config=cfg['finetune']),indent=2))
print('Training head; see finetune.log',flush=True)
with (exp/'finetune.log').open('w',encoding='utf-8') as log:subprocess.run(cmd,cwd=tr,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
history=json.loads((tr/'outputs/finetune/pilot_head/history.json').read_text())
best=min(history,key=lambda r:r['val_loss'])
ckpt=cfg['baseline_checkpoint'] if best['epoch']==0 else f"outputs/finetune/pilot_head/head_ft_ep{best['epoch']}.ckpt"
# Include epoch zero so an inferior adaptation does not silently replace the baseline.
selection=dict(selected_epoch=best['epoch'],checkpoint=ckpt,criterion='minimum validation pseudo-label loss including epoch zero; test recordings unused',history=history)
(exp/'finetune_selection.json').write_text(json.dumps(selection,indent=2));print(json.dumps(selection),flush=True)
