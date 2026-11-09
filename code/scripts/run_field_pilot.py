"""Reproduce baseline or fine-tuned trajectories with the frozen recording split."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['baseline', 'finetuned'], default='baseline')
    ap.add_argument('--checkpoint')
    args = ap.parse_args()
    cfg = json.loads((ROOT/'configs/field_pilot.json').read_text())
    exp = ROOT/'experiments'/cfg['experiment']; exp.mkdir(exist_ok=True, parents=True)
    tr = ROOT/'trajectory'
    env = dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', NUMBA_DISABLE_JIT='1', TORCH_HOME=str(ROOT/'.runtime/torch-cache'))
    clips = cfg['train'] + cfg['validation'] + cfg['test'] if args.stage == 'baseline' else cfg['validation'] + cfg['test']
    tag = cfg['baseline_tag'] if args.stage == 'baseline' else cfg['finetuned_tag']
    ckpt = args.checkpoint or cfg['baseline_checkpoint']
    report = []
    for clip in clips:
        log = exp/f'{clip}_{tag}.log'
        det = tr/'outputs/object_detection/camera-data'/f'{clip}_{tag}'
        n_frames = len(list((tr/'data/camera-data'/clip/'frames_all').glob('*.jpg')))
        n_dets = len(list(det.glob('*_pred.json'))) if det.exists() else 0
        if (det/'calibration_used.json').exists() and n_dets != n_frames:
            raise RuntimeError(f'Incomplete detection cache: {clip} {n_dets}/{n_frames}; use a fresh tag.')
        cmd = [sys.executable, 'scripts/pipeline/extract_trajectories.py', '--clip', clip, '--tag', tag,
               '--ckpt', ckpt, '--score-thresh', str(cfg['score_thresh']), '--track-low-thresh', str(cfg['track_low_thresh'])]
        t = time.time(); print(f'Start {clip} {tag}: {n_frames} frames', flush=True)
        with log.open('w', encoding='utf-8') as f:
            r = subprocess.run(cmd, cwd=tr, env=env, stdout=f, stderr=subprocess.STDOUT)
        if r.returncode:
            print(log.read_text(encoding='utf-8')[-6000:]); raise SystemExit(r.returncode)
        summary = json.loads((tr/'outputs/trajectories'/f'{clip}_{tag}'/'run.json').read_text())
        record = {'clip': clip, 'tag': tag, 'frames': n_frames, 'seconds': round(time.time()-t,2), 'vehicles': summary['vehicles']}
        report.append(record); print(json.dumps(record), flush=True)
        (exp/f'{tag}_runs.json').write_text(json.dumps(report, indent=2))

if __name__ == '__main__': main()
