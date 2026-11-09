"""Bind runnable configurations to this server, preserving frozen artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data',required=True,type=Path)
args=parser.parse_args()
data=args.data.expanduser().resolve()
protocol=json.loads((root/'data_records/records/prepared_protocol.json').read_text())
for name,expected in protocol['sha256'].items():
    p=data/name
    if not p.is_file():raise SystemExit(f'Missing required data: {p}')
    if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
        raise SystemExit(f'Data SHA256 differs from the archived protocol: {p}')
models={
    'cnn':'20261010_refinement/runs/final_cnn_expanded',
    'transformer':'20261010_refinement/runs/final_transformer_expanded',
    'rf':'20261011_native11/runs/native11_rf_expanded',
}
for p in sorted((root/'training').glob('*/experiment.json')):
    cfg=json.loads(p.read_text());cfg['prepared']=str(data)
    if 'original_models' in cfg:
        cfg['original_models']={k:str(root/'artifacts/experiments'/v) for k,v in models.items()}
    p.write_text(json.dumps(cfg,indent=2)+'\n')
    print('Configured',p.relative_to(root))
print('Data hashes verified. Only runnable path settings changed; frozen artifacts preserved.')
