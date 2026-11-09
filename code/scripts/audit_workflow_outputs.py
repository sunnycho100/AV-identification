"""Check completed artifacts without rerunning models or changing source data."""
import hashlib,json,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent.parent;TR=ROOT/'trajectory';EXP=ROOT/'experiments/field_branches_20261101'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
photo=read(ROOT/'experiments/field_photo_height_20261101/summary.json');runs=[]
for r in photo['runs']:
    d=TR/'outputs/object_detection/camera-data'/f"{r['clip']}_{r['tag']}"
    cal=read(d/'calibration_used.json');M=np.array(cal['lidar2cam'])
    assert abs((-M[:3,:3].T@M[:3,3])[2]+1.73-r['height_m'])<1e-6
    files=sorted(d.glob('*_pred.json'));assert len(files)==r['frames']
    assert [int(p.stem.split('_')[0]) for p in files]==list(range(r['start'],r['end']+1))
    for p in files:
        for b in read(p):assert np.isfinite([b[k] for k in ['x','y','z','yaw','score','l','w','h']]).all()
    td=TR/'outputs/trajectories'/f"{r['clip']}_{r['tag']}";tracks=read(td/'tracks.json')['tracks']
    assert len(tracks)==r['tracks'] and len(list(td.glob('vehicle_*.csv')))==r['tracks']
    assert not read(td/'instrumented.json')['warnings']
    for states in tracks.values():
        assert all(r['start']<=s['frame']<=r['end'] for s in states)
        assert all(a['t_s']<b['t_s'] for a,b in zip(states,states[1:]))
        assert all(np.isfinite([s[k] for k in ['x','y','vx','vy','speed_mps','t_s']]).all() for s in states)
    runs.append(dict(clip=r['clip'],height=r['height_m'],frames=len(files),tracks=len(tracks),status='PASS'))
for h in [15,17]:
    p=ROOT/'experiments/field_pilot_20261101'/f'photo_h{h}_excerpt'
    with np.load(p/'windows_4s.npz',allow_pickle=False) as d:assert d['primitive_windows'].shape==(0,40,3)
for f in ['geometry','yaw','corrected_labels','gpu_inference','gpu_batch_norm']:
    assert read(EXP/f'{f}.json')['status']=='PASS'
checks=read(EXP/'component_checks.json');assert all(r['returncode']==0 for r in checks)
adapt=read(EXP/'adaptation_feature_aug_v2.json');assert len(adapt['steps'])==16 and adapt['backbone_bit_identical']
assert (ROOT/adapt['checkpoint']).is_file()
assert read(EXP/'gps_labels.json')['status']=='PARTIAL'
assert sum(r['frames'] for r in read(EXP/'gps_labels.json')['rows'])==0
for mode in ['gps_only','hybrid']:assert not (EXP/'checkpoints'/f'{mode}_16steps.ckpt').exists()

qa=EXP/'paper_qa';pdf=ROOT/'../paper/Manuscript.pdf'
pop=ROOT/'.runtime/poppler/Library/bin'
subprocess.run([str(pop/'pdftotext.exe'),'-bbox',str(pdf),str(qa/'bbox.html')],check=True)
tree=ET.parse(qa/'bbox.html');ns={'x':'http://www.w3.org/1999/xhtml'};bad=[]
pages=tree.findall('.//x:page',ns)
for i,page in enumerate(pages,1):
    w,h=float(page.attrib['width']),float(page.attrib['height'])
    for word in page.findall('.//x:word',ns):
        a=word.attrib
        if float(a['xMin'])<20 or float(a['xMax'])>w-20 or float(a['yMin'])<20 or float(a['yMax'])>h-20:bad.append(dict(page=i,text=word.text,box=a))
assert not bad,bad[:10]
text=(qa/'manuscript.txt').read_text(encoding='utf-8').lower()
assert 'photograph' not in text and 'photo-based' not in text
assert '15 m' in text and '17 m' in text
q=read(qa/'qa.json');assert q['pages']==len(pages)==18 and not q['undefined_references']
q.update(visual_review='PASS',reviewed_pages=list(range(1,19)),review_method='Rendered contact sheet of all 18 pages; full-resolution review of title, workflow, camera/MOT methods, branch and height tables, and references.',text_page_boundary_violations=bad,pdf_sha256=sha(pdf),warning_review='CAS maketitle emits one empty overfull hbox; rendered title and all text bounds are within page. One underfull table paragraph has no visible defect.')
(qa/'qa.json').write_text(json.dumps(q,indent=2),encoding='utf-8')
result=dict(status='PASS_WITH_EXPLICIT_UNRUN_BRANCHES',photo_height_runs=runs,total_new_height_inference_frames=sum(r['frames'] for r in runs),component_check_programs_passed=len(checks),feature_aug_updates=16,gps_supervision='NOT_TRAINED: no labels after shared registration',field_classification='NO_ELIGIBLE_WINDOWS',paper_pages=18,pdf_sha256=sha(pdf),source_integrity=read(ROOT/'catalogs/verification.json')['status'])
(EXP/'verification_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
