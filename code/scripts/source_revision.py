"""Preserve archive baselines and record intentional source revisions."""
import argparse, hashlib, json, shutil
from pathlib import Path
from workspace_paths import resolve_legacy, ARCHIVE
ROOT=Path(__file__).resolve().parent.parent
LEDGER=ROOT/'catalogs/workspace_changes.json'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def snapshot(paths, reason):
    rows=json.loads(LEDGER.read_text()) if LEDGER.exists() else []
    for rel in paths:
        if any(r['path']==rel for r in rows): continue
        p=resolve_legacy(rel)
        backup=ARCHIVE/'revisions/before_field_pilot_20261101'/rel.removeprefix('../')
        backup.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,backup)
        rows.append(dict(path=rel,original_sha256=sha(p),backup='../archive/'+backup.relative_to(ARCHIVE).as_posix(),reason=reason))
    LEDGER.write_text(json.dumps(rows,indent=2),encoding='utf-8')
def refresh():
    rows=json.loads(LEDGER.read_text())
    for r in rows:
        assert sha(resolve_legacy(r['backup']))==r['original_sha256']
        r['current_sha256']=sha(resolve_legacy(r['path']));r['current_bytes']=(resolve_legacy(r['path'])).stat().st_size
    LEDGER.write_text(json.dumps(rows,indent=2),encoding='utf-8')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('paths',nargs='*');ap.add_argument('--refresh',action='store_true');a=ap.parse_args()
    refresh() if a.refresh else snapshot(a.paths,'User-authorized field pipeline integration and manuscript update, 2026-11-01')
