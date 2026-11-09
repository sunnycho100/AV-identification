"""Build a source-only release with synthetic demo data, never field data."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
INCLUDE={'server.py','trajectory_review.py','START.ps1','README.md','CHANGELOG.md'}
FOLDERS={'public','docs','schemas','examples','tests','tools','.vscode'}
EXCLUDE={'__pycache__','.test-runtime','reviews','node_modules'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    files=[]
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():continue
        rel=path.relative_to(ROOT)
        if any(p in EXCLUDE for p in rel.parts) or path.suffix in {'.pyc','.zip'}:continue
        if rel.as_posix() in INCLUDE or rel.parts[0] in FOLDERS:files.append(path)
    dest=args.output/'roadside-trajectory-review-3.0.1.zip'
    manifest={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in files:archive.write(path,'roadside-trajectory-review/'+path.relative_to(ROOT).as_posix())
        archive.writestr('roadside-trajectory-review/SOURCE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    checksum=hashlib.sha256(dest.read_bytes()).hexdigest()
    (args.output/'SHA256SUMS.txt').write_text(checksum+'  '+dest.name+'\n',encoding='utf-8')
    print(json.dumps(dict(archive=str(dest),files=len(files),bytes=dest.stat().st_size,sha256=checksum)))


if __name__=='__main__':main()
