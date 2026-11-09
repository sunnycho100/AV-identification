"""Restore only the five labeled training sources; preserve recorded provenance."""
import concurrent.futures
import hashlib
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
BASE=ROOT/'classification/trajectory-avlike-transformer'
OUT=BASE/'data/restored_20261102'
JRC='https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/TransportExpData/JRCDBT0001/LATEST/'


def main():
    sys.stdout.reconfigure(encoding='utf-8');OUT.mkdir(parents=True,exist_ok=True)
    old=json.loads((BASE/'data_records/records/raw_sources.json').read_text(encoding='utf-8'))
    jobs=[]
    for r in old:
        c=r['campaign'];name=Path(r['file']).name
        if c not in ['Vicolungo','ASta','Casale']:continue
        remote=name
        if c=='ASta':
            num=int(name.split('platoon')[1].split('.')[0]);remote=f'ASta_{"050719" if num<3 else "040719"}_platoon{num}.csv'
        folder={'ASta':'AstaZero','Casale':'Casale','Vicolungo':'Vicolungo'}[c]
        dest=OUT/'openacc'/('' if c=='Vicolungo' else c)/name
        jobs.append(dict(url=JRC+folder+'/'+remote,path=dest,expected_sha256=r['sha256'],campaign=c))
    external=json.loads((BASE/'data_records/records/external_downloads.json').read_text(encoding='utf-8'))['files']
    for r in external:
        if r['file'] not in ['cats_original.zip','vanderbilt_step2.csv','ultra_metadata_v4.json']:continue
        jobs.append(dict(url=r['url'],path=OUT/'external'/r['file'],expected_sha256=r['sha256'],campaign='external'))
    def fetch(j):
        p=j['path'];p.parent.mkdir(parents=True,exist_ok=True)
        for attempt in range(3):
            try:
                if not p.exists():
                    req=urllib.request.Request(j['url'],headers={'User-Agent':'trajectory-research/1.0'})
                    tmp=p.with_suffix(p.suffix+'.partial')
                    with urllib.request.urlopen(req,timeout=40) as response,tmp.open('wb') as f:
                        while True:
                            block=response.read(1024*1024)
                            if not block:break
                            f.write(block)
                    tmp.replace(p)
                sha=hashlib.sha256(p.read_bytes()).hexdigest()
                result=dict(j,path=str(p.relative_to(ROOT)),sha256=sha,bytes=p.stat().st_size,
                            matches_original=sha==j['expected_sha256'],status='downloaded')
                print(p.name,p.stat().st_size,'bytes; original hash',result['matches_original'],flush=True)
                return result
            except Exception as e:
                if attempt==2:return dict(url=j['url'],path=str(p.relative_to(ROOT)),status='failed',error=str(e))
                time.sleep(1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(fetch,jobs))
    (OUT/'downloads.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    failures=[r for r in results if r['status']=='failed']
    if failures:raise RuntimeError(f'{len(failures)} downloads failed; see downloads.json')
    with zipfile.ZipFile(OUT/'external/cats_original.zip') as z:
        for member in z.infolist():
            target=(OUT/'external'/member.filename).resolve()
            if not target.is_relative_to((OUT/'external').resolve()):raise ValueError('Unsafe archive path')
        z.extractall(OUT/'external')
    print('RESTORED',len(results),'files; matching original hashes',sum(r['matches_original'] for r in results),flush=True)


if __name__=='__main__':main()
