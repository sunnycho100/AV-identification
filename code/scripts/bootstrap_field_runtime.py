import hashlib, json, tarfile, urllib.request, zipfile
from pathlib import Path
root=Path(__file__).resolve().parent.parent
runtime=root/'.runtime'; downloads=runtime/'downloads';downloads.mkdir(parents=True,exist_ok=True)
records=[]
def fetch(url,name):
    target=downloads/name
    if not target.exists():
        req=urllib.request.Request(url,headers={'User-Agent':'identification-research-workspace'})
        with urllib.request.urlopen(req,timeout=90) as response,target.open('wb') as out:
            while block:=response.read(1024*1024):out.write(block)
    records.append(dict(url=url,path=target.relative_to(root).as_posix(),sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    return target
def unzip(source,dest):
    dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(source) as z:
        for info in z.infolist():
            target=(dest/info.filename).resolve()
            if not target.is_relative_to(dest.resolve()):raise ValueError(info.filename)
        z.extractall(dest)
source=fetch('https://github.com/open-mmlab/mmdetection3d/archive/refs/tags/v1.0.0rc6.zip','mmdetection3d-v1.0.0rc6.zip')
unzip(source,runtime/'vendor')
print('mmdetection3d source restored',flush=True)
req=urllib.request.Request('https://api.github.com/repos/tectonic-typesetting/tectonic/releases/latest',headers={'User-Agent':'identification-research-workspace'})
with urllib.request.urlopen(req,timeout=60) as response:release=json.load(response)
asset=next(a for a in release['assets'] if a['name'].endswith('x86_64-pc-windows-msvc.zip'))
unzip(fetch(asset['browser_download_url'],asset['name']),runtime/'tectonic')
print('Tectonic restored',release['tag_name'],flush=True)
unzip(fetch('https://nodejs.org/dist/v22.13.1/node-v22.13.1-win-x64.zip','node-v22.13.1-win-x64.zip'),runtime/'node')
print('Node marker runtime restored',flush=True)
(runtime/'download_provenance.json').write_text(json.dumps(records,indent=2)+'\n')