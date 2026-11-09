"""Compile the revised manuscript and render every page for visual review."""
import argparse,json,subprocess,re
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parent.parent
ap=argparse.ArgumentParser();ap.add_argument('--experiment',default='field_branches_20261101');args=ap.parse_args()
EXP=ROOT/'experiments'/args.experiment;paper=ROOT/'../paper'
qa=EXP/'paper_qa';qa.mkdir(parents=True,exist_ok=True)
tex=ROOT/'.runtime/tectonic/tectonic.exe'
result=subprocess.run([str(tex),'--keep-logs','Manuscript.tex'],cwd=paper,capture_output=True,text=True,encoding='utf-8',errors='replace')
(EXP/'paper_compile.log').write_text(result.stdout+result.stderr,encoding='utf-8')
if result.returncode:raise RuntimeError(result.stderr[-4000:])
pop=ROOT/'.runtime/poppler/Library/bin'
subprocess.run([str(pop/'pdftoppm.exe'),'-scale-to','1600','-png',str(paper/'Manuscript.pdf'),str(qa/'page')],check=True)
subprocess.run([str(pop/'pdftotext.exe'),'-layout',str(paper/'Manuscript.pdf'),str(qa/'manuscript.txt')],check=True)
info=subprocess.run([str(pop/'pdfinfo.exe'),str(paper/'Manuscript.pdf')],capture_output=True,text=True,check=True).stdout
n_pages=int(re.search(r'Pages:\s+(\d+)',info).group(1))
pages=sorted(p for p in qa.glob('page-*.png') if int(p.stem.split('-')[-1])<=n_pages);w,h=390,555
sheet=Image.new('RGB',(w*3,h*((len(pages)+2)//3)),'#d8d8d8');draw=ImageDraw.Draw(sheet)
for n,p in enumerate(pages):
    im=Image.open(p);im.thumbnail((w-12,h-24));x=(n%3)*w+6;y=(n//3)*h+20
    sheet.paste(im,(x,y));draw.text((x,y-16),f'Page {n+1}',fill='black')
sheet.save(qa/'contact_sheet.jpg',quality=92)
texts=(qa/'manuscript.txt').read_text(encoding='utf-8').split('\f')
terms=['Collected roadside','Frozen field-pilot','On the common','Timestamp-aligned speed','Temporal pseudo-label']
locations={term:[i+1 for i,t in enumerate(texts) if term in t] for term in terms}
log=(paper/'Manuscript.log').read_text(encoding='utf-8',errors='replace')
report=dict(compile_exit_code=result.returncode,pages=len(pages),section_pages=locations,undefined_references='There were undefined references' in log,overfull_warning_count=log.count('Overfull'),visual_review='pending')
(qa/'qa.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
