"""Use installed Chrome headless in an isolated workspace profile for UI QA."""
import argparse
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
from build_field_core_dataset import ROOT, read, save

OUT=ROOT/'experiments/field_leader_review_20261102'
chrome=Path('C:/Program Files/Google/Chrome/Application/chrome.exe')
profile=ROOT/'.runtime/leader-review-browser-qa'
profile.mkdir(parents=True,exist_ok=True)
shutil.copyfile(ROOT/'scripts/leader_review_ui/qa.html',OUT/'qa.html')
base=[str(chrome),'--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check',
      '--disable-extensions','--disable-background-networking',f'--user-data-dir={profile}',
      '--window-size=1440,1200','--hide-scrollbars']
parser=argparse.ArgumentParser();parser.add_argument('--base-url',default='http://127.0.0.1:8765');args=parser.parse_args()
url=args.base_url.rstrip('/')+'/experiments/field_leader_review_20261102/'
result=subprocess.run(base+['--virtual-time-budget=20000','--dump-dom',url+'qa.html'],capture_output=True,timeout=60)
text=result.stdout.decode('utf-8','replace')
(OUT/'qa_dom.html').write_text(text,encoding='utf-8')
(OUT/'qa_browser.log').write_bytes(result.stderr)
matches=re.findall(r'<pre id="result">(.*?)</pre>',text,re.S)
assert matches,'QA result absent; inspect qa_browser.log'
report=json.loads(html.unescape(matches[0]));save(OUT/'ui_validation.json',report);print(json.dumps(report))
assert report['status']=='PASS',report
shot=OUT/'review_page_preview.png'
subprocess.run(base+['--virtual-time-budget=3000',f'--screenshot={shot}',url+'index.html'],capture_output=True,timeout=30,check=True)
assert shot.exists()
