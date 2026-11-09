"""Headless Chrome UI checks. All saves go to an isolated temporary directory."""
import argparse
import html
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading

PACKAGE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('review_qa_server',PACKAGE/'server.py')
web=importlib.util.module_from_spec(spec);spec.loader.exec_module(web)
from trajectory_review import Project


class Handler(web.Handler):
    def static_path(self):
        if self.path=='/qa':return PACKAGE/'tests/browser_qa.html'
        return super().static_path()
    def log_message(self,*args):pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chrome',type=Path,required=True)
    parser.add_argument('--workspace',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    base=PACKAGE/'.test-runtime';base.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='browser-',dir=base) as tmp:
        root=Path(tmp).resolve();demo=root/'demo';shutil.copytree(PACKAGE/'examples/demo',demo,ignore=shutil.ignore_patterns('reviews'))
        web.ROOT=args.workspace.resolve() if args.workspace else root
        web.PACK=web.ROOT/'experiments'/web.PACK_NAME
        web.PROJECTS={'synthetic_demo':Project(demo/'project.json',root,root/'qa-only-reviews')}
        if args.workspace:
            for path in (web.ROOT/'data/review_projects').glob('*/project.json'):
                p=Project(path,web.ROOT,root/'qa-only-reviews');web.PROJECTS[p.id]=p
        server=web.ThreadingHTTPServer(('127.0.0.1',0),Handler)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        try:
            flags=[str(args.chrome),'--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check',
                '--disable-extensions','--disable-background-networking',f'--user-data-dir={root / "chrome"}',
                '--window-size=1600,1450','--hide-scrollbars','--virtual-time-budget=30000','--dump-dom',
                f'--screenshot={args.output.resolve() / "review-ui.png"}',f'http://127.0.0.1:{server.server_port}/qa']
            result=subprocess.run(flags,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
            (args.output/'browser-stderr.txt').write_text(result.stderr,encoding='utf-8')
            match=re.search(r'<pre id="result">(.*?)</pre>',result.stdout,re.S)
            report=json.loads(html.unescape(match[1])) if match and match[1]!='RUNNING' else {'status':'FAIL','error':'No completed browser report','exit_code':result.returncode}
            (args.output/'browser-results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
            print(json.dumps(report,ensure_ascii=False))
            if report['status']!='PASS':
                (args.output/'failed-dom.html').write_text(result.stdout,encoding='utf-8');raise SystemExit(1)
        finally:server.shutdown();server.server_close()
        assert root.is_relative_to(base.resolve())


if __name__=='__main__':main()
