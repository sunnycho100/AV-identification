"""Stdlib-only review service for VS Code Remote SSH port forwarding."""
import argparse
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trajectory_review import discover

VERSION = '3.0.1'
SERVICE = 'identification-leader-review'
PACK_NAME = 'field_leader_review_20261102'
ROOT = Path(__file__).resolve().parent.parent
PACK = ROOT / 'experiments' / PACK_NAME
PUBLIC = Path(__file__).resolve().parent / 'public'
APP_PATH = f'/experiments/{PACK_NAME}/index.html'
PROJECTS = {}
LEADER_STATES = {'confirm_pair': 'present', 'no_leader': 'absent',
                 'leader_not_visible': 'not_visible', 'uncertain': 'unknown',
                 'target_wrong': 'unknown', 'exclude_quality': 'unknown'}
DECISIONS = set(LEADER_STATES)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_document(doc):
    require(isinstance(doc, dict), '审核内容必须为 JSON 对象')
    manifest = read(PACK / 'manifest.json')
    require(doc.get('schema') == 'field-leader-decisions/v1', '审核格式不匹配')
    require(doc.get('pack_id') == manifest['pack_id'], '审核包版本不匹配')
    records = doc.get('records')
    require(isinstance(records, list) and len(records) <= 10000, '记录数量无效')
    cases = {c['clip']: c for c in manifest['cases']}
    ids, times_cache, tracks_cache = set(), {}, {}
    for r in records:
        require(isinstance(r, dict) and r.get('clip') in cases, '视频不存在')
        c = cases[r['clip']]
        require(r.get('source_video_sha256') == c['source_video_sha256'] and
                r.get('tracks_sha256') == c['tracks_sha256'], '源文件版本不匹配')
        require(isinstance(r.get('record_id'), str) and r['record_id'] not in ids, '记录编号重复或无效')
        ids.add(r['record_id'])
        require(r.get('decision') in DECISIONS, '审核结论无效')
        require(r.get('leader_state', LEADER_STATES[r['decision']]) == LEADER_STATES[r['decision']],
                '前车状态与审核结论不一致')
        if r['decision'] in ('no_leader', 'leader_not_visible'):
            require(r.get('leader') is None, '无前车或前车不可见时，前车字段应为空')
        a, b = r.get('start_frame'), r.get('end_frame')
        require(type(a) == int and type(b) == int and 0 <= a <= b < c['frames'], '起止帧无效')
        if r['clip'] not in times_cache:
            times_cache[r['clip']] = {v['frame']: v['t_s'] for v in
                read(ROOT / 'trajectory/data/camera-data' / r['clip'] / 'frame_times.json')['frames']}
        times = times_cache[r['clip']]
        require(abs(r['start_s'] - times[a]) < 1e-6 and abs(r['end_s'] - times[b]) < 1e-6,
                '时间与原始帧不匹配')
        if r['decision'] in ('confirm_pair', 'no_leader'):
            if r['clip'] not in tracks_cache:
                tracks_cache[r['clip']] = read(ROOT / 'trajectory/outputs/trajectories' /
                    f'{r["clip"]}_{c["tag"]}' / 'tracks.json')['tracks']
            tracks = tracks_cache[r['clip']]
            roles = ('target', 'leader') if r['decision'] == 'confirm_pair' else ('target',)
            for role in roles:
                s = r.get(role)
                require(isinstance(s, dict), '缺少车辆选择')
                require(s.get('type') in ('track', 'detection', 'image', 'manual_box'), '车辆选择类型无效')
                if s['type'] == 'track':
                    require(s.get('id') in tracks, '轨迹编号无效')
                else:
                    require(type(s.get('frame')) == int and a <= s['frame'] <= b, '框选帧不在区间内')
                    bb = s.get('bbox')
                    require(isinstance(bb, list) and len(bb) == 4 and
                            all(type(v) in (float, int) and abs(v) < 100000 for v in bb), '图像框无效')
                    require(bb[2] > bb[0] and bb[3] > bb[1], '图像框面积无效')
            if r['decision'] == 'confirm_pair':
                require(not (r['target'].get('id') and r['target'].get('id') == r['leader'].get('id')),
                        '自车与前车相同')
    return len(records)


def same_local_origin(origin, host):
    """Use browser Host, including LOCAL port, instead of the SSH destination port."""
    if not origin or not host:
        return False
    try:
        parsed = urlsplit(origin)
        authority = urlsplit('http://' + host)
        return (parsed.scheme in ('http', 'https') and not parsed.path and
                not parsed.query and not parsed.fragment and not parsed.username and
                parsed.hostname in ('localhost', '127.0.0.1', '::1') and
                authority.hostname == parsed.hostname and
                parsed.netloc.lower() == host.lower())
    except ValueError:
        return False


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.remaining_bytes = None
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def reply_json(self, status, value):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == '/':
            self.send_response(302)
            self.send_header('Location', '/app/')
            self.send_header('Content-Length', '0')
            self.end_headers()
        elif path == '/healthz':
            self.reply_json(200, dict(service=SERVICE, version=VERSION, review_path='/app/', legacy_review_path=APP_PATH,
                                     data_ready=bool(PROJECTS) or (PACK / 'manifest.json').is_file(),
                                     projects=list(PROJECTS)))
        elif path.startswith('/api/projects'):
            try:
                parts = path.strip('/').split('/')
                if parts == ['api', 'projects']:
                    self.reply_json(200, dict(projects=[p.manifest() for p in PROJECTS.values()],
                        legacy_url=APP_PATH if (PACK / 'manifest.json').is_file() else None))
                    return
                require(len(parts) >= 3 and parts[2] in PROJECTS, '项目不存在')
                project = PROJECTS[parts[2]]
                if len(parts) == 3:
                    value = project.manifest()
                elif len(parts) == 5 and parts[3] == 'clips':
                    value = project.clip_data(parts[4])
                elif len(parts) == 4 and parts[3] == 'reviews':
                    value = project.reviews()
                elif len(parts) == 5 and parts[3] == 'reviews':
                    require(re.fullmatch(r'review_[A-Za-z0-9_]+\.json', parts[4]), '版本名无效')
                    value = read(project.review_dir / parts[4])
                else:
                    raise ValueError('接口不存在')
                self.reply_json(200, value)
            except (ValueError, KeyError, OSError):
                self.reply_json(404, dict(error='找不到项目或审核版本'))
        else:
            try:
                super().do_GET()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    def do_HEAD(self):
        if urlsplit(self.path).path in ('/', '/healthz') or self.path.startswith('/api/projects'):
            self.do_GET()
        else:
            super().do_HEAD()

    def static_path(self):
        name = unquote(urlsplit(self.path).path)
        require('\\' not in name and '..' not in name.split('/'), '路径无效')
        if name.startswith('/app/') or name.startswith('/trajectory-review/'):
            prefix = '/app/' if name.startswith('/app/') else '/trajectory-review/'
            relative = name[len(prefix):] or 'index.html'
            folder = PUBLIC / ('hub' if prefix == '/app/' else 'trajectory')
            path = (folder / relative).resolve()
            require(path.is_relative_to(folder.resolve()), '资源路径无效')
            return path
        if name.startswith('/media/'):
            parts = name.strip('/').split('/')
            require(len(parts) in (4, 5) and parts[1] in PROJECTS, '媒体项目无效')
            project = PROJECTS[parts[1]]
            require((len(parts) == 4 and parts[3] == 'video') or
                    (len(parts) == 5 and parts[3] == 'frame' and parts[4].isdigit()), '媒体路径无效')
            return project.media(parts[2], parts[3], int(parts[4]) if len(parts) == 5 else None)
        for asset in ('index.html', 'app.js', 'style.css'):
            if name == f'/experiments/{PACK_NAME}/{asset}':
                return PUBLIC / asset
        allowed = (f'/experiments/{PACK_NAME}/', '/trajectory/data/camera-data/', '/data/field/')
        require(any(name.startswith(prefix) for prefix in allowed), '不是审核资源')
        path = (ROOT / name.lstrip('/')).resolve()
        require(path.is_relative_to(ROOT.resolve()), '路径超出工作区')
        return path

    def send_head(self):
        self.remaining_bytes = None
        try:
            path = self.static_path()
            require(path.is_file(), '文件不存在')
            stream = path.open('rb')
        except (ValueError, OSError):
            self.send_error(404)
            return None
        size = path.stat().st_size
        start, end = 0, size - 1
        requested = self.headers.get('Range')
        if requested:
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested.strip())
            try:
                require(match is not None and any(match.groups()), '无效 Range')
                left, right = match.groups()
                if left:
                    start = int(left)
                    end = min(int(right), size - 1) if right else size - 1
                else:
                    suffix = int(right)
                    require(suffix > 0, '无效 Range')
                    start = max(0, size - suffix)
                require(0 <= start <= end < size, 'Range 超出文件')
            except ValueError:
                stream.close()
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return None
        self.send_response(206 if requested else 200)
        self.send_header('Content-Type', self.guess_type(str(path)))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(end - start + 1))
        self.send_header('Cache-Control', 'no-cache' if path.suffix in ('.js', '.html', '.css', '.json') else 'private, max-age=3600')
        if requested:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        stream.seek(start)
        self.remaining_bytes = end - start + 1
        return stream

    def copyfile(self, source, outputfile):
        remaining = self.remaining_bytes
        while remaining is not None and remaining > 0:
            chunk = source.read(min(256 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)

    def do_POST(self):
        match = re.fullmatch(r'/api/projects/([A-Za-z0-9_-]+)/(reviews|validate)', self.path)
        if self.path != '/api/leader-review/save' and not match:
            self.send_error(404)
            return
        try:
            require(same_local_origin(self.headers.get('Origin'), self.headers.get('Host')),
                    '请从相同的本机转发地址打开审核页后保存')
            require(self.headers.get_content_type() == 'application/json', '需要 JSON 内容')
            n = int(self.headers.get('Content-Length', '0'))
            require(0 < n < 2_000_000, '文件大小无效')
            doc = json.loads(self.rfile.read(n))
            if match:
                require(match[1] in PROJECTS, '项目不存在')
                project = PROJECTS[match[1]]
                result = project.save(doc) if match[2] == 'reviews' else dict(ok=True, records=project.validate(doc))
                self.reply_json(200, result)
                return
            count = validate_document(doc)
            require(all(r.get('reviewer') == 'user' for r in doc['records']), '测试记录不能保存为用户标注')
            # Additive metadata: legacy invisible/uncertain records are never
            # reinterpreted as an explicitly absent leader.
            for record in doc['records']:
                record['leader_state'] = LEADER_STATES[record['decision']]
            folder = PACK / 'user_decisions'
            folder.mkdir(exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
            dest = folder / f'leader_review_{stamp}.json'
            with dest.open('x', encoding='utf-8') as stream:
                json.dump(doc, stream, ensure_ascii=False, indent=2, allow_nan=False)
            self.reply_json(200, dict(ok=True, records=count, path=dest.relative_to(ROOT).as_posix()))
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            self.reply_json(400, dict(ok=False, error=str(exc)))
        except OSError:
            self.reply_json(500, dict(ok=False, error='无法写入工作区，请导出 JSON 保留审核结果'))


def main():
    global ROOT, PACK, PROJECTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=ROOT, help='identification 数据工作区')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--project', type=Path, help='额外项目的 project.json；独立发布包默认加载合成演示')
    args = parser.parse_args()
    ROOT = args.workspace.resolve()
    PACK = ROOT / 'experiments' / PACK_NAME
    PROJECTS = discover(ROOT, PUBLIC.parent, args.project)
    if not PROJECTS and not (PACK / 'manifest.json').is_file():
        parser.error('找不到项目或演示数据。请指定 --workspace 或 --project。')
    if not all((PUBLIC / p).is_file() for p in ('index.html', 'app.js', 'style.css')):
        parser.error('开发包缺少 public 前端文件。')
    url = f'http://127.0.0.1:{args.port}/'
    try:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    except OSError:
        try:
            with urlopen(url + 'healthz', timeout=2) as response:
                running = json.load(response)
            if running.get('service') == SERVICE and running.get('version') == VERSION:
                print(f'Review service already running: {url}', flush=True)
                return
        except (OSError, ValueError):
            pass
        parser.error(f'端口 {args.port} 已占用；可用 --port 8766 并转发对应端口。')
    print(f'Review ready: {url}', flush=True)
    print(f'VS Code Remote SSH: Ports -> Forward a Port -> {args.port} -> Open in Browser', flush=True)
    print(f'Annotations: {PACK / "user_decisions"}', flush=True)
    for project in PROJECTS.values():
        print(f'Project {project.id}: {project.review_dir}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nReview server stopped.', flush=True)
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
