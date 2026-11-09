"""Portable project manifests and append-only trajectory interval reviews (stdlib)."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import uuid

SCHEMA = 'trajectory-review/v1'
PROJECT_SCHEMA = 'trajectory-review-project/v1'
DECISIONS = {'ready_for_retracking', 'needs_followup', 'unusable'}
ISSUES = {'occlusion', 'detector_jump', 'identity_switch', 'image_freeze', 'timestamp_gap', 'small_target', 'other'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def safe_id(value):
    return isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,100}', value) is not None


class Project:
    def __init__(self, manifest_path, boundary, review_root):
        self.path = Path(manifest_path).resolve()
        self.boundary = Path(boundary).resolve()
        self.doc = read(self.path)
        require(self.doc.get('schema') == PROJECT_SCHEMA, 'Project schema does not match')
        require(safe_id(self.doc.get('id')), 'Invalid project ID')
        self.id = self.doc['id']
        self.root = self.path.parent
        self.media_root = (self.root / self.doc.get('media_root', '.')).resolve()
        require(self.media_root.is_relative_to(self.boundary), 'Media directory is outside the workspace')
        self.review_dir = Path(review_root) / self.id
        self.cases = {c['clip']: c for c in self.doc['cases']}
        require(bool(self.cases) and len(self.cases) == len(self.doc['cases']) and all(safe_id(c) for c in self.cases), 'Duplicate or invalid clip ID')

    def case(self, clip):
        require(clip in self.cases, 'Clip does not belong to this project')
        return self.cases[clip]

    def clip_data(self, clip):
        case = self.case(clip)
        path = (self.root / case['frame_data']).resolve()
        require(path.is_relative_to(self.root), 'Invalid frame data path')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == case['frame_data_sha256'], 'Frame data changed. Regenerate the project manifest.')
        data = read(path)
        require(data['clip'] == clip, 'Frame data and video do not match')
        return data

    def media(self, clip, kind, frame=None):
        case = self.case(clip)
        if kind == 'video':
            require(case.get('video'), 'No original video file for this case')
            relative = case['video']
        elif kind == 'frame':
            require(type(frame) is int, 'Invalid frame number')
            frames = self.clip_data(clip)['frames']
            require(any(f['frame'] == frame for f in frames), 'Frame number does not exist')
            relative = case['frames_template'].format(frame=frame)
        else:
            raise ValueError('Invalid media type')
        path = (self.media_root / relative).resolve()
        require(path.is_relative_to(self.media_root) and path.is_file(), 'Media is missing or outside the project')
        return path

    def reviews(self):
        latest, history = {}, []
        if self.review_dir.exists():
            for path in sorted(self.review_dir.glob('review_*.json')):
                doc = read(path)
                if doc.get('schema') != SCHEMA or doc.get('project_id') != self.id:
                    continue
                history.append(dict(file=path.name, saved_at=doc.get('server_saved_at'),
                                    clips=[r['clip'] for r in doc['records']]))
                for row in doc['records']:
                    latest[row['clip']] = row
        return dict(records=list(latest.values()), revisions=history)

    def manifest(self):
        completed = {r['clip'] for r in self.reviews()['records']}
        cases = []
        for case in self.doc['cases']:
            row = {k: v for k, v in case.items() if k not in {'video', 'frames_template', 'frame_data'}}
            row['reviewed'] = row['clip'] in completed
            row['video_url'] = f'/media/{self.id}/{row["clip"]}/video' if case.get('video') else None
            row['data_url'] = f'/api/projects/{self.id}/clips/{row["clip"]}'
            row.setdefault('note', '')
            cases.append(row)
        return dict(schema=PROJECT_SCHEMA, id=self.id, title=self.doc['title'],
                    description=self.doc.get('description', ''), required_duration_s=self.doc.get('required_duration_s', 3),
                    demo=self.doc.get('demo', False), cases=cases, reviewed=len(completed))

    def validate(self, doc):
        require(isinstance(doc, dict) and doc.get('schema') == SCHEMA, 'Review schema does not match')
        require(doc.get('project_id') == self.id, 'Review belongs to another project')
        records = doc.get('records')
        require(isinstance(records, list) and 0 < len(records) <= len(self.cases), 'Invalid review count')
        seen = set()
        for r in records:
            require(isinstance(r, dict), 'Invalid review record')
            clip = r.get('clip')
            require(clip not in seen, 'Duplicate clips in this submission')
            seen.add(clip)
            case = self.case(clip)
            require(r.get('source_video_sha256') == case['source_video_sha256'] and
                    r.get('frame_data_sha256') == case['frame_data_sha256'], 'Source video or frame data version does not match')
            require(safe_id(r.get('record_id')), 'Invalid record ID')
            require(r.get('reviewer') == 'user', 'Submit as the human reviewer')
            require(r.get('decision') in DECISIONS, 'Select a review decision')
            require(isinstance(r.get('note', ''), str) and len(r.get('note', '')) <= 10000, 'Note is too long')
            require(isinstance(r.get('issues', []), list) and all(i in ISSUES for i in r.get('issues', [])), 'Invalid issue type')
            require(type(r.get('identity_continuous')) is bool, 'Confirm whether target identity is continuous')
            data = self.clip_data(clip)
            times = {f['frame']: f['t_s'] for f in data['frames']}
            width, height = data['width'], data['height']
            a, b = r.get('start_frame'), r.get('end_frame')
            require(type(a) is int and type(b) is int and a in times and b in times and a < b, 'Invalid start or end frame')
            for key, value in [('start_s', times[a]), ('end_s', times[b])]:
                require(finite(r.get(key)) and abs(r[key] - value) < 1e-6, 'Interval times do not match the original frames')
            trim = r.get('trim')
            require(isinstance(trim, dict), 'Retained interval is missing')
            lo, hi = trim.get('keep_start_frame'), trim.get('keep_end_frame')
            require(type(lo) is int and type(hi) is int and lo in times and hi in times,
                    f'Invalid retained interval: {lo}–{hi}. Enter frame numbers, not seconds.')
            require(lo <= hi, f'First retained frame {lo} must not follow last retained frame {hi}.')
            require(lo <= a < b <= hi,
                    f'Review interval {a}–{b} is outside retained frames {lo}–{hi}. '
                    'Adjust the review interval or restore the retained frame range.')
            exclusions = r.get('excluded_ranges', [])
            require(isinstance(exclusions, list) and len(exclusions) <= 100, 'Invalid excluded interval')
            for ex in exclusions:
                require(isinstance(ex, dict) and ex.get('kind') in ISSUES, 'Invalid exclusion reason')
                x, y = ex.get('start_frame'), ex.get('end_frame')
                require(type(x) is int and type(y) is int and x in times and y in times and x <= y, 'Invalid excluded frame range')
                require(isinstance(ex.get('note', ''), str) and len(ex.get('note', '')) <= 2000, 'Invalid exclusion note')
            anchors = r.get('anchors', [])
            require(isinstance(anchors, list) and len(anchors) <= 100, 'Invalid anchor count')
            af = set()
            for anchor in anchors:
                require(isinstance(anchor, dict), 'Invalid anchor')
                frame = anchor.get('frame')
                require(type(frame) is int and frame in times and a <= frame <= b and frame not in af,
                        'Keep one target anchor per frame, within the review interval')
                af.add(frame)
                bb = anchor.get('bbox')
                require(isinstance(bb, list) and len(bb) == 4 and all(finite(v) for v in bb), 'Invalid target box coordinates')
                require(0 <= bb[0] < bb[2] <= width and 0 <= bb[1] < bb[3] <= height, 'Target box is outside the image')
                require(anchor.get('source') in {'manual', 'candidate'}, 'Invalid anchor source')
            if r['decision'] == 'ready_for_retracking':
                require(times[b] - times[a] >= self.doc.get('required_duration_s', 3) - 1e-6, 'Re-extraction requires at least three continuous seconds')
                require(r['identity_continuous'] and len(anchors) >= 2, 'Confirm continuous identity and mark the target in at least two different frames')
                require(not any(ex['start_frame'] <= b and ex['end_frame'] >= a for ex in exclusions),
                        'A re-extraction interval cannot include excluded frames. Adjust the interval.')
                selected = [f for f in data['frames'] if a <= f['frame'] <= b]
                require(not any(y['t_s'] - x['t_s'] > .2 + 1e-6 for x, y in zip(selected, selected[1:])),
                        'Original timestamps contain a gap. Shorten the interval or select Needs follow-up')
        return len(records)

    def save(self, doc):
        count = self.validate(doc)
        stamp = datetime.now(timezone.utc)
        snapshot = dict(doc, server_saved_at=stamp.isoformat(), app_version='3.0.1')
        self.review_dir.mkdir(parents=True, exist_ok=True)
        dest = self.review_dir / ('review_' + stamp.strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid.uuid4().hex[:8] + '.json')
        pending = dest.with_suffix('.pending')
        with pending.open('x', encoding='utf-8') as stream:
            json.dump(snapshot, stream, ensure_ascii=False, indent=2, allow_nan=False)
        pending.rename(dest)
        return dict(ok=True, records=count, revision=dest.name,
                    path=str(dest.relative_to(self.boundary)) if dest.is_relative_to(self.boundary) else str(dest))


def discover(workspace, package, explicit=None):
    workspace, package = Path(workspace).resolve(), Path(package).resolve()
    projects = {}
    for path in sorted((workspace / 'data/review_projects').glob('*/project.json')):
        project = Project(path, workspace, workspace / 'data/annotations/trajectory_reviews')
        projects[project.id] = project
    if explicit:
        path = Path(explicit).resolve()
        if path.is_dir():
            path = path / 'project.json'
        boundary = workspace if path.is_relative_to(workspace) else path.parent
        project = Project(path, boundary, path.parent / 'reviews')
        projects[project.id] = project
    if not projects:
        path = package / 'examples/demo/project.json'
        if path.is_file():
            project = Project(path, path.parent, path.parent / 'reviews')
            projects[project.id] = project
    return projects
