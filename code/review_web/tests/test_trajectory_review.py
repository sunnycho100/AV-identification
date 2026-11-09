"""Real HTTP round trips against a temporary synthetic project only."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

PACKAGE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('trajectory_test_server', PACKAGE / 'server.py')
web = importlib.util.module_from_spec(spec)
spec.loader.exec_module(web)


class QuietHandler(web.Handler):
    def log_message(self, *args):
        pass


class TrajectoryReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Keep generated revision paths short even in deeply nested checkouts.
        cls.temp_base = Path(tempfile.gettempdir()).resolve()
        cls.temp = tempfile.TemporaryDirectory(prefix='roadside-trajectory-', dir=cls.temp_base)
        cls.root = Path(cls.temp.name).resolve()
        assert cls.root.is_relative_to(cls.temp_base)
        dest = cls.root / 'data/review_projects/synthetic_demo'
        shutil.copytree(PACKAGE / 'examples/demo', dest, ignore=shutil.ignore_patterns('reviews'))
        web.ROOT = cls.root
        web.PACK = cls.root / 'experiments' / web.PACK_NAME
        web.PROJECTS = web.discover(cls.root, PACKAGE)
        cls.project = web.PROJECTS['synthetic_demo']
        case = cls.project.doc['cases'][1]
        cls.doc = dict(schema='trajectory-review/v1', project_id='synthetic_demo', records=[dict(
            clip=case['clip'], record_id='TEST_SYNTHETIC', reviewer='user',
            source_video_sha256=case['source_video_sha256'], frame_data_sha256=case['frame_data_sha256'],
            start_frame=0, end_frame=30, start_s=0, end_s=3,
            trim=dict(keep_start_frame=0, keep_end_frame=40),
            anchors=[dict(frame=f, bbox=[80, 220, 165, 270], source='manual') for f in [0, 15, 30]],
            excluded_ranges=[dict(start_frame=41, end_frame=50, kind='image_freeze')],
            issues=['image_freeze'], identity_continuous=True, decision='ready_for_retracking', note='TEST ONLY')])
        cls.http = web.ThreadingHTTPServer(('127.0.0.1', 0), QuietHandler)
        cls.origin = f'http://127.0.0.1:{cls.http.server_port}'
        threading.Thread(target=cls.http.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown(); cls.http.server_close()
        assert cls.root.is_relative_to(cls.temp_base)
        cls.temp.cleanup()

    def fetch(self, path, doc=None, headers=None):
        hdr = headers or {}
        if doc is not None:
            hdr.update({'Content-Type':'application/json', 'Origin':self.origin})
        return urlopen(Request(self.origin+path, headers=hdr, data=json.dumps(doc).encode() if doc is not None else None), timeout=5)

    def test_save_append_reload_import_and_revision(self):
        with self.fetch('/api/projects/synthetic_demo/validate', self.doc) as r:
            self.assertTrue(json.load(r)['ok'])
        revisions = []
        for _ in range(2):
            with self.fetch('/api/projects/synthetic_demo/reviews', self.doc) as r:
                revisions.append(json.load(r)['revision'])
        self.assertNotEqual(*revisions)
        with self.fetch('/api/projects/synthetic_demo/reviews') as r:
            saved = json.load(r)
        self.assertEqual(saved['records'], self.doc['records'])
        self.assertGreaterEqual(len(saved['revisions']), 2)
        with self.fetch('/api/projects/synthetic_demo/reviews/'+revisions[0]) as r:
            self.assertEqual(json.load(r)['records'], self.doc['records'])

    def test_all_frames_and_media_range(self):
        with self.fetch('/api/projects') as r:
            self.assertEqual(len(json.load(r)['projects'][0]['cases']), 2)
        with self.fetch('/api/projects/synthetic_demo/clips/DEMO_FROZEN_TAIL') as r:
            self.assertEqual(len(json.load(r)['frames']), 51)
        with self.fetch('/media/synthetic_demo/DEMO_FROZEN_TAIL/frame/50', headers={'Range':'bytes=0-19'}) as r:
            self.assertEqual(r.status, 206)
            self.assertEqual(len(r.read()), 20)

    def test_invalid_interval_identity_source_and_bounds(self):
        changes = [dict(end_frame=20,end_s=2),dict(end_s=3.1),dict(identity_continuous=False),
            dict(source_video_sha256='stale'),dict(anchors=[]),dict(trim=dict(keep_start_frame=1,keep_end_frame=40)),
            dict(excluded_ranges=[dict(start_frame=25,end_frame=40,kind='image_freeze')]),
            dict(anchors=[dict(frame=0,bbox=[-1,2,20,30],source='manual')]),
            dict(anchors=[dict(frame=0,bbox=[0,2,float('nan'),30],source='manual')])]
        for change in changes:
            with self.subTest(change=change):
                bad=copy.deepcopy(self.doc);bad['records'][0].update(change)
                with self.assertRaises(HTTPError) as error:
                    self.fetch('/api/projects/synthetic_demo/validate',bad)
                self.assertEqual(error.exception.code,400)

    def test_native_pts_gap_cannot_count_as_observed_seconds(self):
        case=self.project.doc['cases'][1];path=self.project.root/case['frame_data'];original=path.read_bytes();original_hash=case['frame_data_sha256']
        try:
            data=json.loads(original)
            for frame in data['frames'][20:]:frame['t_s']+=1
            path.write_text(json.dumps(data),encoding='utf-8')
            case['frame_data_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            doc=copy.deepcopy(self.doc);doc['records'][0].update(frame_data_sha256=case['frame_data_sha256'],end_s=4)
            with self.assertRaises(HTTPError):self.fetch('/api/projects/synthetic_demo/validate',doc)
            doc['records'][0]['decision']='needs_followup'
            with self.fetch('/api/projects/synthetic_demo/validate',doc) as r:self.assertTrue(json.load(r)['ok'])
        finally:
            path.write_bytes(original);case['frame_data_sha256']=original_hash

    def test_trim_conflict_reports_both_ranges(self):
        doc=copy.deepcopy(self.doc)
        doc['records'][0]['trim']['keep_end_frame']=20
        with self.assertRaises(HTTPError) as error:
            self.fetch('/api/projects/synthetic_demo/validate',doc)
        message=json.load(error.exception)['error']
        self.assertIn('0–30',message)
        self.assertIn('0–20',message)
        doc['records'][0]['trim']['keep_end_frame']=2.5
        with self.assertRaises(HTTPError) as error:
            self.fetch('/api/projects/synthetic_demo/validate',doc)
        self.assertIn('frame numbers, not seconds',json.load(error.exception)['error'])

    def test_path_traversal_and_missing_media(self):
        for path in ['/media/synthetic_demo/DEMO_FROZEN_TAIL/frame/999', '/media/synthetic_demo/DEMO_FROZEN_TAIL/video',
                     '/app/%2e%2e/server.py','/api/projects/synthetic_demo/reviews/%2e%2e']:
            with self.assertRaises(HTTPError) as error:self.fetch(path)
            self.assertEqual(error.exception.code,404)


if __name__=='__main__':unittest.main(verbosity=2)
