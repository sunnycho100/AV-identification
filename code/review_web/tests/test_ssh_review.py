"""Run with Python -S: only the standard library is required.

Uses an isolated workspace and a real TCP forwarder with a different local port.
No real user annotation or source video is modified.
"""
import copy
import importlib.util
import json
from pathlib import Path
import select
import socket
import socketserver
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

PACKAGE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('review_server', PACKAGE / 'server.py')
web = importlib.util.module_from_spec(spec)
spec.loader.exec_module(web)


class Forwarder(socketserver.BaseRequestHandler):
    def handle(self):
        with socket.create_connection(self.server.destination, timeout=5) as remote:
            remote.settimeout(None)
            peers = [self.request, remote]
            while True:
                readable, _, _ = select.select(peers, [], [], 10)
                if not readable:
                    return
                for source in readable:
                    try:
                        chunk = source.recv(262144)
                    except (ConnectionAbortedError, ConnectionResetError):
                        return
                    if not chunk:
                        return
                    try:
                        (remote if source is self.request else self.request).sendall(chunk)
                    except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                        return


class ProxyServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


class QuietHandler(web.Handler):
    def log_message(self, *args):
        pass


class SSHReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Keep generated revision paths short even in deeply nested checkouts.
        cls.temp_base = Path(tempfile.gettempdir()).resolve()
        cls.temp = tempfile.TemporaryDirectory(prefix='roadside-ssh-', dir=cls.temp_base)
        cls.root = Path(cls.temp.name).resolve()
        assert cls.root.is_relative_to(cls.temp_base)
        web.ROOT = cls.root
        web.PACK = cls.root / 'experiments' / web.PACK_NAME
        web.PACK.mkdir(parents=True)
        case = dict(clip='TEST_ONLY', tag='fixture', frames=3,
                    source_video_sha256='fixture-video', tracks_sha256='fixture-tracks')
        cls.manifest = dict(pack_id=web.PACK_NAME, cases=[case])
        (web.PACK / 'manifest.json').write_text(json.dumps(cls.manifest))
        (web.PACK / 'manifest.js').write_text('window.REVIEW_MANIFEST={};')
        times = cls.root / 'trajectory/data/camera-data/TEST_ONLY/frame_times.json'
        times.parent.mkdir(parents=True)
        times.write_text(json.dumps(dict(frames=[dict(frame=i, t_s=i*.1) for i in range(3)])))
        tracks = cls.root / 'trajectory/outputs/trajectories/TEST_ONLY_fixture/tracks.json'
        tracks.parent.mkdir(parents=True)
        tracks.write_text(json.dumps(dict(tracks={'1': [], '2': []})))
        cls.video = bytes(range(256)) * 128
        video_path = cls.root / 'data/field/fixture.mp4'
        video_path.parent.mkdir(parents=True)
        video_path.write_bytes(cls.video)
        cls.doc = dict(schema='field-leader-decisions/v1', pack_id=web.PACK_NAME, records=[dict(
            record_id='TEST_ONLY', clip='TEST_ONLY', source_video_sha256='fixture-video',
            tracks_sha256='fixture-tracks', start_frame=0, end_frame=2, start_s=0., end_s=.2,
            target=dict(type='track', id='1'), leader=dict(type='track', id='2'),
            decision='confirm_pair', leader_state='present', reviewer='user', note='synthetic test, not research data')])
        cls.http = web.ThreadingHTTPServer(('127.0.0.1', 0), QuietHandler)
        threading.Thread(target=cls.http.serve_forever, daemon=True).start()
        cls.proxy = ProxyServer(('127.0.0.1', 0), Forwarder)
        cls.proxy.destination = cls.http.server_address
        threading.Thread(target=cls.proxy.serve_forever, daemon=True).start()
        cls.port = cls.proxy.server_address[1]
        cls.origin = f'http://localhost:{cls.port}'

    @classmethod
    def tearDownClass(cls):
        cls.proxy.shutdown(); cls.proxy.server_close()
        cls.http.shutdown(); cls.http.server_close()
        assert cls.root.is_relative_to(cls.temp_base)
        cls.temp.cleanup()

    def fetch(self, path, headers=None, method='GET', data=None):
        return urlopen(Request(self.origin + path, headers=headers or {}, method=method, data=data), timeout=5)

    def post(self, doc, origin=None):
        return self.fetch('/api/leader-review/save', method='POST', data=json.dumps(doc).encode(),
                          headers={'Content-Type': 'application/json', 'Origin': origin or self.origin})

    def test_root_and_health_through_different_port(self):
        self.assertNotEqual(self.port, self.http.server_port)
        with self.fetch('/') as response:
            self.assertEqual(response.status, 200)
            self.assertTrue(response.url.endswith('/app/'))
            self.assertIn('Roadside trajectory review', response.read().decode())
        with self.fetch(web.APP_PATH) as response:
            self.assertIn('一起确认前车', response.read().decode())
        with self.fetch('/healthz') as response:
            self.assertEqual(json.load(response)['version'], web.VERSION)

    def test_forwarded_localhost_save_preserves_history(self):
        saved = []
        for _ in range(2):
            with self.post(self.doc) as response:
                result = json.load(response)
            self.assertTrue(result['ok'])
            saved.append(self.root / result['path'])
        self.assertNotEqual(saved[0], saved[1])
        self.assertEqual(json.loads(saved[0].read_text())['records'], self.doc['records'])

    def test_loopback_aliases_and_origin_rejection(self):
        for host in ('localhost:19000', '127.0.0.1:21000', '[::1]:32000'):
            self.assertTrue(web.same_local_origin('http://' + host, host))
        for origin, host in [('http://localhost:19000', 'localhost:8765'),
                             ('http://unrelated.example', 'unrelated.example'),
                             ('null', 'localhost:8765')]:
            self.assertFalse(web.same_local_origin(origin, host))
        with self.assertRaises(HTTPError) as error:
            self.post(self.doc, 'http://unrelated.example')
        self.assertEqual(error.exception.code, 400)

    def test_no_leader_and_invisible_remain_distinct(self):
        for decision, state in [('no_leader', 'absent'), ('leader_not_visible', 'not_visible')]:
            doc = copy.deepcopy(self.doc)
            row = doc['records'][0]
            row.update(decision=decision, leader=None)
            row.pop('leader_state')  # Older records may lack the additive field.
            with self.post(doc) as response:
                result = json.load(response)
            stored = json.loads((self.root / result['path']).read_text())['records'][0]
            self.assertEqual(stored['leader_state'], state)
            self.assertIsNone(stored['leader'])
            self.assertEqual(stored['target'], row['target'])
        for field, value in [('leader_state', 'present'), ('leader', dict(type='track', id='2')), ('target', None)]:
            bad = copy.deepcopy(self.doc)
            bad['records'][0].update(decision='no_leader', leader=None, leader_state='absent')
            bad['records'][0][field] = value
            with self.assertRaises(HTTPError) as error:
                self.post(bad)
            self.assertEqual(error.exception.code, 400)

    def test_video_ranges_and_head(self):
        for header, expected in [('bytes=100-299', self.video[100:300]),
                                 ('bytes=-31', self.video[-31:]),
                                 (f'bytes={len(self.video)-10}-', self.video[-10:])]:
            with self.fetch('/data/field/fixture.mp4', {'Range': header}) as response:
                self.assertEqual(response.status, 206)
                self.assertEqual(response.read(), expected)
                self.assertEqual(int(response.headers['Content-Length']), len(expected))
        with self.fetch('/data/field/fixture.mp4', {'Range': 'bytes=100-299'}, 'HEAD') as response:
            self.assertEqual(response.read(), b'')
            self.assertEqual(response.headers['Content-Range'], f'bytes 100-299/{len(self.video)}')
        for value in ('bytes=999999-', 'bytes=5-2', 'bytes=0-1,3-4'):
            with self.assertRaises(HTTPError) as error:
                self.fetch('/data/field/fixture.mp4', {'Range': value})
            self.assertEqual(error.exception.code, 416)

    def test_invalid_annotations_and_path_rejected(self):
        for field, value in [('end_frame', -1), ('source_video_sha256', 'stale'),
                             ('leader', dict(type='track', id='1'))]:
            bad = copy.deepcopy(self.doc)
            bad['records'][0][field] = value
            with self.assertRaises(HTTPError) as error:
                self.post(bad)
            self.assertEqual(error.exception.code, 400)
        for path in ('/review_web/server.py', f'/experiments/{web.PACK_NAME}/%2e%2e/manifest.json'):
            with self.assertRaises(HTTPError) as error:
                self.fetch(path)
            self.assertEqual(error.exception.code, 404)


if __name__ == '__main__':
    unittest.main(verbosity=2)
