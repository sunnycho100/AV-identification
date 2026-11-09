"""The publication boundary must exclude private data and restricted dependencies."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

path = Path(__file__).resolve().parents[1] / "tools/export_repository.py"
spec = importlib.util.spec_from_file_location("release_policy", path)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


class ReleaseTests(unittest.TestCase):
    def test_private_payloads_excluded(self):
        for name in (
            "code/data/field/video.mp4",
            "code/data/annotations/review.json",
            "todd_se_extension_reviews.json",
            "code/experiments/test/predictions.csv",
            "code/trajectory/third_party/AB3DMOT/AB3DMOT_libs/model.py",
            "code/trajectory/checkpoints/model.ckpt",
            "code/.venv-bev/pyvenv.cfg",
            "code/configs/field_full_manual_seeds.json",
            "code/catalogs/field_clips.csv",
            "code/classification/trajectory-avlike-transformer/data/prepared_3s/windows.csv",
            "../outside.py",
            "paper/Manuscript.pdf",
            "paper/Manuscript.tex",
            "paper/reference.bib",
            "paper/figs/example.pdf",
            "paper/thumbnails/cas-email.jpeg",
            "code/src/example.egg-info/SOURCES.txt",
            "code/src/example.dist-info/METADATA",
            "paper/thumbnails/private_frame.jpeg",
            "code/docs/INTEGRATION.md",
        ):
            with self.subTest(path=name):
                self.assertFalse(policy.included(name))

    def test_required_sources_included(self):
        for name in (
            "README.md",
            "LICENSE",
            ".github/workflows/tests.yml",
            "code/manage.py",
            "code/src/roadside_identification/classifier.py",
            "code/trajectory/ops/pool.cu",
            "code/review_web/examples/demo/project.json",
            "code/review_web/examples/demo/frames/demo/000.svg",
            "code/configs/classifier_default.json",
            "code/docs/GETTING_STARTED.md",
            "code/scripts/publication/build_table.py",
            "code/classification/trajectory-avlike-transformer/data/prepare_highspeed_data.py",
        ):
            with self.subTest(path=name):
                self.assertTrue(policy.included(name))

    def test_export_contains_only_selected_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "project"
            for name in (
                "README.md",
                "code/manage.py",
                "paper/thumbnails/cas-email.jpeg",
                "paper/Manuscript.tex",
                "code/src/example.egg-info/SOURCES.txt",
                "code/src/example.dist-info/entry_points.txt",
                "code/scripts/publication/build_table.py",
                "code/data/field/private.csv",
                "code/trajectory/third_party/AB3DMOT/private.py",
            ):
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("example\n")
            report = policy.export(Path(folder) / "output", root)
            tree = Path(report["source_tree"])
            self.assertTrue((tree / "code/manage.py").is_file())
            self.assertFalse((tree / "paper").exists())
            self.assertFalse((tree / "code/src/example.egg-info").exists())
            self.assertFalse((tree / "code/src/example.dist-info").exists())
            self.assertTrue((tree / "code/scripts/publication/build_table.py").is_file())
            self.assertFalse((tree / "code/data").exists())
            self.assertFalse((tree / "code/trajectory/third_party/AB3DMOT").exists())
            self.assertFalse(report["uploaded"])


if __name__ == "__main__":
    unittest.main()
