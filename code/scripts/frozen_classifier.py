"""Compatibility entry for recorded experiments; maintained code lives in code/src/."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "classification/trajectory-avlike-transformer"
sys.path.insert(0, str(ROOT / "src"))
# Retained for recorded duration diagnostics importing data_protocol.
sys.path.insert(0, str(BASE / "inference_cnn"))
from roadside_identification.classifier import FrozenClassifier as _FrozenClassifier
from roadside_identification.cli import main as _main
from roadside_identification.io import save_json as save, sha256 as sha
from roadside_identification.metrics import classification_metrics as metrics
from roadside_identification.workspace import Workspace

CFG = json.loads((ROOT / "configs/classifier_default.json").read_text(encoding="utf-8"))

class FrozenClassifier(_FrozenClassifier):
    def __init__(self):
        super().__init__(Workspace(ROOT), CFG)

def main():
    return _main(["--workspace", str(ROOT), "classify", *sys.argv[1:]])

if __name__ == "__main__":
    raise SystemExit(main())
