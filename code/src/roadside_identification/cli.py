"""Cross-platform commands; heavy dependencies are loaded only when needed."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from .io import read_json
from .workspace import Workspace


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Roadside trajectory research workspace")
    result.add_argument("--workspace", type=Path, help="code/ directory or parent checkout")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Report dependencies and assets without loading models")
    classify = commands.add_parser("classify", help="Run the frozen source classifier")
    classify.add_argument("--config", default="configs/classifier_default.json")
    classify.add_argument("--dataset")
    classify.add_argument("--metadata")
    classify.add_argument("--output")
    demo = commands.add_parser(
        "demo", help="Create synthetic inputs, without field data or weights"
    )
    demo.add_argument("--output", default="outputs/demo")
    review = commands.add_parser("review", help="Start the review product")
    review.add_argument("--port", type=int, default=8765)
    review.add_argument("--project", type=Path)
    commands.add_parser("test", help="Run maintained core and review-service tests")
    release = commands.add_parser(
        "release", help="Export an audited source-only GitHub tree and ZIP"
    )
    release.add_argument("--output", default="outputs/github_release")
    commands.add_parser("inventory", help="Rebuild the code navigation index")
    return result


def make_demo(workspace: Workspace, output: str) -> dict:
    import numpy as np
    import pandas as pd
    from .features import motion_features

    destination = workspace.path(output)
    destination.mkdir(parents=True, exist_ok=True)
    values, masks, valid, metadata = [], [], [], []
    for number, length in enumerate((26, 21, 11)):
        time = np.arange(length) * 0.1
        ego = 12 + np.sin(time) * 0.3
        available = np.ones(length, dtype=bool)
        if number == 1:
            available[:] = False
        if number == 2:
            available[length // 2 :] = False
        x, mask = motion_features(ego, ego + 0.2, 20 + time * 0.2, available)
        values.append(np.pad(x, ((0, 26 - length), (0, 0))))
        masks.append(np.pad(mask, ((0, 26 - length), (0, 0))))
        valid.append(np.arange(26) < length)
        metadata.append(
            {
                "clip": f"SYNTHETIC_{number + 1}",
                "samples": length,
                "duration_s": (length - 1) / 10,
                "synthetic": True,
            }
        )
    np.savez_compressed(
        destination / "dataset.npz",
        features=np.stack(values),
        feature_mask=np.stack(masks),
        valid_tokens=np.stack(valid),
    )
    pd.DataFrame(metadata).to_csv(destination / "windows.csv", index=False)
    return {
        "output": str(destination),
        "synthetic_windows": 3,
        "labels": "omitted; interface demonstration, not an accuracy benchmark",
    }


def dispatch(args: argparse.Namespace, workspace: Workspace) -> dict | int:
    if args.command == "doctor":
        config = read_json(workspace.path("configs/classifier_default.json"))
        return {
            "python": sys.version.split()[0],
            "workspace": str(workspace.root),
            "packages": {
                name: importlib.util.find_spec(name) is not None
                for name in ("numpy", "scipy", "pandas", "torch", "mmcv", "mmdet")
            },
            "assets": {
                "classifier_checkpoint": workspace.path(config["checkpoint"]).is_file(),
                "default_inputs": workspace.path(config["default_dataset"]).is_file(),
                "ab3dmot": workspace.path("trajectory/third_party/AB3DMOT").is_dir(),
            },
            "note": "Source-only checkouts omit field data, weights and AB3DMOT.",
        }
    if args.command == "classify":
        import torch
        from .classifier import classify_dataset

        torch.set_num_threads(2)
        config = read_json(workspace.path(args.config))
        report = classify_dataset(
            workspace,
            config,
            args.dataset or config["default_dataset"],
            args.metadata or config["default_metadata"],
            args.output or config["output"] + "/field",
        )
        return report.get("metrics", {"status": "classified", "trained_parameters": 0})
    if args.command == "demo":
        return make_demo(workspace, args.output)
    if args.command == "review":
        command = [
            sys.executable,
            str(workspace.path("review_web/server.py")),
            "--workspace",
            str(workspace.root),
            "--port",
            str(args.port),
        ]
        if args.project:
            command.extend(["--project", str(args.project.resolve())])
        return subprocess.call(command)
    if args.command == "test":
        env = dict(os.environ)
        env["PYTHONPATH"] = str(workspace.path("src"))
        for folder in ("tests", "review_web/tests"):
            result = subprocess.call(
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    str(workspace.path(folder)),
                    "-v",
                ],
                env=env,
            )
            if result:
                return result
        return 0
    if args.command in ("release", "inventory"):
        script = "export_repository.py" if args.command == "release" else "index_code.py"
        command = [sys.executable, str(workspace.path("tools") / script)]
        if args.command == "release":
            command.extend(["--output", str(workspace.path(args.output))])
        return subprocess.call(command)
    raise ValueError(f"Unsupported command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = dispatch(args, Workspace.discover(args.workspace))
    except (FileNotFoundError, ValueError, ImportError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    if isinstance(result, int):
        return result
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
