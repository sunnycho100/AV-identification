"""Export an allowlisted source tree for GitHub; never traverse local data/runtime roots."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]
ROOT_FILES = {
    "README.md",
    "LICENSE",
    "THIRD_PARTY.md",
    "CONTRIBUTING.md",
    "pyproject.toml",
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
    "identification.code-workspace",
}
CODE_DIRECTORIES = {
    "src",
    "tests",
    "tools",
    "scripts",
    "configs",
    "docs",
    "environments",
    "trajectory",
    "classification",
    "review_web",
}
SKIP_PARTS = {
    "__pycache__",
    "node_modules",
    ".git",
    ".test-runtime",
    ".ruff_cache",
    ".pytest_cache",
    "reviews",
    "build",
    "dist",
}
PRIVATE_PREFIXES = (
    "code/trajectory/Camera data",
    "code/trajectory/data",
    "code/trajectory/outputs",
    "code/trajectory/checkpoints",
    "code/trajectory/third_party/AB3DMOT",
    "code/classification/trajectory-avlike-transformer/artifacts",
    "code/classification/trajectory-avlike-transformer/data_records",
    "code/classification/trajectory-avlike-transformer/paper_results",
    "code/docs/migration",
    "code/docs/classification_migration",
    "code/docs/history",
    "paper",
)
PUBLIC_DOCS = {
    "ARCHITECTURE.md",
    "CODE_INDEX.md",
    "DATA_CONTRACTS.md",
    "ENVIRONMENTS.md",
    "GETTING_STARTED.md",
    "RELEASE.md",
    "RESULTS_OVERVIEW.md",
    "STABILIZATION_HANDOFF.md",
    "source_index.json",
}
PUBLIC_CONFIGS = {
    "classifier_default.json",
    "classifier.example.json",
    "workspace.json",
    "environments.json",
    "field_todd_complete.json",
    "field_todd_3s.json",
    "field_todd_variable.json",
}
TEXT_SUFFIXES = {
    ".py",
    ".pyi",
    ".md",
    ".txt",
    ".json",
    ".toml",
    ".yml",
    ".yaml",
    ".ps1",
    ".sh",
    ".html",
    ".css",
    ".js",
    ".svg",
    ".cu",
    ".cpp",
    ".c",
    ".h",
    ".cuh",
    ".tex",
    ".bib",
    ".bbl",
    ".bst",
    ".sty",
    ".cls",
    ".in",
}
TEXT_NAMES = {
    "LICENSE",
    "LICENSE.BEVHeight",
    "NOTICE",
    "Makefile",
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
}


def excluded_part(part: str) -> bool:
    return part in SKIP_PARTS or part.endswith((".egg-info", ".dist-info"))


def beneath(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix + "/")


def included(relative: str) -> bool:
    """The release boundary; tested independently with representative private paths."""
    relative = relative.replace("\\", "/")
    path = PurePosixPath(relative)
    if ".." in path.parts or path.is_absolute() or any(excluded_part(p) for p in path.parts):
        return False
    if any(beneath(relative, p) for p in PRIVATE_PREFIXES):
        return False
    if relative in ROOT_FILES or relative == "code/manage.py" or relative == "code/run.ps1":
        return True
    if relative == "code/docs/PROJECT_HISTORY.md":
        return False
    if beneath(relative, "code/docs") and path.name not in PUBLIC_DOCS:
        return False
    if beneath(relative, "code/configs") and path.name not in PUBLIC_CONFIGS:
        return False
    data_prefix = "code/classification/trajectory-avlike-transformer/data"
    if beneath(relative, data_prefix) and (
        path.parent.as_posix() != data_prefix or path.suffix != ".py"
    ):
        return False
    anycalib = "code/trajectory/third_party/AnyCalib"
    if beneath(relative, anycalib):
        remainder = relative[len(anycalib) + 1 :]
        if not (
            remainder.startswith("anycalib/")
            or remainder in {"LICENSE", "README.md", "pyproject.toml"}
        ):
            return False
    permitted_root = (
        relative.startswith(".github/")
        or any(relative.startswith(f"code/{name}/") for name in CODE_DIRECTORIES)
    )
    if not permitted_root:
        return False
    return path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_NAMES


def candidates(root: Path):
    """Avoid descending into environments, datasets, outputs or excluded dependencies."""
    for name in sorted(ROOT_FILES | {"code/manage.py", "code/run.ps1"}):
        path = root / name
        if path.is_file():
            yield path
    for name in [".github", *(f"code/{d}" for d in sorted(CODE_DIRECTORIES))]:
        start = root / name
        if not start.exists():
            continue
        for directory, folders, files in os.walk(start, followlinks=False):
            current = Path(directory)
            folders[:] = [
                folder
                for folder in folders
                if not excluded_part(folder)
                and not (current / folder).is_symlink()
                and not any(
                    beneath((current / folder).relative_to(root).as_posix(), p)
                    for p in PRIVATE_PREFIXES
                )
                and not (
                    current.name == "data" and "trajectory-avlike-transformer" in current.parts
                )
            ]
            for file in sorted(files):
                path = current / file
                if not path.is_symlink() and included(path.relative_to(root).as_posix()):
                    yield path


def audit(path: Path) -> list[str]:
    """Basic release guard, not a substitute for a dedicated secret scanner."""
    problems = []
    if path.stat().st_size > 20 * 1024 * 1024:
        problems.append("unexpected_file_over_20MiB")
    if path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_NAMES:
        text = path.read_text(encoding="utf-8", errors="replace")
        patterns = (
            r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
            r"\bgh[pousr]_[A-Za-z0-9]{30,}\b",
            r"\bsk-proj-[A-Za-z0-9_-]{40,}\b",
        )
        if any(re.search(pattern, text) for pattern in patterns):
            problems.append("possible_secret")
    return problems


def export(output: Path, root: Path = ROOT) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = output / stamp
    number = 1
    while destination.exists():
        destination = output / f"{stamp}_{number}"
        number += 1
    tree = destination / "roadside-identification"
    files = sorted(set(candidates(root)))
    findings = [
        {"path": p.relative_to(root).as_posix(), "issues": audit(p)} for p in files if audit(p)
    ]
    if findings:
        raise ValueError(f"Source export audit failed (paths only): {findings}")
    manifest = {}
    for source in files:
        relative = source.relative_to(root)
        target = tree / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest[relative.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    (tree / "SOURCE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    archive = destination / "roadside-identification-source.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
        for path in sorted(tree.rglob("*")):
            if path.is_file():
                package.write(path, "roadside-identification/" + path.relative_to(tree).as_posix())
    report = {
        "source_tree": str(tree),
        "zip": str(archive),
        "files": len(manifest),
        "bytes": archive.stat().st_size,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "audit": "PASS",
        "uploaded": False,
        "omitted": [
            "field video/GPS/annotations",
            "weights",
            "environments and binaries",
            "historical experiment outputs",
            "AB3DMOT source",
            "local catalogs",
            "manuscript files and publication figures",
            "generated package metadata",
        ],
    }
    (destination / "release_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (output / "latest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.output), indent=2))


if __name__ == "__main__":
    main()
