"""Resolve workspace paths without depending on a particular host or working directory."""

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Workspace:
    """The code/ directory, containing configs, modules, and local experiment assets."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())

    def path(self, value: str | Path) -> Path:
        """Resolve an absolute path or a path relative to code/."""
        path = Path(str(value).replace("\\", "/")).expanduser()
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()

    @classmethod
    def discover(cls, explicit: str | Path | None = None) -> "Workspace":
        """Use an explicit path, ROADSIDE_WORKSPACE, or a surrounding checkout."""
        location = explicit or os.environ.get("ROADSIDE_WORKSPACE")
        if location:
            root = Path(location).expanduser().resolve()
            if (root / "code/configs/workspace.json").is_file():
                root = root / "code"
            if not (root / "configs/workspace.json").is_file():
                raise ValueError(f"Not a code workspace: {root}. Supply --workspace PATH_TO_CODE.")
            return cls(root)
        for parent in (Path.cwd(), *Path.cwd().parents):
            for candidate in (parent, parent / "code"):
                if (candidate / "configs/workspace.json").is_file():
                    return cls(candidate)
        raise ValueError("Workspace not found. Supply --workspace PATH_TO_CODE.")
