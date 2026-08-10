# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
from pathlib import Path
from types import SimpleNamespace as NS

import yaml


def _find_repo_root() -> Path:
    """Walk up from this file looking for pyproject.toml.

    Robust to where the package is installed/cloned. The previous version used
    three .parent calls, which landed at src/autotokamak/ (the package root),
    not the repo root — causing every workspace to be created under
    src/autotokamak/ instead of at the repo top level.
    """
    here = Path(__file__).resolve()
    for ancestor in [here, *here.parents]:
        if (ancestor / "pyproject.toml").is_file():
            return ancestor
    raise RuntimeError("Could not locate repo root (no pyproject.toml found above config.py).")


REPO_ROOT = _find_repo_root()


def load_config(path: str) -> NS:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("Top-level YAML must be a mapping/object.")
    return NS(**raw)


def resolve_workspace(workspace: str) -> Path:
    ws = Path(workspace)
    if not ws.is_absolute():
        ws = REPO_ROOT / ws
    return ws


def materialize_symlinks(workspace_path: Path, entries) -> None:
    """Create the symlinks listed under `symlinks:` in the prompt YAML.

    URSA's ExecutionAgent only supports a single symlinkdir dict and crashes
    when handed a list. We create the links ourselves before invoking the agent
    and pass symlinkdir=None to URSA. Missing sources are warned-and-skipped,
    not raised, so a run isn't blocked by an absent side-clone (./ursa or
    ./OpenFUSIONToolkit — both pip-installed anyway).
    """
    if not entries:
        return
    workspace_path.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        source = entry.get("source")
        dest = entry.get("dest")
        if not source or not dest:
            continue
        src = Path(source).expanduser()
        if not src.is_absolute():
            src = (REPO_ROOT / src).resolve()
        else:
            src = src.resolve()
        dst = workspace_path / dest
        if dst.exists() or dst.is_symlink():
            continue
        if not src.exists():
            print(f"WARNING: symlink source missing, skipping: {src}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src, target_is_directory=src.is_dir())
        print(f"Symlinked: {src} -> {dst}")
