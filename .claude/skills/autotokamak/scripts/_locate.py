# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Shared repo-locate + env-header helpers for autotokamak Skill scripts.

Every Skill wrapper imports from this module. The wrappers themselves never
`import autotokamak.*` — they dispatch via subprocess so the OFT_env process
singleton lives in a fresh child interpreter each time.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional


def _looks_like_autotokamak(root: Path) -> bool:
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return False
    try:
        text = pyproject.read_text(errors="replace")
    except OSError:
        return False
    return 'name = "autotokamak"' in text or 'name = "autokamak"' in text


def locate_root() -> Optional[Path]:
    """Resolution order: $AUTOTOKAMAK_ROOT → walk up from cwd → None."""
    env = os.environ.get("AUTOTOKAMAK_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if _looks_like_autotokamak(p):
            return p
    cwd = Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        if _looks_like_autotokamak(parent):
            return parent
    return None


def python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def print_env_header(root: Optional[Path]) -> None:
    """First line every wrapper prints. Machine-readable, sortable."""
    root_str = str(root) if root else "<not-found>"
    cwd_str = str(Path.cwd())
    print(
        f"autotokamak: root={root_str}  cwd={cwd_str}  python={python_version()}",
        flush=True,
    )


def read_only_advisory(reason: str = "no autotokamak checkout detected") -> None:
    """Emit the read-only-mode notice and exit 0. Wrappers call this when locate_root() is None."""
    print(f"autotokamak: read-only advisory mode — {reason}", flush=True)
    print(
        "  set AUTOTOKAMAK_ROOT to a valid checkout to enable actions.",
        flush=True,
    )
    print_json_summary({"mode": "advisory", "root": None, "action_taken": False})
    sys.exit(0)


def print_json_summary(summary: dict) -> None:
    """Emit a machine-parseable summary block. One line, prefixed with a sentinel."""
    print("===AUTOTOKAMAK-JSON===", flush=True)
    print(json.dumps(summary, indent=2, default=str), flush=True)
    print("===END-AUTOTOKAMAK-JSON===", flush=True)


def repo_python(root: Path) -> str:
    """Prefer the repo's venv Python if present; else the current interpreter."""
    venv_py = root / "venv" / "bin" / "python"
    return str(venv_py) if venv_py.exists() else sys.executable


def agent_env(root: Path) -> dict:
    """PYTHONPATH- and PATH-injected env for runners under agent.runners.*.

    Two things must happen for the nested ExecutionAgent to find autotokamak:

      1. PYTHONPATH → src/autotokamak so the runners import as
         `agent.runners.plan_execute_feedback` (not autotokamak.agent.runners.*).
      2. PATH prepended with <root>/venv/bin so when the ExecutionAgent's
         shell tool runs `python3 -c "import autotokamak"`, `python3`
         resolves to the venv's interpreter (which has autotokamak
         installed), NOT the system python that the user's shell defaulted
         to. Without this, the nested agent spirals inventing preflight,
         warmup, and env_capture scripts trying to "fix" a missing package
         that is actually installed one PATH entry away.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src" / "autotokamak")
    venv_bin = root / "venv" / "bin"
    if venv_bin.is_dir():
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
        env["VIRTUAL_ENV"] = str(root / "venv")
    return env
