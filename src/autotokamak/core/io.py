# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Atomic file I/O for run artifacts.

Why atomic: a 1000-case sweep that crashes mid-write leaves a corrupted .npz on
disk and a dataloader that explodes hours later. The atomic-write pattern is
``write to temp file in same dir → fsync → os.replace``, which means the
destination either contains the complete previous file or the complete new file —
never a partial one.

Also includes a unified output-directory helper so all runners produce the same
``outputs/<run_id>/`` layout (instead of the three diverging schemes that
existed pre-refactor).
"""

from __future__ import annotations

import datetime as _dt
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def mkdir_p(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write a text file atomically (best-effort)."""
    mkdir_p(path.parent)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=str(path.parent), encoding=encoding
    ) as tf:
        tf.write(text)
        tf.flush()
        os.fsync(tf.fileno())
        tmp_name = tf.name
    os.replace(tmp_name, path)


def atomic_savez(path: Path, **arrays: Any) -> None:
    """Write a NumPy ``.npz`` file atomically (best-effort)."""
    mkdir_p(path.parent)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(path.parent)) as tf:
        tmp_name = tf.name
    try:
        np.savez(tmp_name, **arrays)
        # numpy appends .npz if not present; handle both possibilities
        tmp_npz = Path(tmp_name)
        if not str(tmp_npz).endswith(".npz") and Path(str(tmp_npz) + ".npz").exists():
            tmp_npz = Path(str(tmp_npz) + ".npz")
        os.replace(tmp_npz, path)
    finally:
        # Clean up if something went wrong
        for cand in [Path(tmp_name), Path(str(tmp_name) + ".npz")]:
            if cand.exists() and cand != path:
                try:
                    cand.unlink()
                except Exception:  # noqa: BLE001
                    pass


def assert_nonempty_file(path: Path, *, min_bytes: int = 16) -> None:
    st = path.stat()
    if st.st_size < min_bytes:
        raise RuntimeError(
            f"Output file appears empty/corrupt (size={st.st_size} bytes): {path}"
        )


def utc_run_id() -> str:
    """A run id like ``20260613T235959Z`` — sortable, unique to the second."""
    return _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def unified_output_dir(out_base: Path | str, run_id: str | None = None) -> Path:
    """Standard layout: ``<out_base>/<run_id>/``. Creates it if missing."""
    base = Path(out_base)
    run = run_id or utc_run_id()
    out = base / run
    mkdir_p(out)
    return out


__all__ = [
    "mkdir_p",
    "atomic_write_text",
    "atomic_savez",
    "assert_nonempty_file",
    "utc_run_id",
    "unified_output_dir",
]
