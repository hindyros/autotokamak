# provenance: Human/Claude-authored platform code (engineered, not agent-generated)
"""Static (zero-LLM) code metrics for agent-generated benchmark workspaces.

Reads every run under ``experiments/<tag>/<condition>/<run_id>/`` and measures
the code the agent wrote in its ``workspace/`` — WITHOUT executing it and
without any LLM. Complements the deliverable contract (did it work?) with
"what did the agent build and how" facts that are cheap, deterministic, and
directly comparable across cells:

  * volume/structure: file count, SLOC, functions/classes, max function length,
    branch count (cyclomatic-complexity proxy), docstring + comment coverage
  * dependencies: every top-level import; whether ``autotokamak`` is imported
    (library leverage at L2, contract-violation cross-check at L3); ML stack
    (sklearn/torch/gpytorch/optuna); LLM-in-the-loop detection (openai /
    anthropic / dspy / langchain imports, API URLs, agent-CLI subprocess calls)
  * reproducibility: literal seeds (np seeds, ``random_state=``), config files
  * hygiene: ruff error count (serious: E9/F821/F823) and total default-rule
    count, syntax errors, bare ``except``s, ``eval``/``exec`` use

Symlinks (task-provided, e.g. ``OpenFUSIONToolkit``) and generated artifacts
(``__pycache__``, data files) are excluded — only agent-authored text counts.

Outputs, per run: ``<run_dir>/eval/code_metrics.json`` (never touches the
agent's workspace). Per tag: ``experiments/<tag>/code_metrics.csv`` plus a
console table.

Usage:
    python tools/eval_code_metrics.py --tag matrix-v2-20260810
    python tools/eval_code_metrics.py --run experiments/<tag>/<cond>/<run_id>
    python tools/eval_code_metrics.py --tag <tag> --no-ruff   # skip lint pass

Dependencies: stdlib only (ruff optional, used via subprocess when on PATH
or importable as ``python -m ruff``).
"""
from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import re
import shutil
import subprocess
import sys
import tokenize
from dataclasses import dataclass, field
from pathlib import Path

# Directories never authored by the agent (task symlinks are skipped separately).
EXCLUDE_DIRS = {"__pycache__", ".git", ".venv", "venv", "node_modules",
                ".ruff_cache", ".pytest_cache", ".ipynb_checkpoints"}
# Text files larger than this are almost certainly generated dumps, not code.
MAX_CODE_BYTES = 400_000

LLM_IMPORTS = {"openai", "anthropic", "litellm", "dspy", "langchain",
               "langchain_openai", "langgraph", "google.generativeai", "genai",
               "mistralai", "cohere", "ollama", "claude_agent_sdk"}
LLM_TEXT_PATTERNS = [
    r"api\.openai\.com", r"api\.anthropic\.com", r"generativelanguage\.googleapis",
    r"\bclaude\s+-p\b", r"\bcursor-agent\b", r"chat\.completions", r"messages\.create",
]
ML_IMPORTS = {"sklearn", "torch", "gpytorch", "optuna", "xgboost", "lightgbm",
              "scipy", "skopt", "botorch"}
SEED_PATTERNS = [
    r"\bseed\s*[=(]\s*\d+", r"default_rng\(\s*\d+", r"random_state\s*=\s*\d+",
    r"np\.random\.seed\(\s*\d+", r"manual_seed\(\s*\d+", r"random\.seed\(\s*\d+",
]


@dataclass
class FileMetrics:
    path: str
    sloc: int = 0
    comment_lines: int = 0
    n_functions: int = 0
    n_classes: int = 0
    n_documented: int = 0        # defs/classes/module carrying a docstring
    n_documentable: int = 0
    max_function_sloc: int = 0
    branches: int = 0            # If/For/While/Try/With/BoolOp — complexity proxy
    bare_excepts: int = 0
    eval_exec: int = 0
    imports: set = field(default_factory=set)
    syntax_error: bool = False


def _iter_agent_files(workspace: Path) -> list[Path]:
    files: list[Path] = []
    for p in sorted(workspace.rglob("*")):
        if p.is_symlink() or not p.is_file():
            continue
        if any(part in EXCLUDE_DIRS or Path(part).is_symlink() for part in ()):
            continue
        rel = p.relative_to(workspace)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        # Skip anything under a symlinked dir (task-provided, e.g. OFT clone).
        if any((workspace / Path(*rel.parts[: i + 1])).is_symlink()
               for i in range(len(rel.parts) - 1)):
            continue
        files.append(p)
    return files


def _analyze_python(path: Path, rel: str) -> FileMetrics:
    fm = FileMetrics(path=rel)
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        fm.syntax_error = True
        return fm

    code_lines = [ln for ln in src.splitlines() if ln.strip()]
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
        comment_rows = {t.start[0] for t in toks if t.type == tokenize.COMMENT}
        fm.comment_lines = len(comment_rows)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    fm.sloc = len(code_lines) - fm.comment_lines

    try:
        tree = ast.parse(src)
    except SyntaxError:
        fm.syntax_error = True
        return fm

    fm.n_documentable = 1
    fm.n_documented = 1 if ast.get_docstring(tree) else 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            fm.n_documentable += 1
            if ast.get_docstring(node):
                fm.n_documented += 1
            if isinstance(node, ast.ClassDef):
                fm.n_classes += 1
            else:
                fm.n_functions += 1
                span = (getattr(node, "end_lineno", node.lineno) or node.lineno) - node.lineno + 1
                fm.max_function_sloc = max(fm.max_function_sloc, span)
        elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try,
                               ast.With, ast.BoolOp, ast.IfExp)):
            fm.branches += 1
        elif isinstance(node, ast.ExceptHandler) and node.type is None:
            fm.bare_excepts += 1
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id in {"eval", "exec"}):
            fm.eval_exec += 1
        elif isinstance(node, ast.Import):
            fm.imports.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            fm.imports.add(node.module.split(".")[0])
    return fm


def _autotokamak_submodules(src: str) -> set[str]:
    subs = set()
    for m in re.finditer(r"(?:from|import)\s+autotokamak\.(\w+)", src):
        subs.add(m.group(1))
    if re.search(r"(?:from|import)\s+autotokamak\b(?!\.)", src):
        subs.add("<top>")
    return subs


def _ruff_counts(py_files: list[Path]) -> dict | None:
    """Return {"serious": n, "total": n} or None when ruff is unavailable."""
    if not py_files:
        return {"serious": 0, "total": 0}
    ruff = shutil.which("ruff")
    base = [ruff] if ruff else [sys.executable, "-m", "ruff"]
    counts = {}
    for key, select in (("serious", "E9,F821,F823"), ("total", None)):
        cmd = base + ["check", "--output-format", "json", "--exit-zero",
                      "--isolated", "--no-cache"]
        if select:
            cmd += ["--select", select]
        cmd += [str(p) for p in py_files]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            counts[key] = len(json.loads(proc.stdout or "[]"))
        except Exception:  # noqa: BLE001 — ruff missing/broken: metric is optional
            return None
    return counts


def analyze_workspace(workspace: Path, *, run_ruff: bool = True) -> dict:
    files = _iter_agent_files(workspace)
    py_files = [p for p in files
                if p.suffix == ".py" and p.stat().st_size <= MAX_CODE_BYTES]
    doc_files = [p for p in files if p.suffix.lower() in {".md", ".rst", ".txt"}]

    per_file: list[FileMetrics] = []
    all_imports: set[str] = set()
    atk_submodules: set[str] = set()
    llm_hits: list[str] = []
    seed_hits = 0

    for p in py_files:
        rel = str(p.relative_to(workspace))
        fm = _analyze_python(p, rel)
        per_file.append(fm)
        all_imports |= fm.imports
        src = p.read_text(encoding="utf-8", errors="replace")
        atk_submodules |= _autotokamak_submodules(src)
        for pat in SEED_PATTERNS:
            seed_hits += len(re.findall(pat, src))
        for pat in LLM_TEXT_PATTERNS:
            if re.search(pat, src):
                llm_hits.append(f"{rel}: /{pat}/")

    llm_imports = sorted(all_imports & LLM_IMPORTS)
    readme = next((p for p in doc_files if p.name.lower().startswith("readme")), None)

    total = {
        "n_files_total": len(files),
        "n_py_files": len(py_files),
        "py_files": [f.path for f in per_file],
        "sloc": sum(f.sloc for f in per_file),
        "comment_lines": sum(f.comment_lines for f in per_file),
        "n_functions": sum(f.n_functions for f in per_file),
        "n_classes": sum(f.n_classes for f in per_file),
        "max_function_sloc": max((f.max_function_sloc for f in per_file), default=0),
        "branches": sum(f.branches for f in per_file),
        "docstring_coverage": round(
            sum(f.n_documented for f in per_file)
            / max(1, sum(f.n_documentable for f in per_file)), 3),
        "bare_excepts": sum(f.bare_excepts for f in per_file),
        "eval_exec_calls": sum(f.eval_exec for f in per_file),
        "syntax_errors": [f.path for f in per_file if f.syntax_error],
        "imports": sorted(all_imports),
        "imports_autotokamak": sorted(atk_submodules),
        "imports_ml": sorted(all_imports & ML_IMPORTS),
        "imports_llm": llm_imports,
        "llm_call_evidence": llm_hits,
        "llm_in_loop": bool(llm_imports or llm_hits),
        "seed_literals": seed_hits,
        "has_tests": any(Path(f.path).name.startswith("test_") for f in per_file),
        "readme_words": len(readme.read_text(errors="replace").split()) if readme else 0,
    }
    if run_ruff:
        total["ruff"] = _ruff_counts(py_files)
    return total


# ---------------------------------------------------------------------------

def _find_runs(tag_dir: Path) -> list[Path]:
    return sorted(p.parent for p in tag_dir.glob("*/*/result.json"))


CSV_COLS = ["condition", "run_id", "n_py_files", "sloc", "n_functions",
            "n_classes", "max_function_sloc", "branches", "docstring_coverage",
            "bare_excepts", "ruff_serious", "ruff_total", "seed_literals",
            "llm_in_loop", "autotokamak_imports", "readme_words"]


def _row(run_dir: Path, metrics: dict) -> dict:
    result = json.loads((run_dir / "result.json").read_text())
    ruff = metrics.get("ruff") or {}
    return {
        "condition": result.get("condition", run_dir.parent.name),
        "run_id": run_dir.name,
        "n_py_files": metrics["n_py_files"],
        "sloc": metrics["sloc"],
        "n_functions": metrics["n_functions"],
        "n_classes": metrics["n_classes"],
        "max_function_sloc": metrics["max_function_sloc"],
        "branches": metrics["branches"],
        "docstring_coverage": metrics["docstring_coverage"],
        "bare_excepts": metrics["bare_excepts"],
        "ruff_serious": ruff.get("serious", ""),
        "ruff_total": ruff.get("total", ""),
        "seed_literals": metrics["seed_literals"],
        "llm_in_loop": metrics["llm_in_loop"],
        "autotokamak_imports": ";".join(metrics["imports_autotokamak"]),
        "readme_words": metrics["readme_words"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tag", help="experiments/<tag> — analyze every run in it")
    g.add_argument("--run", help="one experiments/<tag>/<cond>/<run_id> dir")
    ap.add_argument("--experiments-dir", default=None,
                    help="override the experiments/ root (default: repo's)")
    ap.add_argument("--no-ruff", action="store_true", help="skip the ruff pass")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    exp_dir = Path(args.experiments_dir) if args.experiments_dir \
        else repo_root / "experiments"

    run_dirs = [Path(args.run)] if args.run else _find_runs(exp_dir / args.tag)
    if not run_dirs:
        print("No runs found.", file=sys.stderr)
        return 1

    rows = []
    for run_dir in run_dirs:
        workspace = run_dir / "workspace"
        if not workspace.is_dir():
            print(f"[skip] no workspace: {run_dir}", file=sys.stderr)
            continue
        metrics = analyze_workspace(workspace, run_ruff=not args.no_ruff)
        out = run_dir / "eval" / "code_metrics.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2))
        rows.append(_row(run_dir, metrics))
        print(f"[ok] {rows[-1]['condition']:>16s}/{run_dir.name}  "
              f"py={metrics['n_py_files']:>2d} sloc={metrics['sloc']:>5d}  → {out}")

    if args.tag and rows:
        csv_path = exp_dir / args.tag / "code_metrics.csv"
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_COLS)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {csv_path}")
        widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in CSV_COLS}
        print("  ".join(c.ljust(widths[c]) for c in CSV_COLS))
        for r in sorted(rows, key=lambda r: r["condition"]):
            print("  ".join(str(r[c]).ljust(widths[c]) for c in CSV_COLS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
