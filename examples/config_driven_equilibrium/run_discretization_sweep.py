#!/usr/bin/env python3
# provenance: URSA-generated (URSA plan->execute agent)
"""Run multiple OFT TokaMaker equilibria from a sweep YAML.

This is a thin wrapper around `run_equilibrium_from_config.py` that:
  * expands a sweep file into per-case configs (base + overrides),
  * runs each case sequentially (as a subprocess),
  * preserves provenance by writing the exact resolved YAML config into each
    case output directory (handled by run_equilibrium_from_config.py as of this repo).

Constraints: does not require any non-stdlib dependencies.
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path


def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyYAML is required (should already be present in this environment)."
        ) from e
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dump_yaml(data: dict, path: Path) -> None:
    import yaml  # type: ignore

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _deep_merge(base, override):
    """Deep-merge override into base (dicts only; lists replaced except regions list).

    Special-case:
      mesh.regions: if override provides a list of regions with `name`, merge by name.
    """
    if isinstance(base, dict) and isinstance(override, dict):
        out = copy.deepcopy(base)
        for k, v in override.items():
            if k in out:
                if k == "mesh" and isinstance(v, dict) and isinstance(out.get(k), dict):
                    out[k] = _deep_merge(out[k], v)
                elif k == "regions" and isinstance(v, list) and isinstance(out.get(k), list):
                    # merge list of regions by name
                    out_regions = copy.deepcopy(out[k])
                    by_name = {r.get("name"): r for r in out_regions if isinstance(r, dict)}
                    for r_ov in v:
                        if not isinstance(r_ov, dict):
                            continue
                        nm = r_ov.get("name")
                        if nm in by_name:
                            by_name[nm] = _deep_merge(by_name[nm], r_ov)
                        else:
                            by_name[nm] = copy.deepcopy(r_ov)
                    # keep deterministic order: original order then any new names
                    seen = set()
                    merged_list = []
                    for r in out_regions:
                        nm = r.get("name") if isinstance(r, dict) else None
                        if nm and nm in by_name and nm not in seen:
                            merged_list.append(by_name[nm])
                            seen.add(nm)
                    for nm, r in by_name.items():
                        if nm not in seen:
                            merged_list.append(r)
                            seen.add(nm)
                    out[k] = merged_list
                else:
                    out[k] = _deep_merge(out[k], v)
            else:
                out[k] = copy.deepcopy(v)
        return out
    # for lists/scalars: override replaces
    return copy.deepcopy(override)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python run_discretization_sweep.py sweep.yaml", file=sys.stderr)
        return 2

    sweep_path = Path(argv[1]).resolve()
    sweep = _load_yaml(sweep_path)

    base_cfg_path = sweep.get("base_config")
    cases = sweep.get("cases")
    if not base_cfg_path or not cases:
        raise ValueError("Sweep YAML must contain base_config and cases.")

    base_cfg_path = (sweep_path.parent / base_cfg_path).resolve()
    base_cfg = _load_yaml(base_cfg_path)

    runner = (Path(__file__).parent / "run_equilibrium_from_config.py").resolve()
    if not runner.exists():
        raise FileNotFoundError(f"Runner not found: {runner}")

    workdir = Path.cwd()
    tmp_dir = workdir / "outputs" / "_sweep_tmp_configs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for i, case in enumerate(cases, start=1):
        if "config" in case:
            cfg_path = (sweep_path.parent / case["config"]).resolve()
            print(f"[{i}/{len(cases)}] Running config file: {cfg_path}")
        else:
            case_id = case.get("case_id", f"case_{i:03d}")
            overrides = case.get("overrides", {})
            cfg = _deep_merge(base_cfg, overrides)
            cfg["case_id"] = case_id
            cfg_path = tmp_dir / f"{case_id}.yaml"
            _dump_yaml(cfg, cfg_path)
            print(f"[{i}/{len(cases)}] Running case_id={case_id} (generated config)")

        env = os.environ.copy()
        # Keep runs deterministic-ish by fixing thread count if user didn't set.
        env.setdefault("OMP_NUM_THREADS", "1")

        cmd = [sys.executable, str(runner), str(cfg_path)]
        proc = subprocess.run(cmd, env=env)
        if proc.returncode != 0:
            print(f"Case failed with return code {proc.returncode}: {cfg_path}", file=sys.stderr)
            return proc.returncode

    print("Sweep complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
