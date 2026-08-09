#!/usr/bin/env python3
# provenance: URSA-generated, then hand/Claude-refactored to route through autotokamak.core
"""
Run a config-driven dataset sweep of fixed-boundary Grad-Shafranov equilibria
using autotokamak.core. Writes outputs/dataset.h5 and diagnostic plots.

Usage:
  python run_dataset_sweep.py dataset_config.yaml

This script follows the project's required schema and records per-sample
get_last_solve_info()["isoflux_used"]. It counts samples as successful
if solve_equilibrium returns a solved object and psi(R,Z) interpolation
is finite-valued.

This is a simple PoC runner, not production-grade. See README.md for details.
"""
import sys
import os
import time
import json
import signal
import math
from pathlib import Path

import yaml
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy import stats
from scipy.interpolate import griddata

import autotokamak.core.geometry as geometry
import autotokamak.core.solver as solver

# Globals set by signal handler
STOP_AFTER_CURRENT = False

def sig_handler(signum, frame):
    global STOP_AFTER_CURRENT
    print(f"Received signal {signum}; will stop after current sample...")
    STOP_AFTER_CURRENT = True

signal.signal(signal.SIGINT, sig_handler)
signal.signal(signal.SIGTERM, sig_handler)


def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def sample_parameters(cfg):
    samp = cfg['sampling']
    n = int(samp['n_samples'])
    method = samp.get('method', 'lhs')
    seed = int(samp.get('seed', 0))
    rng = np.random.default_rng(seed)

    params = cfg['parameters']
    names = list(params.keys())
    lows = np.array([params[n]['low'] for n in names], dtype=float)
    highs = np.array([params[n]['high'] for n in names], dtype=float)

    if method == 'lhs':
        sampler = stats.qmc.LatinHypercube(d=len(names), seed=seed)
        unit = sampler.random(n=n)
    elif method == 'uniform':
        unit = rng.random((n, len(names)))
    else:
        raise RuntimeError(f"Unknown sampling.method: {method}")

    samples = lows + unit * (highs - lows)
    return names, samples


def make_output_dirs(plot_cfg, output_path):
    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if plot_cfg and plot_cfg.get('enabled'):
        Path(plot_cfg.get('output_dir', 'outputs/plots')).mkdir(parents=True, exist_ok=True)


def build_grid(grid_cfg):
    Rcfg = grid_cfg['R']
    Zcfg = grid_cfg['Z']
    R = np.linspace(Rcfg['min'], Rcfg['max'], int(Rcfg['n']), dtype=np.float64)
    Z = np.linspace(Zcfg['min'], Zcfg['max'], int(Zcfg['n']), dtype=np.float64)
    return R, Z


def interp_psi_to_grid(mesh_pts, mesh_lc, psi_nodes, R, Z):
    # Return psi_grid (nz, nr)
    try:
        tri = mtri.Triangulation(mesh_pts[:,0], mesh_pts[:,1], triangles=mesh_lc)
        interp = mtri.LinearTriInterpolator(tri, psi_nodes)
        RR, ZZ = np.meshgrid(R, Z, indexing='xy')
        psi_grid = np.asarray(interp(RR, ZZ).filled(np.nan), dtype=np.float64)
        if np.all(np.isnan(psi_grid)):
            # fallback to griddata
            xi = np.column_stack([mesh_pts[:,0], mesh_pts[:,1]])
            pts = np.column_stack([RR.ravel(), ZZ.ravel()])
            vals = griddata(xi, psi_nodes, pts, method='linear')
            psi_grid = vals.reshape(RR.shape).astype(np.float64)
        return psi_grid
    except Exception as e:
        # final fallback: scattered griddata
        try:
            xi = np.column_stack([mesh_pts[:,0], mesh_pts[:,1]])
            RR, ZZ = np.meshgrid(R, Z, indexing='xy')
            pts = np.column_stack([RR.ravel(), ZZ.ravel()])
            vals = griddata(xi, psi_nodes, pts, method='linear')
            psi_grid = vals.reshape(RR.shape).astype(np.float64)
            return psi_grid
        except Exception as e2:
            raise


def ensure_datasets(h5f, cfg, n_samples, R, Z):
    # Create datasets if missing. If present, validate shapes
    def mk(name, shape, dtype, fillvalue=None):
        if name in h3:
            return h3[name]
        else:
            return h3.create_dataset(name, shape=shape, dtype=dtype)

    h3 = h5f
    if '/grid/R' in h3:
        R0 = h3['/grid/R'][:]
        Z0 = h3['/grid/Z'][:]
        if not (np.allclose(R0, R) and np.allclose(Z0, Z)):
            raise RuntimeError('Existing HDF5 grid differs from config output_grid; aborting to avoid corruption')
    else:
        h3.create_dataset('/grid/R', data=R)
        h3.create_dataset('/grid/Z', data=Z)

    if '/inputs/r0' not in h3:
        n = int(n_samples)
        h3.create_dataset('/inputs/r0', shape=(n,), dtype=np.float64)
        h3.create_dataset('/inputs/a', shape=(n,), dtype=np.float64)
        h3.create_dataset('/inputs/kappa', shape=(n,), dtype=np.float64)
        h3.create_dataset('/inputs/delta', shape=(n,), dtype=np.float64)
        h3.create_dataset('/inputs/Ip', shape=(n,), dtype=np.float64)

    nz = len(Z)
    nr = len(R)
    if '/outputs/psi' not in h3:
        h3.create_dataset('/outputs/psi', shape=(int(n_samples), nz, nr), dtype=np.float64)
    if '/outputs/success' not in h3:
        h3.create_dataset('/outputs/success', shape=(int(n_samples),), dtype='?')
    if '/outputs/isoflux_used' not in h3:
        h3.create_dataset('/outputs/isoflux_used', shape=(int(n_samples),), dtype='?')
    if '/outputs/error_msgs' not in h3:
        h3.create_dataset('/outputs/error_msgs', shape=(int(n_samples),), dtype=h5py.string_dtype(encoding='utf-8'))
    if '/outputs/solve_info' not in h3:
        h3.create_dataset('/outputs/solve_info', shape=(int(n_samples),), dtype=h5py.string_dtype(encoding='utf-8'))

    # root attrs
    h3.attrs['n_samples'] = int(n_samples)
    if 'created' not in h3.attrs:
        h3.attrs['created'] = time.asctime()


def plot_running(plots_dir, successes, total_done):
    plt.figure()
    x = np.arange(1, total_done+1)
    y = np.cumsum(successes[:total_done]) / x
    plt.plot(x, y, '-o', markersize=3)
    plt.xlabel('samples')
    plt.ylabel('running success fraction')
    plt.grid(True)
    p = Path(plots_dir) / 'running_success.png'
    plt.savefig(p)
    plt.close()


def plot_params_scatter(plots_dir, samples, names, successes):
    # scatter of r0 vs a as a simple diagnostic, color by success
    r0_idx = names.index('r0')
    a_idx = names.index('a')
    plt.figure(figsize=(4,4))
    s = successes
    plt.scatter(samples[:,r0_idx], samples[:,a_idx], c=s, cmap='bwr', vmin=0, vmax=1, s=8)
    plt.xlabel('r0'); plt.ylabel('a')
    plt.title('r0 vs a (blue=success)')
    p = Path(plots_dir) / 'r0_vs_a.png'
    plt.savefig(p)
    plt.close()


def plot_latest_psi(plots_dir, psi_grid, R, Z):
    if psi_grid is None:
        return
    plt.figure(figsize=(4,6))
    RR, ZZ = np.meshgrid(R, Z, indexing='xy')
    plt.pcolormesh(RR, ZZ, psi_grid, shading='auto')
    plt.colorbar(label='psi')
    plt.xlabel('R'); plt.ylabel('Z')
    p = Path(plots_dir) / 'latest_psi.png'
    plt.savefig(p)
    plt.close()


def plot_histograms(plots_dir, samples, names):
    plt.figure(figsize=(8,6))
    for i, nm in enumerate(names):
        plt.subplot(2,3,i+1)
        plt.hist(samples[:,i], bins=30)
        plt.title(nm)
    plt.tight_layout()
    p = Path(plots_dir) / 'input_histograms.png'
    plt.savefig(p)
    plt.close()


def main(cfg_path):
    cfg = load_config(cfg_path)
    names, samples = sample_parameters(cfg)
    n_samples = int(cfg['sampling']['n_samples'])
    fixed = cfg.get('fixed', {})
    plotting = cfg.get('plotting', {})

    R, Z = build_grid(cfg['output_grid'])
    make_output_dirs(plotting, cfg['output_path'])

    out_path = cfg['output_path']
    plots_dir = plotting.get('output_dir', 'outputs/plots')
    every_n = int(plotting.get('every_n_samples', 1))

    # open/create hdf5
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    h5f = h5py.File(out_path, 'a')
    ensure_datasets(h5f, cfg, n_samples, R, Z)

    # determine resume index: first index where success is False/empty
    succ_ds = h5f['/outputs/success']
    start_idx = 0
    for i in range(n_samples):
        v = succ_ds[i]
        if not bool(v):
            start_idx = i
            break
        start_idx = i+1
    print(f"Starting at index {start_idx} / {n_samples}")

    latest_successful_psi = None
    successes = np.array(h5f['/outputs/success'][:], dtype=int)

    for idx in range(start_idx, n_samples):
        global STOP_AFTER_CURRENT
        if STOP_AFTER_CURRENT:
            print('Stop requested; breaking before starting sample', idx)
            break

        pvals = samples[idx]
        # map names to values
        pv = dict(zip(names, pvals))
        r0 = float(pv['r0']); z0 = float(fixed.get('z0', 0.0));
        a = float(pv['a']); kappa = float(pv['kappa']); delta = float(pv['delta'])
        Ip = float(pv['Ip'])
        cfg_solver = {
            'equation': {'name': 'gs'},
            'boundary': {'type': 'isoflux', 'r0': r0, 'z0': z0, 'a': a, 'kappa': kappa, 'delta': delta, 'npts': int(fixed.get('npts',80))},
            'mesh': {'method': 'gs_domain', 'regions': [{'name': 'plasma', 'type': 'plasma', 'dx': float(fixed.get('mesh_dx', 0.015))}]},
            'solver': {'order': int(fixed.get('solver_order', 1)), 'F0': float(fixed.get('F0', 0.10752)), 'free_boundary': False},
            'targets': {'Ip': float(Ip), 'Ip_ratio': float(fixed.get('Ip_ratio', 1.0))},
            'init_psi': {'method': 'isoflux'}
        }

        # write inputs into hdf5 inputs arrays
        h5f['/inputs/r0'][idx] = r0
        h5f['/inputs/a'][idx] = a
        h5f['/inputs/kappa'][idx] = kappa
        h5f['/inputs/delta'][idx] = delta
        h5f['/inputs/Ip'][idx] = Ip

        success = False
        isoflux_used = False
        err_msg = ''
        solve_info = {}
        psi_grid = np.full((len(Z), len(R)), np.nan, dtype=np.float64)

        try:
            lcfs = geometry.build_lcfs(r0=r0, z0=z0, a=a, kappa=kappa, delta=delta, npts=int(fixed.get('npts',80)))
            mesh_ret = geometry.build_mesh(lcfs, mesh_dx=float(fixed.get('mesh_dx',0.015)), region_name='plasma', region_tag='plasma')
            # geometry.build_mesh may return (gs_mesh, mesh_pts, mesh_lc, mesh_reg) or (mesh_pts, mesh_lc, mesh_reg)
            if isinstance(mesh_ret, tuple) and len(mesh_ret) == 4:
                _, mesh_pts, mesh_lc, mesh_reg = mesh_ret
            elif isinstance(mesh_ret, tuple) and len(mesh_ret) == 3:
                mesh_pts, mesh_lc, mesh_reg = mesh_ret
            else:
                raise RuntimeError(f'Unexpected return from build_mesh: {type(mesh_ret)}')
            # Call solver
            gs = solver.solve_equilibrium(mesh_pts=mesh_pts, mesh_lc=mesh_lc, mesh_reg=mesh_reg, lcfs=lcfs, cfg=cfg_solver)
            info = solver.get_last_solve_info()
            isoflux_used = bool(info.get('isoflux_used', False))
            solve_info = info

            psi_nodes = np.asarray(gs.get_psi(), dtype=float).ravel()
            if len(psi_nodes) != len(mesh_pts):
                # This shouldn't happen for order==1, but guard
                raise RuntimeError(f'psi node length {len(psi_nodes)} != mesh_pts {len(mesh_pts)}')

            psi_grid = interp_psi_to_grid(mesh_pts, mesh_lc, psi_nodes, R, Z)
            # success criteria: at least some finite values
            if np.isfinite(psi_grid).sum() > 0:
                success = True
                latest_successful_psi = psi_grid
            else:
                success = False
                err_msg = 'psi interpolation produced no finite values'

        except Exception as e:
            err_msg = f'{type(e).__name__}: {str(e)}'

        # record results
        h5f['/outputs/psi'][idx,:,:] = psi_grid
        h5f['/outputs/success'][idx] = bool(success)
        h5f['/outputs/isoflux_used'][idx] = bool(isoflux_used)
        h5f['/outputs/error_msgs'][idx] = err_msg
        h5f['/outputs/solve_info'][idx] = json.dumps(solve_info)
        h5f.flush()

        successes[idx] = 1 if success else 0

        # plotting
        if plotting.get('enabled', False) and ((idx+1) % every_n == 0 or success):
            try:
                plot_running(plots_dir, successes, idx+1)
                plot_params_scatter(plots_dir, samples, names, successes)
                plot_latest_psi(plots_dir, latest_successful_psi, R, Z)
            except Exception as e:
                print('Plotting error:', e)

        print(f"Sample {idx+1}/{n_samples}: success={success} isoflux_used={isoflux_used} err={err_msg}")

        if STOP_AFTER_CURRENT:
            print('Stop requested; breaking after finishing sample', idx)
            break

    # final plots
    try:
        if plotting.get('enabled', False):
            plot_histograms(plots_dir, samples, names)
    except Exception as e:
        print('Final plotting error:', e)

    total_succeeded = int(np.sum(successes))
    total_isoflux = int(np.sum(np.array(h5f['/outputs/isoflux_used'][:], dtype=int)))
    print(f"Wrote {out_path}: {total_succeeded}/{n_samples} succeeded (isoflux_used: {total_isoflux}/{n_samples})")
    h5f.close()


def apply_override(cfg, key, val):
    # key is dot-delimited path into cfg dict
    parts = key.split('.')
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    # parse val
    try:
        v = yaml.safe_load(val)
    except Exception:
        v = val
    cur[parts[-1]] = v


def quick_sweep(cfg_path, overrides, force=False, env_overrides=None):
    # Load base config
    with open(cfg_path, 'r') as f:
        base = yaml.safe_load(f)
    cfg = base.copy()
    # apply CLI overrides
    for ov in overrides:
        if '=' not in ov:
            raise RuntimeError(f'Bad override spec: {ov} (expected key=val)')
        k, v = ov.split('=', 1)
        apply_override(cfg, k, v)
    # apply env overrides (semicolon or comma separated key=val)
    if env_overrides:
        for part in env_overrides.replace(';', ',').split(','):
            part = part.strip()
            if not part:
                continue
            if '=' not in part:
                continue
            k, v = part.split('=', 1)
            apply_override(cfg, k, v)
    # ensure output_path is explicit and safe
    outp = cfg.get('output_path') or cfg.get('outputs', {}).get('path')
    if not outp:
        outp = 'outputs/tmp_dataset.h5'
        cfg['output_path'] = outp
    # if output_path equals original config's output_path and not force, refuse
    orig_out = base.get('output_path') or base.get('outputs', {}).get('path')
    if orig_out and os.path.abspath(outp) == os.path.abspath(orig_out) and not force:
        raise RuntimeError('Quick sweep refuses to overwrite original output_path; use --force to override')
    # write temp config to a file next to original with suffix .quick.yaml
    tmp_path = Path(cfg_path).with_suffix('.quick.yaml')
    with open(tmp_path, 'w') as f:
        yaml.safe_dump(cfg, f)
    print(f'Running quick sweep with config {tmp_path} -> output {outp}')
    try:
        main(str(tmp_path))
    finally:
        # leave tmp config for inspection
        pass
    # Validation: open output h5 and produce quick report
    try:
        import h5py
        with h5py.File(outp, 'r') as h5f:
            n = int(h5f['/grid/R'].shape[0])
            m = int(h5f['/grid/Z'].shape[0])
            n_samples = int(h5f.attrs.get('n_samples', h5f['/outputs/success'].shape[0]))
            succ = int(np.sum(h5f['/outputs/success'][:]))
            isof = int(np.sum(np.array(h5f['/outputs/isoflux_used'][:], dtype=int)))
            report = {
                'output_path': outp,
                'n_samples': int(n_samples),
                'grid_nr': int(n),
                'grid_nz': int(m),
                'succeeded': int(succ),
                'isoflux_used': int(isof)
            }
            rep_txt = json.dumps(report, indent=2)
            rptfile = Path(outp).parent / 'quick_sweep_report.json'
            with open(rptfile, 'w') as f:
                f.write(rep_txt)
            print('Quick sweep validation report:')
            print(rep_txt)
            print(f'Wrote report to {rptfile}')
    except Exception as e:
        print('Quick sweep validation failed:', e)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run dataset sweep or a quick non-destructive validation sweep')
    parser.add_argument('config', help='Path to dataset_config.yaml')
    parser.add_argument('--quick', action='store_true', help='Run a quick validation sweep (does not overwrite original output unless --force)')
    parser.add_argument('--override', action='append', default=[], help='Override config values, e.g. --override sampling.n_samples=20 (can be repeated)')
    parser.add_argument('--force', action='store_true', help='Allow quick sweep to overwrite original output_path')
    parser.add_argument('--env-override', default=None, help='Overrides from environment-style string, e.g. "sampling.n_samples=10,output_path=outputs/tmp.h5"')
    args = parser.parse_args()
    if args.quick:
        try:
            quick_sweep(args.config, args.override, force=args.force, env_overrides=args.env_override or os.environ.get('QUICK_OVERRIDE'))
        except Exception as e:
            print('Quick sweep failed:', e)
            sys.exit(1)
    else:
        # default behavior: run full sweep
        if args.override:
            print('Warning: --override ignored for full run; use quick mode for non-destructive overrides')
        main(args.config)

    
    