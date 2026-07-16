#!/usr/bin/env python
"""Extract the exact W_scatter linewidth gamma on the full BZ grid for a single
run, using explicit file paths (no production_epw/ directory convention).

Same physics/reduction as extract_wscatter_gamma_epw.extract_one, but takes the
W h5, the gamma_detail cache dir, and the phono3py yaml directly, so it works
from a flat example/run folder:

    gamma(q0,b0) = sum_triplets weight * sum_{b1,b2} gamma_detail[0,t,b0,b1,b2]

Output: gamma_W_full_T<T>.npz with gamma_W_full (num_grid, nband) THz,
qpoints_frac (num_grid,3), nband, mesh_numbers, temperature_K.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import h5py

# _grid_metadata_from_yaml (yaml -> num_grid, nband, qpoints_frac, grg2bzg) and
# w_scatter_lib live alongside / next to this file on PYTHONPATH via the env.
from .extract_wscatter_gamma_epw import _grid_metadata_from_yaml
from . import w_scatter_lib as wl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--w-h5", required=True, help="W_scattering_mesh_*_GPU.h5 (ir_grid_map/ir_grid_points).")
    ap.add_argument("--cache-dir", required=True, help="phono3py_cache_*_T<T>.0K dir of gamma_detail-*.hdf5.")
    ap.add_argument("--yaml", required=True, help="phono3py displacement yaml (grid metadata).")
    ap.add_argument("--mesh", nargs=3, type=int, required=True)
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--out", required=True, help="Output gamma_W_full_T<T>.npz path.")
    args = ap.parse_args()

    mesh = list(args.mesh)
    num_grid, nband, qpts_grg, grg2bzg = _grid_metadata_from_yaml(args.yaml, mesh)

    with h5py.File(args.w_h5, "r") as f:
        ir_grid_map = np.array(f["ir_grid_map"][:], dtype=np.int64)
        ir_grid_points = np.array(f["ir_grid_points"][:], dtype=np.int64)

    gamma_ir = wl.extract_gamma_exact(str(args.cache_dir), mesh, ir_grid_points, ir_grid_map, nband)
    grg_to_ir = ir_grid_map[grg2bzg]
    gamma_full = gamma_ir[grg_to_ir]

    if not np.all(np.isfinite(gamma_full)):
        raise ValueError("non-finite gamma after expansion")
    if np.any(gamma_full < -1e-9):
        raise ValueError(f"negative gamma encountered (min={gamma_full.min():.3e})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        gamma_W_full=gamma_full,
        qpoints_frac=qpts_grg,
        nband=nband,
        mesh_numbers=np.array(mesh, dtype=np.int64),
        temperature_K=float(args.temperature),
    )
    print(f"[T={args.temperature}K] gamma_W_full {gamma_full.shape}  "
          f"min={gamma_full.min():.4e} max={gamma_full.max():.4e} THz -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
