#!/usr/bin/env python
"""Extract the canonical EXACT W_scatter linewidth gamma on the FULL BZ grid,
for one or more temperatures, from a system's production_epw outputs.

Generalization of dynamics/figures_band_path/extract_wscatter_gamma.py:
  - arbitrary --temperatures list instead of hardcoded 50K
  - production_epw/ paths instead of production/
  - cache dir naming production_epw actually uses: phono3py_cache_<mesh>_T<T>.0K
    (a per-T tagged directory, avoiding the untagged gamma_detail T-reuse bug)
  - grid metadata (num_grid, qpoints_frac, grg2bzg) is derived directly from
    the phono3py yaml (Phono3py.load + mesh_numbers + _build_regular_grid_metadata,
    identical to what build_pp_collision_cache.py itself computes) instead of
    being read from an existing pp_collision_events_*.h5 -- this is pure grid
    bookkeeping, not physics, so there is no need to wait for/depend on the
    (possibly much slower) collision-cache job just to get it.

    gamma(q0, b0) = Sum_triplets weight * Sum_{b1,b2} gamma_detail[0, t, b0, b1, b2]

Output (per system, per temperature): <root>/production_epw/figures/data/
  gamma_W_full_T<T>.npz with
    gamma_W_full   (num_grid, nband)  THz, full regular grid
    qpoints_frac   (num_grid, 3)
    nband, mesh_numbers, temperature_K
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import h5py

# Local layout: w_scatter_lib lives in ../w_scatter, build_pp_collision_cache
# (used only for _build_regular_grid_metadata grid bookkeeping) sits alongside
# this file in figures_band_path/.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "w_scatter"))
from . import w_scatter_lib as wl  # noqa: E402
sys.path.insert(0, str(_HERE))
from . import build_pp_collision_cache as B  # noqa: E402


def _grid_metadata_from_yaml(phono3py_yaml: str, mesh):
    import phono3py as Phono3py

    ph3 = Phono3py.load(str(phono3py_yaml), is_mesh_symmetry=False, log_level=0)
    ph3.mesh_numbers = list(mesh)
    meta = B._build_regular_grid_metadata(ph3.grid)
    nband = 3 * len(ph3.primitive)
    return meta["num_grg"], nband, meta["qpoints_frac"], meta["grg2bzg"]


def extract_one(root: Path, mesh, mtag: str, temperature: int, out_dir: Path,
                 phono3py_yaml: str) -> Path:
    prod = root / "production_epw"
    w_h5 = prod / "W" / f"{temperature}K" / f"W_scattering_mesh_{mtag}_T{temperature}_GPU.h5"
    cache_dir = (prod / "W" / f"{temperature}K"
                 / f"phono3py_cache_{mesh[0]}x{mesh[1]}x{mesh[2]}_T{temperature}.0K")

    for p in (w_h5, cache_dir):
        if not Path(p).exists():
            raise FileNotFoundError(f"[T={temperature}K] missing input: {p}")

    num_grid, nband, qpts_grg, grg2bzg = _grid_metadata_from_yaml(phono3py_yaml, mesh)

    with h5py.File(w_h5, "r") as f:
        ir_grid_map = np.array(f["ir_grid_map"][:], dtype=np.int64)
        ir_grid_points = np.array(f["ir_grid_points"][:], dtype=np.int64)

    gamma_ir = wl.extract_gamma_exact(str(cache_dir), mesh, ir_grid_points, ir_grid_map, nband)

    grg_to_ir = ir_grid_map[grg2bzg]
    gamma_full = gamma_ir[grg_to_ir]

    if not np.all(np.isfinite(gamma_full)):
        raise ValueError(f"[T={temperature}K] non-finite gamma after expansion")
    if np.any(gamma_full < -1e-9):
        raise ValueError(f"[T={temperature}K] negative gamma encountered "
                          f"(min={gamma_full.min():.3e})")

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"gamma_W_full_T{temperature}.npz"
    np.savez(
        out,
        gamma_W_full=gamma_full,
        qpoints_frac=qpts_grg,
        nband=nband,
        mesh_numbers=np.array(mesh, dtype=np.int64),
        temperature_K=float(temperature),
    )
    print(f"[T={temperature}K] gamma_W_full {gamma_full.shape}  "
          f"min={gamma_full.min():.4e} max={gamma_full.max():.4e} THz  -> {out}",
          flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="system root dir, e.g. .../1-MoSe2")
    ap.add_argument("phono3py_yaml", help="phono3py displacement yaml for grid metadata")
    ap.add_argument("--mesh", nargs=3, type=int, required=True)
    ap.add_argument("--mtag", required=True, help="e.g. m36x36x1")
    ap.add_argument("--temperatures", nargs="+", type=int, default=[50, 100, 300])
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or (args.root / "production_epw" / "figures" / "data")
    for T in args.temperatures:
        extract_one(args.root, args.mesh, args.mtag, T, out_dir, args.phono3py_yaml)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
