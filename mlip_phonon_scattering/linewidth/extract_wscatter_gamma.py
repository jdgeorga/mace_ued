#!/usr/bin/env python
"""Extract the canonical EXACT W_scatter linewidth gamma on the FULL BZ grid.

This is a pure CPU reduction of the already-computed per-grid-point
`gamma_detail-*.hdf5` caches (the GPU step was done when the collision cache was
built), so no GPU recompute is needed:

    gamma(q0, b0) = Sum_triplets weight * Sum_{b1,b2} gamma_detail[0, t, b0, b1, b2]

We read grid metadata from the existing collision cache + assembled-W h5, reduce
gamma_detail on the irreducible grid via w_scatter_lib.extract_gamma_exact, then
expand to the full regular grid so it is directly comparable to the collision
gamma0 (same num_grid x nband layout, same qpoints_frac order).

Output (per system): figures_band_path/data/<sys>/gamma_W_full_T50.npz with
  gamma_W_full   (num_grid, nband)  THz, full regular grid
  qpoints_frac   (num_grid, 3)
  nband, mesh_numbers, temperature_K
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import h5py

PV4 = Path(os.environ.get(
    "PV4",
    "/pscratch/sd/j/jdgeorga/twist-anything/phonon_unfolding/scratch/phonon_diff/ued_paper/paper_v4",
))
WSCATTER = PV4 / "dynamics" / "w_scatter"
sys.path.insert(0, str(WSCATTER))
from . import w_scatter_lib as wl  # noqa: E402

OUT_ROOT = PV4 / "dynamics" / "figures_band_path" / "data"

SYSTEMS = {
    "mose2_monolayer": {
        "root": PV4 / "1-MoSe2",
        "mesh": [36, 36, 1],
        "mtag": "m36x36x1",
    },
    "mose2_wse2_aligned": {
        "root": PV4 / "2-MoSe2-WSe2",
        "mesh": [36, 36, 1],
        "mtag": "m36x36x1",
    },
    "mose2_wse2_22deg": {
        "root": PV4 / "3-MoSe2-WSe2_42_22deg",
        "mesh": [12, 12, 1],
        "mtag": "m12x12x1",
    },
}


def extract_one(name: str, cfg: dict) -> Path:
    root = cfg["root"]
    mesh = cfg["mesh"]
    mtag = cfg["mtag"]
    prod = root / "production"
    collision = prod / f"pp_collision_events_{mtag}.h5"
    w_h5 = prod / "W" / "50K" / f"W_scattering_mesh_{mtag}_T50_GPU.h5"
    cache_dir = prod / "W" / "50K" / f"phono3py_cache_{mesh[0]}x{mesh[1]}x{mesh[2]}"

    for p in (collision, w_h5, cache_dir):
        if not Path(p).exists():
            raise FileNotFoundError(f"[{name}] missing input: {p}")

    with h5py.File(collision, "r") as f:
        num_grid = int(f.attrs["num_grid"])
        nband = int(f.attrs["nband"])
        qpts_grg = np.array(f["grid/qpoints_frac"][:], dtype=np.float64)  # (num_grid, 3)
        grg2bzg = np.array(f["grid/grg2bzg"][:], dtype=np.int64)          # (num_grid,)

    with h5py.File(w_h5, "r") as f:
        ir_grid_map = np.array(f["ir_grid_map"][:], dtype=np.int64)        # bzg -> ir-pos
        ir_grid_points = np.array(f["ir_grid_points"][:], dtype=np.int64)  # ir-pos -> bzg

    # Canonical exact gamma on the irreducible grid (n_ir, nband).
    gamma_ir = wl.extract_gamma_exact(
        str(cache_dir), mesh, ir_grid_points, ir_grid_map, nband
    )

    # Expand to the full regular grid: row g <- its irreducible representative.
    grg_to_ir = ir_grid_map[grg2bzg]                      # (num_grid,) ir-positions
    gamma_full = gamma_ir[grg_to_ir]                       # (num_grid, nband)

    if not np.all(np.isfinite(gamma_full)):
        raise ValueError(f"[{name}] non-finite gamma after expansion")
    if np.any(gamma_full < -1e-9):
        raise ValueError(f"[{name}] negative gamma encountered (min={gamma_full.min():.3e})")

    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "gamma_W_full_T50.npz"
    np.savez(
        out,
        gamma_W_full=gamma_full,
        qpoints_frac=qpts_grg,
        nband=nband,
        mesh_numbers=np.array(mesh, dtype=np.int64),
        temperature_K=50.0,
    )
    print(f"[{name}] gamma_W_full {gamma_full.shape}  "
          f"min={gamma_full.min():.4e} max={gamma_full.max():.4e} THz  -> {out}",
          flush=True)
    return out


def main() -> int:
    names = sys.argv[1:] or list(SYSTEMS)
    for name in names:
        if name not in SYSTEMS:
            print(f"unknown system '{name}'; choices: {list(SYSTEMS)}")
            return 2
        extract_one(name, SYSTEMS[name])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
