#!/usr/bin/env python
"""Plot the equilibrium ph-ph linewidth & lifetime band-path figures for a
single system, pointing all inputs at an example/run folder.

Thin driver over linewidth_path.py: it overrides the SYSTEMS[name] entry with
the example-folder yaml / fc2 / gamma_W npz and renders into the example figures
dir, reproducing the paper_v4 figures:
    <name>_linewidth_lifetime_band_path.{png,pdf}            (q,omega-interpolated)
    <name>_linewidth_lifetime_band_path_meshpoints.{png,pdf} (exact on-path mesh)

Example:
  python plot_linewidth_example.py \
      --name mose2_wse2_aligned \
      --yaml     RUN/phono3py_disp.yaml \
      --fc2      RUN/phonon_cache_m36x36x1/fc2.npy \
      --gamma-w  RUN/gamma_W_full_T50.npz \
      --mesh 36 36 1 \
      --out-dir  RUN/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import linewidth_path as lp


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--name", default="mose2_wse2_aligned",
                    help="SYSTEMS key to override/render (label only; paths come from flags).")
    ap.add_argument("--label", default=None, help="Plot label override.")
    ap.add_argument("--yaml", required=True, help="phono3py displacement yaml (grid + fc2 dynamical matrix).")
    ap.add_argument("--fc2", required=True, help="phonon_cache_*/fc2.npy (symmetrized 2nd-order FCs).")
    ap.add_argument("--gamma-w", required=True, help="gamma_W_full_T<T>.npz from extract_wscatter_gamma_epw.")
    ap.add_argument("--mesh", nargs=3, type=int, default=[36, 36, 1])
    ap.add_argument("--out-dir", required=True, help="Directory to write the figures into.")
    ap.add_argument("--npoints", type=int, default=101)
    ap.add_argument("--min-freq-thz", type=float, default=0.1)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--mode", choices=["interp", "exact", "both"], default="both")
    args = ap.parse_args()

    name = args.name
    cfg = dict(lp.SYSTEMS.get(name, {}))
    cfg.update(
        label=args.label or cfg.get("label", name),
        mesh=tuple(args.mesh),
        mtag="m{}x{}x{}".format(*args.mesh),
        yaml=Path(args.yaml),
        fc2=Path(args.fc2),
        gamma_w=Path(args.gamma_w),
    )
    lp.SYSTEMS[name] = cfg
    out_dirs = [Path(args.out_dir)]

    if args.mode in ("interp", "both"):
        lp.plot_system(name, npoints=args.npoints, min_freq_thz=args.min_freq_thz,
                       dpi=args.dpi, out_dirs=out_dirs)
    if args.mode in ("exact", "both"):
        lp.plot_system_exact(name, npoints=args.npoints, min_freq_thz=args.min_freq_thz,
                             dpi=args.dpi, out_dirs=out_dirs)
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
