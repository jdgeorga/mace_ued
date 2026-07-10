"""
w_scatter_lib — self-contained W_scatter / phonon-linewidth library.

VENDORED from `workflow/gpu_scattering_W_phonons_bilayer_comm_mesh.py` (the proven
GPU gamma_detail driver) and adapted so the W_scatter calculation lives INSIDE the
dynamics repo and is self-contained.  The original workflow/ driver is left intact;
other scripts depend on it.

What this module provides
-------------------------
1. `compute_gamma_detail_on_mesh_gpu` — the proven GPU path that writes per-grid-point
   phono3py `gamma_detail` HDF5 files (unchanged behavior).
2. `assemble_w_matrix` — the proven W-assembly logic.  It can additionally return the
   *unsymmetrized* W (needed for the exact-gamma self-consistency identity).
3. `gamma_exact_from_file` / `extract_gamma_exact` — the CANONICAL EXACT per-mode
   phonon linewidth gamma(q0,b0), computed by the direct phono3py self-energy
   reduction of `gamma_detail`:

       gamma(q0, b0) = Sum_triplets weight * Sum_{b1,b2} gamma_detail[0, t, b0, b1, b2]

   It is validated directly against stock phono3py `run_imag_self_energy`
   (ratio 1.000 across acoustic+optical modes, K/M/near-Gamma; see
   `verify_extraction.py` and `dynamics/docs/specs/2026-06-30-gamma0-removal.md`).
   It is the primary linewidth output of this workflow.

   NOTE (2026-06-30): the collision-cache `gamma0` (from the now-removed
   `gamma0_extract.py`) is NOT a reliable comparison target in general -- it
   diverges for Bose-enhanced soft scattering partners and its abs_plus/abs_minus
   collision brackets can double-count a single physical process. It only
   happened to agree with the quantity above at top-optical K/M, which is why an
   earlier version of this docstring claimed "~0.0000% for every mode". Do not
   reintroduce a gamma0-based comparison as ground truth.

Convention finding (see README.md and the diagnostics probe)
------------------------------------------------------------
`row_sum(symmetrized W)` is a LEGACY / DIAGNOSTIC-only linewidth estimator.  The
assembled-W row double-counts the two scattering legs (factor 2 before
symmetrization), and `0.5*(W+W.T)` is intended to recover gamma but leaves a
+0.5%..3.5% half-column-sum residual on the irreducible grid.  The canonical gamma
therefore comes from `extract_gamma_exact` (the direct gamma_detail reduction),
NOT from `row_sum(symW)`.  Equivalent exact forms:
    extract_gamma_exact  ==  0.5 * row_sum(UNsymmetrized W)  ==  a single leg's row-sum.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Local forked phono3py (contains lang="GPU" path).  Honor $PHONO3PY_EINSUM_PATH
# (set by env/setup_env.sh) first; fall back to a couple of repo-relative guesses
# so the module also works without the env var.
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_phono3py_einsum_path() -> Optional[str]:
    env_path = os.environ.get("PHONO3PY_EINSUM_PATH")
    candidates: List[str] = []
    if env_path:
        candidates.append(env_path)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    # dynamics/w_scatter/ -> paper_v4 is 2 levels up; phonopy_codes is a sibling
    # of the ued_paper tree (../../../../phonopy_codes from here historically).
    candidates.append(
        os.path.abspath(os.path.join(this_dir, "../../../../../phonopy_codes/phono3py_einsum"))
    )
    for cand in candidates:
        if cand and os.path.isdir(cand):
            return os.path.abspath(cand)
    return None


LOCAL_PHONO3PY_EINSUM = _resolve_phono3py_einsum_path()
if LOCAL_PHONO3PY_EINSUM and LOCAL_PHONO3PY_EINSUM not in sys.path:
    sys.path.insert(0, LOCAL_PHONO3PY_EINSUM)

try:
    import h5py

    HAS_H5PY = True
except Exception:
    HAS_H5PY = False


# ─────────────────────────────────────────────────────────────────────────────
# Small utilities (vendored unchanged).
# ─────────────────────────────────────────────────────────────────────────────
def build_reciprocal_from_direct(a_direct: np.ndarray) -> np.ndarray:
    """Return reciprocal lattice matrix (columns are b1,b2,b3) with 2pi convention."""
    return 2.0 * np.pi * np.linalg.inv(a_direct)


def unit_range_fixed(x: np.ndarray, lval: float = 1.0, eps: float = 1e-9) -> np.ndarray:
    """Fold fractional coordinates into [0, L) with tolerance at boundaries."""
    y = np.array(x, dtype=float, copy=True)
    y = y % lval
    y[(np.fabs(y) < eps) | (np.fabs(lval - y) < eps)] = 0
    return y


def mesh_tag(mesh_numbers: List[int]) -> str:
    mx, my, mz = mesh_numbers
    return f"m{mx}x{my}x{mz}"


def gamma_detail_filename_candidates(
    cache_dir: str, mesh_numbers: List[int], gp: int
) -> List[str]:
    """Return accepted gamma_detail filename variants for a grid point."""
    mx, my, mz = mesh_numbers
    concat = "m" + "".join(map(str, mesh_numbers))
    explicit = f"m{mx}x{my}x{mz}"
    return [
        os.path.join(cache_dir, f"gamma_detail-{concat}-g{gp}.hdf5"),
        os.path.join(cache_dir, f"gamma_detail-{explicit}-g{gp}.hdf5"),
    ]


def choose_existing_gamma_detail_file(
    cache_dir: str, mesh_numbers: List[int], gp: int
) -> Optional[str]:
    for fn in gamma_detail_filename_candidates(cache_dir, mesh_numbers, gp):
        if os.path.exists(fn):
            return fn
    return None


def maybe_create_mesh_tag_alias(cache_dir: str, mesh_numbers: List[int], gp: int) -> None:
    """Optionally create `m12x12x1` alias next to canonical phono3py naming."""
    mx, my, mz = mesh_numbers
    concat = "m" + "".join(map(str, mesh_numbers))
    explicit = f"m{mx}x{my}x{mz}"
    src = os.path.join(cache_dir, f"gamma_detail-{concat}-g{gp}.hdf5")
    dst = os.path.join(cache_dir, f"gamma_detail-{explicit}-g{gp}.hdf5")
    if os.path.exists(src) and (not os.path.exists(dst)):
        shutil.copy2(src, dst)


def gamma_detail_has_signal(gamma_file: str) -> bool:
    """Heuristic check that gamma_detail is not trivially all-zero."""
    if not HAS_H5PY:
        return True
    try:
        with h5py.File(gamma_file, "r") as f:
            if "gamma_detail" not in f:
                return False
            ds = f["gamma_detail"]
            if ds.size == 0:
                return False
            if ds.size <= 5_000_000:
                return bool(np.any(ds[()] != 0))
            if len(ds.shape) >= 2:
                triplet_axis = 2 if len(ds.shape) == 6 else 1
                n_triplets = ds.shape[triplet_axis]
                slab = [slice(None)] * len(ds.shape)
                for start in range(0, n_triplets, 64):
                    slab[triplet_axis] = slice(start, min(start + 64, n_triplets))
                    if np.any(ds[tuple(slab)] != 0):
                        return True
            else:
                for start in range(0, ds.shape[0], 1_000_000):
                    if np.any(ds[start : start + 1_000_000] != 0):
                        return True
                return False
    except Exception:
        return False
    return False


# ─────────────────────────────────────────────────────────────────────────────
# CANONICAL EXACT linewidth  γ(q0,b0)  from gamma_detail.
#
#   γ(q0, b0) = Σ_triplets weight · Σ_{b1,b2} gamma_detail[0, t, b0, b1, b2]
#
# This is phono3py's imaginary self-energy (HWHM). This — NOT row_sum(symmetrized
# W) — is the canonical linewidth.
#
# RAW vs. degenerate-averaged (see dynamics/docs/specs/
# 2026-07-01-degenerate-band-averaging-and-low-mem-gamma.md for the full writeup):
# phono3py has a built-in, essentially-free post-processing step,
# `average_by_degeneracy()`, that replaces each band's individual gamma with the
# mean over its degenerate set at that grid point (bands with exactly matching
# frequency -- common at Gamma/K due to little-group symmetry). The functions
# below return the RAW (un-averaged) value by default, matching every existing
# published number in this project. Pass `return_degenerate_avg=True` (plus the
# grid point's band frequencies) to additionally get the averaged value -- this
# costs microseconds (a post-hoc reduction on the already-computed raw array,
# no new phono3py calculation), never the raw value's default behavior.
# ─────────────────────────────────────────────────────────────────────────────
def gamma_degenerate_average(gamma_raw: np.ndarray, freqs_at_gp: np.ndarray) -> np.ndarray:
    """Degenerate-band-averaged variant of a raw per-band gamma array.

    Wraps phono3py's own `average_by_degeneracy()`: for every set of bands with
    exactly matching frequency at this grid point, replaces each band's gamma
    with the mean gamma over that degenerate set. This is standard, physically
    meaningful post-processing (an individual eigenvector picked within a
    degenerate manifold is gauge-dependent; only the symmetrized average is
    well-defined) -- it is NOT a bug fix for `gamma_raw`, which remains equally
    valid; the two are different, both legitimate, conventions.

    Parameters
    ----------
    gamma_raw : array (nband,)
        Raw per-band gamma, e.g. from `gamma_exact_from_file`.
    freqs_at_gp : array (nband,)
        Phonon frequencies at the same grid point, same band ordering.
    """
    from phono3py.phonon3.imag_self_energy import average_by_degeneracy

    band_indices = np.arange(len(gamma_raw), dtype="int64")
    return average_by_degeneracy(np.asarray(gamma_raw, dtype=np.float64), band_indices, freqs_at_gp)


def gamma_exact_from_file(
    gamma_file: str,
    freqs_at_gp: Optional[np.ndarray] = None,
    return_degenerate_avg: bool = False,
):
    """Exact per-band imag self-energy γ(b0) read from ONE gamma_detail file.

    Parameters
    ----------
    gamma_file : str
        Path to a `gamma_detail-*.hdf5` file.
    freqs_at_gp : array (nband,), optional
        Phonon frequencies at this grid point. Required if
        `return_degenerate_avg=True`.
    return_degenerate_avg : bool, optional
        If True, additionally return the degenerate-band-averaged value (see
        `gamma_degenerate_average`). Default False (unchanged behavior).

    Returns
    -------
    np.ndarray, shape (nband,), dtype float64
        γ(q0, b0) for the grid point stored in this file, in THz (RAW value).
    Or, if `return_degenerate_avg=True`:
        (gamma_raw, gamma_degenerate_avg) : tuple of two (nband,) float64 arrays.
    """
    if not HAS_H5PY:
        raise RuntimeError("h5py is required to read gamma_detail HDF5 files.")
    with h5py.File(gamma_file, "r") as f:
        gamma_detail = np.asarray(f["gamma_detail"][0], dtype=np.float64)  # (trip,b0,b1,b2)
        weight = np.asarray(f["weight"][:], dtype=np.float64)              # (trip,)
    # reduce b1,b2 together, weight, sum over triplets -> (b0,)
    per_triplet_full = np.sum(gamma_detail, axis=(2, 3))  # (n_triplet, b0)
    gamma_b0 = np.einsum("t,tb->b", weight, per_triplet_full)
    if not return_degenerate_avg:
        return gamma_b0
    if freqs_at_gp is None:
        raise ValueError("freqs_at_gp is required when return_degenerate_avg=True.")
    return gamma_b0, gamma_degenerate_average(gamma_b0, freqs_at_gp)


def extract_gamma_exact(
    cache_dir: str,
    mesh_numbers: List[int],
    ir_grid_points_bzg: np.ndarray,
    ir_grid_map_bzg: np.ndarray,
    n_bands: int,
    frequencies_bzg: Optional[np.ndarray] = None,
    return_degenerate_avg: bool = False,
):
    """Canonical EXACT per-mode phonon linewidth on the irreducible grid.

    For each irreducible grid point, read its `gamma_detail` file and reduce it to
    the phono3py imaginary self-energy
        γ(q0, b0) = Σ_triplets weight · Σ_{b1,b2} gamma_detail[0, t, b0, b1, b2].

    Parameters
    ----------
    cache_dir : str
        Directory holding `gamma_detail-*.hdf5` files.
    mesh_numbers : list[int]
        [mx, my, mz] (selects the gamma_detail filename variant).
    ir_grid_points_bzg : array (n_ir,)
        BZ-grid indices of the irreducible grid points.
    ir_grid_map_bzg : array (n_bzg,)
        Map BZ-grid index -> irreducible-position (0..n_ir-1).  Row i of the output
        corresponds to the irreducible position `ir_grid_map_bzg[ir_grid_points_bzg[i]]`,
        i.e. the same row indexing used by `assemble_w_matrix`.
    n_bands : int
        Number of phonon bands.
    frequencies_bzg : array (n_bzg, n_bands), optional
        Phonon frequencies indexed by real BZ-grid index (e.g.
        `ph3.get_phonon_data()[0]`). Required if `return_degenerate_avg=True`.
    return_degenerate_avg : bool, optional
        If True, additionally return the degenerate-band-averaged array. Default
        False (unchanged behavior).

    Returns
    -------
    np.ndarray, shape (n_ir, n_bands), dtype float64
        The canonical exact linewidth γ in THz (RAW value).
    Or, if `return_degenerate_avg=True`:
        (gamma, gamma_degenerate_avg) : tuple of two (n_ir, n_bands) float64 arrays.
    """
    n_ir = len(ir_grid_points_bzg)
    gamma = np.zeros((n_ir, n_bands), dtype=np.float64)
    gamma_avg = np.zeros((n_ir, n_bands), dtype=np.float64) if return_degenerate_avg else None
    missing: List[int] = []
    for gp in ir_grid_points_bzg:
        gp_i = int(gp)
        gf = choose_existing_gamma_detail_file(cache_dir, mesh_numbers, gp_i)
        if gf is None:
            missing.append(gp_i)
            continue
        ir0 = int(ir_grid_map_bzg[gp_i])
        if return_degenerate_avg:
            if frequencies_bzg is None:
                raise ValueError("frequencies_bzg is required when return_degenerate_avg=True.")
            g_raw, g_avg = gamma_exact_from_file(
                gf, freqs_at_gp=frequencies_bzg[gp_i], return_degenerate_avg=True
            )
            gamma[ir0, :] = g_raw
            gamma_avg[ir0, :] = g_avg
        else:
            gamma[ir0, :] = gamma_exact_from_file(gf)
    if missing:
        print(
            f"WARNING [extract_gamma_exact]: missing gamma_detail for "
            f"{len(missing)} grid points: {missing[:10]}...",
            flush=True,
        )
    if return_degenerate_avg:
        return gamma, gamma_avg
    return gamma


# ─────────────────────────────────────────────────────────────────────────────
# Forces + cached force constants (vendored unchanged).
# ─────────────────────────────────────────────────────────────────────────────
def set_phono3py_forces_with_mesh_fc_cache(
    ph3,
    fc2_forces: str,
    fc3_forces: str,
    cache_dir_mesh: str,
    legacy_cache_dir: str = "phonon_cache",
    populate_mesh_cache: bool = True,
    comm=None,
) -> None:
    """Attach forces and build/load cached fc2/fc3, with legacy fallback."""
    from mpi4py import MPI

    if comm is None:
        comm = MPI.COMM_SELF
    rank = comm.Get_rank()

    forces2 = np.load(fc2_forces)
    forces3 = np.load(fc3_forces)
    ph3.phonon_forces = forces2
    ph3.forces = forces3

    fc3_mesh = os.path.join(cache_dir_mesh, "fc3.npy")
    fc2_mesh = os.path.join(cache_dir_mesh, "fc2.npy")
    fc3_legacy = os.path.join(legacy_cache_dir, "fc3.npy")
    fc2_legacy = os.path.join(legacy_cache_dir, "fc2.npy")

    mesh_has = os.path.exists(fc3_mesh) and os.path.exists(fc2_mesh)
    legacy_has = os.path.exists(fc3_legacy) and os.path.exists(fc2_legacy)

    if not mesh_has and legacy_has:
        if rank == 0:
            print(f"Using legacy FC cache from `{legacy_cache_dir}/`.", flush=True)
            if populate_mesh_cache:
                os.makedirs(cache_dir_mesh, exist_ok=True)
                shutil.copy2(fc3_legacy, fc3_mesh)
                shutil.copy2(fc2_legacy, fc2_mesh)
                print(f"Populated mesh-tagged FC cache `{cache_dir_mesh}/`.", flush=True)
        comm.Barrier()
        ph3.fc3 = np.load(fc3_mesh if os.path.exists(fc3_mesh) else fc3_legacy, allow_pickle=True)
        ph3.fc2 = np.load(fc2_mesh if os.path.exists(fc2_mesh) else fc2_legacy, allow_pickle=True)
        return

    need_compute = not mesh_has
    if need_compute:
        if rank == 0:
            os.makedirs(cache_dir_mesh, exist_ok=True)
            print(f"Computing force constants in `{cache_dir_mesh}/` on rank 0...", flush=True)
            ph3.produce_fc3(symmetrize_fc3r=True)
            ph3.produce_fc2(symmetrize_fc2=True)
            np.save(fc3_mesh, ph3.fc3, allow_pickle=True)
            np.save(fc2_mesh, ph3.fc2, allow_pickle=True)
        comm.Barrier()

    if rank == 0:
        print(f"Loading cached force constants from `{cache_dir_mesh}/`...", flush=True)
    ph3.fc3 = np.load(fc3_mesh, allow_pickle=True)
    ph3.fc2 = np.load(fc2_mesh, allow_pickle=True)


# ─────────────────────────────────────────────────────────────────────────────
# GPU gamma_detail generation (vendored unchanged behavior).
# ─────────────────────────────────────────────────────────────────────────────
def setup_gpu_for_rank(rank: int, size: int) -> None:
    """Bind this MPI rank to a visible GPU device."""
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "CuPy is required for GPU mode. Install cupy-cudaXX on this environment."
        ) from exc

    n_devices = cp.cuda.runtime.getDeviceCount()
    if n_devices < 1:
        raise RuntimeError("No CUDA devices are visible for this rank.")
    dev_id = rank % n_devices
    cp.cuda.Device(dev_id).use()
    free_mem, total_mem = cp.cuda.Device(dev_id).mem_info
    if rank == 0:
        print(f"[GPU] MPI ranks: {size}, visible CUDA devices: {n_devices}", flush=True)
    print(
        f"[GPU] rank {rank} -> device {dev_id} "
        f"(free {free_mem / 1e9:.2f} GB / total {total_mem / 1e9:.2f} GB)",
        flush=True,
    )


def compute_gamma_detail_on_mesh_gpu(
    ph3,
    temperature: float,
    mesh_numbers: List[int],
    cache_dir: str,
    ir_grid_points_bzg: np.ndarray,
    batch_size: int = 1,
    lang: str = "GPU",
    make_mesh_tag_alias: bool = False,
    validate_gamma_detail: bool = True,
    fallback_lang: str = "C",
    force_recompute_gamma_detail: bool = False,
    comm=None,
) -> None:
    """Ensure gamma_detail files exist for irreducible grid points, using GPU backend."""
    from mpi4py import MPI

    if comm is None:
        comm = MPI.COMM_SELF
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        os.makedirs(cache_dir, exist_ok=True)
    comm.Barrier()

    remaining_gps = []
    invalid_cached_gps = []
    for gp in ir_grid_points_bzg:
        gp_i = int(gp)
        existing = choose_existing_gamma_detail_file(cache_dir, mesh_numbers, gp_i)
        if force_recompute_gamma_detail:
            remaining_gps.append(gp)
            continue
        if existing is None:
            remaining_gps.append(gp)
        elif validate_gamma_detail and (not gamma_detail_has_signal(existing)):
            invalid_cached_gps.append(gp_i)
            remaining_gps.append(gp)
            if rank == 0:
                print(
                    f"Recomputing grid point {gp_i}: cached file appears invalid/all-zero "
                    f"({os.path.basename(existing)}).",
                    flush=True,
                )
        elif rank == 0:
            print(
                f"Skipping grid point {gp_i}: already cached ({os.path.basename(existing)})",
                flush=True,
            )
    remaining_gps = np.array(remaining_gps, dtype="int64")

    my_grid_points = np.array(remaining_gps[rank::size], dtype="int64")
    if rank == 0:
        print(f"Grid points to compute: {len(remaining_gps)} (after filtering)", flush=True)
        if validate_gamma_detail and invalid_cached_gps:
            print(f"Found {len(invalid_cached_gps)} invalid cached gamma_detail files.", flush=True)
        print(f"Running with {size} MPI processes", flush=True)
        print("GPU backend requires tetrahedron integration (sigma=None).", flush=True)
    print(f"Rank {rank}: assigned {len(my_grid_points)} grid points", flush=True)

    try:
        ph3.sigmas = [None]
    except Exception:
        pass

    tval = float(temperature)
    bad_after_gpu_local: List[int] = []
    if len(my_grid_points) > 0:
        for start in range(0, len(my_grid_points), batch_size):
            gp_chunk = my_grid_points[start : start + batch_size]
            print(f"Rank {rank}: GPU computing grid points {gp_chunk.tolist()}", flush=True)
            t1 = time.time()
            original_dir = os.getcwd()
            os.chdir(cache_dir)
            try:
                ph3.run_imag_self_energy(
                    grid_points=gp_chunk.tolist(),
                    temperatures=[tval],
                    frequency_points_at_bands=True,
                    write_txt=False,
                    write_gamma_detail=True,
                    keep_gamma_detail=False,
                    output_filename=None,
                    lang=lang,
                )
            finally:
                os.chdir(original_dir)
            print(f"Rank {rank}: GPU chunk done in {time.time() - t1:.1f}s", flush=True)

            if make_mesh_tag_alias:
                for gp in gp_chunk.tolist():
                    maybe_create_mesh_tag_alias(cache_dir, mesh_numbers, int(gp))
            if validate_gamma_detail:
                for gp in gp_chunk.tolist():
                    fn = choose_existing_gamma_detail_file(cache_dir, mesh_numbers, int(gp))
                    if fn is None or (not gamma_detail_has_signal(fn)):
                        bad_after_gpu_local.append(int(gp))

    bad_after_gpu_all = comm.gather(bad_after_gpu_local, root=0)
    if rank == 0:
        flat_bad = sorted(set(int(x) for part in bad_after_gpu_all for x in part))
    else:
        flat_bad = None
    flat_bad = comm.bcast(flat_bad, root=0)

    if validate_gamma_detail and flat_bad:
        if fallback_lang.upper() == "NONE":
            if rank == 0:
                raise RuntimeError(
                    f"Detected {len(flat_bad)} all-zero/invalid gamma_detail files after GPU run. "
                    "Set --fallback-lang C to repair, or disable validation explicitly."
                )
        else:
            if rank == 0:
                print(
                    f"Repairing {len(flat_bad)} gamma_detail files using "
                    f"lang='{fallback_lang.upper()}'...",
                    flush=True,
                )
            my_repair = np.array(flat_bad[rank::size], dtype="int64")
            for start in range(0, len(my_repair), batch_size):
                gp_chunk = my_repair[start : start + batch_size]
                print(
                    f"Rank {rank}: repairing grid points {gp_chunk.tolist()} "
                    f"with lang={fallback_lang.upper()}",
                    flush=True,
                )
                t1 = time.time()
                original_dir = os.getcwd()
                os.chdir(cache_dir)
                try:
                    ph3.run_imag_self_energy(
                        grid_points=gp_chunk.tolist(),
                        temperatures=[tval],
                        frequency_points_at_bands=True,
                        write_txt=False,
                        write_gamma_detail=True,
                        keep_gamma_detail=False,
                        output_filename=None,
                        lang=fallback_lang.upper(),
                    )
                finally:
                    os.chdir(original_dir)
                print(f"Rank {rank}: repair chunk done in {time.time() - t1:.1f}s", flush=True)

    comm.Barrier()


# ─────────────────────────────────────────────────────────────────────────────
# W assembly (vendored; assembly math unchanged).  Optionally returns the
# UNsymmetrized W so the exact-gamma identity can be checked.
# ─────────────────────────────────────────────────────────────────────────────
def assemble_w_matrix(
    cache_dir: str,
    mesh_numbers: List[int],
    ir_grid_points_bzg: np.ndarray,
    ir_grid_map_bzg: np.ndarray,
    n_bands: int,
    include_q2_coupling: bool = True,
    return_unsym: bool = False,
):
    """Assemble symmetric W_rate matrix over irreducible q points.

    If `return_unsym` is True, returns `(w_sym, w_unsym)`; otherwise returns
    `w_sym` (identical to the original `assemble_symmetric_w_matrix`).
    """
    if not HAS_H5PY:
        raise RuntimeError("h5py is required to read gamma_detail HDF5 files.")

    n_ir = len(ir_grid_points_bzg)
    n_states = n_ir * n_bands
    w_rate = np.zeros((n_states, n_states), dtype=np.float32)

    t_start = time.time()
    for igp, gp in enumerate(ir_grid_points_bzg, start=1):
        gp_i = int(gp)
        gamma_file = choose_existing_gamma_detail_file(cache_dir, mesh_numbers, gp_i)
        if gamma_file is None:
            print(f"WARNING: Missing gamma_detail for grid point {gp_i}; skipping.", flush=True)
            continue

        with h5py.File(gamma_file, "r") as f:
            gamma_detail = f["gamma_detail"][0]  # (triplet, b0, b1, b2)
            triplet = f["triplet"][:]
            weight = f["weight"][:]

        ir0 = int(ir_grid_map_bzg[gp_i])
        row_slice = slice(ir0 * n_bands, (ir0 + 1) * n_bands)

        ir1_all = ir_grid_map_bzg[triplet[:, 1]].astype(np.int64, copy=False)
        contrib_q1_all = np.sum(gamma_detail, axis=3)  # (n_triplet, b0, b1)
        weighted_q1 = contrib_q1_all * weight[:, None, None]
        for ir1 in np.unique(ir1_all):
            mask = ir1_all == ir1
            col_slice1 = slice(int(ir1) * n_bands, (int(ir1) + 1) * n_bands)
            w_rate[row_slice, col_slice1] += np.sum(
                weighted_q1[mask], axis=0, dtype=np.float64
            ).astype(np.float32, copy=False)

        if include_q2_coupling:
            ir2_all = ir_grid_map_bzg[triplet[:, 2]].astype(np.int64, copy=False)
            contrib_q2_all = np.sum(gamma_detail, axis=2)  # (n_triplet, b0, b2)
            weighted_q2 = contrib_q2_all * weight[:, None, None]
            for ir2 in np.unique(ir2_all):
                mask = ir2_all == ir2
                col_slice2 = slice(int(ir2) * n_bands, (int(ir2) + 1) * n_bands)
                w_rate[row_slice, col_slice2] += np.sum(
                    weighted_q2[mask], axis=0, dtype=np.float64
                ).astype(np.float32, copy=False)

        if igp % 10 == 0 or igp == n_ir:
            elapsed = time.time() - t_start
            rate = igp / elapsed if elapsed > 0 else 0.0
            print(
                f"[W assembly] processed {igp}/{n_ir} irreducible q-points "
                f"({rate:.2f} q/s, {elapsed:.1f}s elapsed)",
                flush=True,
            )

    w_unsym = w_rate
    w_sym = 0.5 * (w_unsym + w_unsym.T)
    if return_unsym:
        return w_sym, w_unsym
    return w_sym


# Back-compat alias matching the original driver's public name.
def assemble_symmetric_w_matrix(*args, **kwargs):
    kwargs.pop("return_unsym", None)
    return assemble_w_matrix(*args, return_unsym=False, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Output writers.
# ─────────────────────────────────────────────────────────────────────────────
def save_gamma_exact_npz(
    out_npz: str,
    gamma_linewidth: np.ndarray,
    ir_grid_points: np.ndarray,
    ir_grid_map: np.ndarray,
    qpoints_frac: np.ndarray,
    frequencies_ir: np.ndarray,
    mesh_numbers: List[int],
    temperature: float,
    n_bands: int,
    gamma_linewidth_degenerate_avg: Optional[np.ndarray] = None,
) -> None:
    """Write the CANONICAL exact linewidth gamma to a standalone .npz.

    `gamma_linewidth` is the RAW (un-averaged) canonical value -- unchanged
    default behavior. If `gamma_linewidth_degenerate_avg` is provided, it is
    additionally written as a separate key (see `gamma_degenerate_average`);
    existing readers of `gamma_linewidth` are unaffected either way.
    """
    extra = {}
    if gamma_linewidth_degenerate_avg is not None:
        extra["gamma_linewidth_degenerate_avg"] = np.asarray(
            gamma_linewidth_degenerate_avg, dtype=np.float64
        )
    np.savez(
        out_npz,
        gamma_linewidth=np.asarray(gamma_linewidth, dtype=np.float64),
        frequencies=np.asarray(frequencies_ir, dtype=np.float64),
        ir_grid_points=np.asarray(ir_grid_points, dtype=np.int64),
        ir_grid_map=np.asarray(ir_grid_map, dtype=np.int64),
        qpoints=np.asarray(qpoints_frac, dtype=np.float64),
        mesh_numbers=np.asarray(mesh_numbers, dtype=np.int64),
        temperature_K=np.float64(temperature),
        nband=np.int64(n_bands),
        units="THz",
        definition=(
            "gamma(q0,b0) = sum_triplets weight * sum_{b1,b2} gamma_detail "
            "(phono3py imaginary self-energy / HWHM); CANONICAL exact linewidth "
            "(RAW, no degenerate-band averaging). Row i indexes irreducible "
            "position ir_grid_map[ir_grid_points[i]]. See "
            "gamma_linewidth_degenerate_avg (if present) for the "
            "degenerate-band-averaged variant."
        ),
        **extra,
    )


def save_w_matrix_hdf5(
    out_h5: str,
    lattice: np.ndarray,
    reciprocal_lattice: np.ndarray,
    qpoints_frac: np.ndarray,
    ir_grid_points: np.ndarray,
    ir_grid_map: np.ndarray,
    w_rate: np.ndarray,
    w_tau: np.ndarray,
    mesh_numbers: List[int],
    temperature: float,
    n_bands: int,
    gamma_linewidth: Optional[np.ndarray] = None,
    gamma_legacy_rowsum_symW: Optional[np.ndarray] = None,
    gamma_linewidth_degenerate_avg: Optional[np.ndarray] = None,
) -> None:
    """Write assembled W_rate / W_tau, the CANONICAL gamma_linewidth, and metadata.

    `gamma_linewidth` remains the RAW (un-averaged) canonical value. If
    `gamma_linewidth_degenerate_avg` is provided, it is additionally written as
    its own dataset (see `gamma_degenerate_average`) -- purely additive, does
    not change `gamma_linewidth`'s meaning or any existing consumer's behavior.
    """
    if not HAS_H5PY:
        raise RuntimeError("h5py is required to write W-matrix output.")

    with h5py.File(out_h5, "w") as h5:
        h5.create_dataset("lattice", data=np.array(lattice, dtype=float))
        h5.create_dataset("reciprocal_lattice", data=np.array(reciprocal_lattice, dtype=float))
        h5.create_dataset("qpoints", data=np.array(qpoints_frac, dtype=float))
        h5.create_dataset("ir_grid_points", data=np.array(ir_grid_points, dtype="int64"))
        h5.create_dataset("ir_grid_map", data=np.array(ir_grid_map, dtype="int64"))
        h5.create_dataset("W_rate", data=np.array(w_rate, dtype=np.float32))
        h5.create_dataset("W_tau", data=np.array(w_tau, dtype=np.float32))

        # PRIMARY canonical output: exact per-mode linewidth (RAW, no
        # degenerate-band averaging).
        if gamma_linewidth is not None:
            dl = h5.create_dataset(
                "gamma_linewidth", data=np.asarray(gamma_linewidth, dtype=np.float64)
            )
            dl.attrs["units"] = "THz"
            dl.attrs["definition"] = (
                "CANONICAL exact linewidth gamma(q0,b0) = sum_triplets weight * "
                "sum_{b1,b2} gamma_detail (phono3py imag self-energy / HWHM), RAW "
                "value (no degenerate-band averaging -- see "
                "gamma_linewidth_degenerate_avg if present). Validated bit-for-bit "
                "against phono3py's low-level ImagSelfEnergy raw attribute "
                "(dynamics/w_scatter/verify_extraction.py). Do NOT compare against "
                "collision-cache gamma0 (gamma0_extract.py, removed 2026-06-30) -- "
                "it is unreliable outside top-optical K/M."
            )
            dl.attrs["indexing"] = "(ir_position, band)"

        # Degenerate-band-averaged variant of gamma_linewidth (see
        # gamma_degenerate_average). Purely additive.
        if gamma_linewidth_degenerate_avg is not None:
            dla = h5.create_dataset(
                "gamma_linewidth_degenerate_avg",
                data=np.asarray(gamma_linewidth_degenerate_avg, dtype=np.float64),
            )
            dla.attrs["units"] = "THz"
            dla.attrs["definition"] = (
                "gamma_linewidth with phono3py's built-in average_by_degeneracy() "
                "applied: bands with exactly matching frequency at a grid point "
                "have their gamma replaced by the mean over that degenerate set. "
                "Physically meaningful (removes gauge-dependent individual-band "
                "noise within a degenerate eigenspace), NOT a correction to "
                "gamma_linewidth -- both are valid conventions; this project's "
                "prior published numbers use the RAW gamma_linewidth."
            )
            dla.attrs["indexing"] = "(ir_position, band)"

        # LEGACY diagnostic only: row_sum(symmetrized W).
        if gamma_legacy_rowsum_symW is not None:
            dlg = h5.create_dataset(
                "gamma_legacy_rowsum_symW",
                data=np.asarray(gamma_legacy_rowsum_symW, dtype=np.float64),
            )
            dlg.attrs["units"] = "THz"
            dlg.attrs["status"] = "LEGACY / DIAGNOSTIC ONLY — do not use as the linewidth"
            dlg.attrs["definition"] = (
                "row_sum(0.5*(W+W.T)) reshaped to (ir_position, band). Carries a "
                "+0.5%..3.5% symmetrization residual vs the exact gamma_linewidth."
            )

        h5.attrs["mesh_numbers"] = np.array(mesh_numbers, dtype=int)
        h5.attrs["temperature_K"] = float(temperature)
        h5.attrs["qpoints_convention"] = "fractional in primitive reciprocal basis"
        h5.attrs["state_indexing"] = "s = ir_q_index * nband + band_index"
        h5.attrs["nband"] = int(n_bands)
        h5.attrs["Nq_ir"] = int(len(ir_grid_points))
        h5.attrs["Nstates"] = int(w_rate.shape[0])
        h5.attrs["grid_index_convention"] = (
            "phono3py BZ-grid indexing (matches gamma_detail triplet indices)"
        )
        h5.attrs["W_q_indexing_scope"] = "ir"
        h5.attrs["canonical_linewidth_dataset"] = "gamma_linewidth"
        h5.attrs["legacy_linewidth_dataset"] = "gamma_legacy_rowsum_symW"
        if gamma_linewidth_degenerate_avg is not None:
            h5.attrs["degenerate_avg_linewidth_dataset"] = "gamma_linewidth_degenerate_avg"


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end driver.
# ─────────────────────────────────────────────────────────────────────────────
def main(
    phono3py_yaml: str,
    fc2_forces_npy: str,
    fc3_forces_npy: str,
    mesh_numbers: List[int],
    temperature: float,
    batch_size: int = 1,
    populate_mesh_fc_cache: bool = True,
    make_mesh_tag_alias: bool = False,
    include_q2_coupling: bool = True,
    out_h5: Optional[str] = None,
    out_gamma_npz: Optional[str] = None,
    skip_assemble: bool = False,
    max_w_gb: Optional[float] = None,
    validate_gamma_detail: bool = False,
    fallback_lang: str = "NONE",
    force_recompute_gamma_detail: bool = False,
    lang: str = "GPU",
    ir_index_start: Optional[int] = None,
    ir_index_stop: Optional[int] = None,
) -> None:
    from mpi4py import MPI
    import phono3py as Phono3py
    from phonopy.phonon.grid import get_ir_grid_points

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    setup_gpu_for_rank(rank, size)
    comm.Barrier()

    tag = mesh_tag(mesh_numbers)
    fc_cache_dir = f"phonon_cache_{tag}"
    legacy_fc_cache_dir = "phonon_cache"
    ph3_cache_dir = f"phono3py_cache_{mesh_numbers[0]}x{mesh_numbers[1]}x{mesh_numbers[2]}"
    if out_h5 is None:
        out_h5 = f"W_scattering_mesh_{tag}.h5"
    if out_gamma_npz is None:
        out_gamma_npz = f"gamma_W_exact_{tag}_T{int(temperature)}.npz"

    if rank == 0:
        print("\n" + "=" * 78)
        print("W_SCATTER (dynamics/w_scatter) — gamma_detail + W + EXACT linewidth")
        print(f"MPI size: {size}")
        print(f"Mesh tag: {tag}")
        print(f"FC cache dir: `{fc_cache_dir}/` (legacy fallback: `{legacy_fc_cache_dir}/`)")
        print(f"Phono3py cache dir: `{ph3_cache_dir}/`")
        print(f"W output file: `{out_h5}`")
        print(f"Exact-gamma output: `{out_gamma_npz}`")
        print(f"Using local phono3py fork path: `{LOCAL_PHONO3PY_EINSUM}`")
        print("=" * 78 + "\n")

    comm.Barrier()
    ph3 = Phono3py.load(phono3py_yaml, log_level=1 if rank == 0 else 0)
    ph3.log_level = 1 if rank == 0 else 0

    set_phono3py_forces_with_mesh_fc_cache(
        ph3,
        fc2_forces=fc2_forces_npy,
        fc3_forces=fc3_forces_npy,
        cache_dir_mesh=fc_cache_dir,
        legacy_cache_dir=legacy_fc_cache_dir,
        populate_mesh_cache=populate_mesh_fc_cache,
        comm=comm,
    )

    ph3.mesh_numbers = mesh_numbers
    ph3.init_phph_interaction(symmetrize_fc3q=True)

    if rank == 0:
        print(f"Running harmonic phonon solver on mesh {mesh_numbers}", flush=True)
    t0 = time.time()
    ph3.run_phonon_solver()
    if rank == 0:
        print(f"Harmonic mesh done in {time.time() - t0:.1f}s", flush=True)

    ir_grid_points_grg, _, ir_grid_map = get_ir_grid_points(ph3.grid)
    ir_grid_points_bzg = np.array(ph3.grid.grg2bzg[ir_grid_points_grg], dtype="int64")

    qpoints_frac = np.dot(ph3.grid.addresses, ph3.grid.QDinv)
    qpoints_frac = unit_range_fixed(qpoints_frac, lval=1.0, eps=1e-9)

    bzg2grg = np.array(ph3.grid.bzg2grg, dtype="int64")
    ir_pos_for_grg = {int(grg): i for i, grg in enumerate(ir_grid_points_grg)}
    ir_grg_for_bzg = np.array(ir_grid_map[bzg2grg], dtype="int64")
    ir_grid_map_bzg = np.array(
        [ir_pos_for_grg[int(grg)] for grg in ir_grg_for_bzg], dtype="int64"
    )

    if rank == 0:
        print(f"Total BZ grid points: {len(qpoints_frac)}", flush=True)
        print(f"Total irreducible grid points: {len(ir_grid_points_bzg)}", flush=True)

    if ir_index_start is not None or ir_index_stop is not None:
        if not skip_assemble:
            raise RuntimeError("--ir-index-start/--ir-index-stop require --skip-assemble.")
        n_ir_total = len(ir_grid_points_bzg)
        start_idx = 0 if ir_index_start is None else int(ir_index_start)
        stop_idx = n_ir_total if ir_index_stop is None else int(ir_index_stop)
        if start_idx < 0 or stop_idx < start_idx or stop_idx > n_ir_total:
            raise ValueError(
                f"Invalid ir-index slice [{start_idx}:{stop_idx}] for {n_ir_total} ir points."
            )
        ir_grid_points_bzg = ir_grid_points_bzg[start_idx:stop_idx]
        if rank == 0:
            print(
                f"Restricting gamma generation to irreducible-index slice "
                f"[{start_idx}:{stop_idx}] ({len(ir_grid_points_bzg)} grid points).",
                flush=True,
            )

    compute_gamma_detail_on_mesh_gpu(
        ph3,
        temperature=temperature,
        mesh_numbers=mesh_numbers,
        cache_dir=ph3_cache_dir,
        ir_grid_points_bzg=ir_grid_points_bzg,
        batch_size=batch_size,
        lang=lang,
        make_mesh_tag_alias=make_mesh_tag_alias,
        validate_gamma_detail=validate_gamma_detail,
        fallback_lang=fallback_lang,
        force_recompute_gamma_detail=force_recompute_gamma_detail,
        comm=comm,
    )

    if rank != 0:
        return

    frequencies = ph3.get_phonon_data()[0]
    n_bands = frequencies.shape[-1]

    # ── PRIMARY OUTPUT: canonical EXACT linewidth from gamma_detail ──────────────
    # Computes both the RAW (canonical, unchanged) value and the degenerate-band-
    # averaged variant (see gamma_degenerate_average) -- the latter costs
    # microseconds (post-hoc reduction on the already-computed raw array).
    print("\n[exact-gamma] computing canonical linewidth via direct gamma_detail "
          "reduction (the primary output)...", flush=True)
    gamma_linewidth, gamma_linewidth_degenerate_avg = extract_gamma_exact(
        cache_dir=ph3_cache_dir,
        mesh_numbers=mesh_numbers,
        ir_grid_points_bzg=ir_grid_points_bzg,
        ir_grid_map_bzg=ir_grid_map_bzg,
        n_bands=n_bands,
        frequencies_bzg=frequencies,
        return_degenerate_avg=True,
    )
    # frequencies at the irreducible points (row i -> ir position).
    frequencies_ir = np.zeros((len(ir_grid_points_bzg), n_bands), dtype=np.float64)
    for gp in ir_grid_points_bzg:
        gp_i = int(gp)
        ir0 = int(ir_grid_map_bzg[gp_i])
        frequencies_ir[ir0, :] = frequencies[gp_i]

    save_gamma_exact_npz(
        out_npz=out_gamma_npz,
        gamma_linewidth=gamma_linewidth,
        ir_grid_points=ir_grid_points_bzg,
        ir_grid_map=ir_grid_map_bzg,
        qpoints_frac=qpoints_frac,
        frequencies_ir=frequencies_ir,
        mesh_numbers=mesh_numbers,
        temperature=temperature,
        n_bands=n_bands,
        gamma_linewidth_degenerate_avg=gamma_linewidth_degenerate_avg,
    )
    print(f"[exact-gamma] Saved canonical linewidth: `{out_gamma_npz}`", flush=True)

    if skip_assemble:
        print("Skipping W assembly (`--skip-assemble` set). Exact gamma already saved.",
              flush=True)
        return

    n_ir = len(ir_grid_points_bzg)
    n_states = n_ir * n_bands
    est_gb = (n_states * n_states * 4) / (1024**3)
    print(f"Allocating W_rate with {n_states} states (~{est_gb:.2f} GB at float32)", flush=True)
    if max_w_gb is not None and est_gb > max_w_gb:
        raise RuntimeError(
            f"Estimated W_rate memory ({est_gb:.2f} GB) exceeds --max-w-gb={max_w_gb:.2f}. "
            "Rerun with larger limit or use --skip-assemble."
        )

    w_rate, w_unsym = assemble_w_matrix(
        cache_dir=ph3_cache_dir,
        mesh_numbers=mesh_numbers,
        ir_grid_points_bzg=ir_grid_points_bzg,
        ir_grid_map_bzg=ir_grid_map_bzg,
        n_bands=n_bands,
        include_q2_coupling=include_q2_coupling,
        return_unsym=True,
    )

    # ── Self-consistency: 0.5*row_sum(unsym W) must equal the direct reduction ───
    gamma_from_unsym = 0.5 * w_unsym.sum(axis=1).reshape(n_ir, n_bands)
    sel = gamma_linewidth > 1e-9
    if sel.any():
        rel = np.abs(gamma_from_unsym[sel] - gamma_linewidth[sel]) / gamma_linewidth[sel]
        print(
            f"[self-consistency] max |0.5*rowsum(unsymW) - extract_gamma_exact| / gamma "
            f"= {rel.max():.3e}  (mean {rel.mean():.3e}, over {int(sel.sum())} modes; "
            f"expect ~float32 storage level)",
            flush=True,
        )
    # LEGACY diagnostic: row_sum(symmetrized W).
    gamma_legacy = w_rate.sum(axis=1).reshape(n_ir, n_bands).astype(np.float64)

    w_tau = np.zeros_like(w_rate, dtype=np.float32)
    positive = w_rate > 0
    w_tau[positive] = (1.0 / (4.0 * np.pi * w_rate[positive])).astype(np.float32)

    lattice = np.array(ph3.primitive.cell, dtype=float)
    reciprocal_lattice = build_reciprocal_from_direct(lattice)
    save_w_matrix_hdf5(
        out_h5=out_h5,
        lattice=lattice,
        reciprocal_lattice=reciprocal_lattice,
        qpoints_frac=qpoints_frac,
        ir_grid_points=ir_grid_points_bzg,
        ir_grid_map=ir_grid_map_bzg,
        w_rate=w_rate,
        w_tau=w_tau,
        mesh_numbers=mesh_numbers,
        temperature=temperature,
        n_bands=n_bands,
        gamma_linewidth=gamma_linewidth,
        gamma_legacy_rowsum_symW=gamma_legacy,
        gamma_linewidth_degenerate_avg=gamma_linewidth_degenerate_avg,
    )
    print(
        f"Saved W matrix (+ canonical gamma_linewidth, degenerate-avg variant, "
        f"legacy diagnostic): `{out_h5}`",
        flush=True,
    )
