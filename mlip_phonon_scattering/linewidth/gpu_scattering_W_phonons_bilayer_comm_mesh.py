"""
GPU-parallel commensurate MoSe2-WSe2 bilayer phonon scattering workflow.

This script mirrors the CPU comm-mesh workflow but computes gamma_detail on GPUs:
1) Load phono3py, forces, and cached force constants.
2) Split irreducible grid points across MPI ranks, one GPU per rank.
3) Run `run_imag_self_energy(..., lang="GPU")` to write gamma_detail HDF5 files.
4) Optionally assemble and save symmetric W_rate / W_tau matrix on rank 0.

Example:
  srun -N 4 -n 16 --gpus-per-task=1 \
      python gpu_scattering_W_phonons_bilayer_comm_mesh.py \
      --mesh 12 12 1 --temperature 300 --batch-size 1
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from typing import List, Optional

import numpy as np
from mpi4py import MPI

# The forked phono3py (phono3py_einsum, which provides the lang="GPU" path) is
# pip-installed editable, so `import phono3py` already resolves to it.
# PHONO3PY_EINSUM_PATH (exported by load_mace_phonon_env.sh) is an optional
# belt-and-suspenders override. The former repo-relative guess (../../phono3py_einsum)
# is invalid now that this module lives inside the installed package, so it is removed.
LOCAL_PHONO3PY_EINSUM = os.environ.get("PHONO3PY_EINSUM_PATH")
if LOCAL_PHONO3PY_EINSUM and os.path.isdir(LOCAL_PHONO3PY_EINSUM) and LOCAL_PHONO3PY_EINSUM not in sys.path:
    sys.path.insert(0, LOCAL_PHONO3PY_EINSUM)

try:
    import h5py

    HAS_H5PY = True
except Exception:
    HAS_H5PY = False

import phono3py as Phono3py
from phonopy.phonon.grid import get_ir_grid_points


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
    cache_dir: str, mesh_numbers: List[int], gp: int, temperature: Optional[float] = None
) -> List[str]:
    """Return accepted gamma_detail filename variants for a grid point.

    When *temperature* is provided the preferred filename includes a T-tag
    (``T{temp:.1f}K``) immediately before the mesh tag, mirroring the already-
    correct T-tagged convention used for reduced-gamma filenames.  Legacy
    untagged filenames are also returned as fallback candidates so that existing
    unambiguous single-T caches remain readable without a migration step.

    Callers that find an existing untagged file MUST verify that it was computed
    at *temperature* before reuse (see ``check_cached_gamma_detail_temperature``).
    """
    mx, my, mz = mesh_numbers
    concat = "m" + "".join(map(str, mesh_numbers))
    explicit = f"m{mx}x{my}x{mz}"
    if temperature is not None:
        t_tag = f"T{temperature:.1f}K"
        return [
            os.path.join(cache_dir, f"gamma_detail-{t_tag}-{concat}-g{gp}.hdf5"),
            os.path.join(cache_dir, f"gamma_detail-{t_tag}-{explicit}-g{gp}.hdf5"),
            # Legacy untagged fallbacks — must be temperature-verified before use.
            os.path.join(cache_dir, f"gamma_detail-{concat}-g{gp}.hdf5"),
            os.path.join(cache_dir, f"gamma_detail-{explicit}-g{gp}.hdf5"),
        ]
    return [
        os.path.join(cache_dir, f"gamma_detail-{concat}-g{gp}.hdf5"),
        os.path.join(cache_dir, f"gamma_detail-{explicit}-g{gp}.hdf5"),
    ]


def check_cached_gamma_detail_temperature(
    gamma_file: str, expected_temperature: float, rtol: float = 1e-3
) -> bool:
    """Return True if the cached gamma_detail file matches *expected_temperature*.

    Checks the HDF5 ``temperature_K`` attribute.  Returns True (safe to reuse)
    if the attribute is absent (old cache format — assume correct since production
    currently uses per-T directories).  Returns False on temperature mismatch.
    """
    if not HAS_H5PY:
        return True  # Can't verify without h5py — trust the cache.
    try:
        with h5py.File(gamma_file, "r") as f:
            if "temperature_K" not in f.attrs:
                return True  # Old format: no T info stored, assume correct.
            cached_T = float(f.attrs["temperature_K"])
            return abs(cached_T - expected_temperature) <= rtol * max(abs(expected_temperature), 1.0)
    except Exception:
        return True  # Unreadable file — let the existing signal check handle it.


def choose_existing_gamma_detail_file(
    cache_dir: str,
    mesh_numbers: List[int],
    gp: int,
    temperature: Optional[float] = None,
) -> Optional[str]:
    """Return the first existing gamma_detail file candidate, or None.

    When *temperature* is provided, cached files are verified to have been
    computed at that temperature (via ``check_cached_gamma_detail_temperature``).
    If a legacy untagged file is found at a mismatched temperature, it is skipped
    and the caller will recompute the grid point.
    """
    for fn in gamma_detail_filename_candidates(cache_dir, mesh_numbers, gp, temperature):
        if os.path.exists(fn):
            if temperature is not None and not check_cached_gamma_detail_temperature(fn, temperature):
                # Temperature mismatch on a legacy untagged cache — do not reuse.
                continue
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
    """Heuristic check that gamma_detail is not trivially all-zero.

    The current GPU backend may emit all-zero gamma_detail while still returning
    non-zero total gammas. Scan the dataset in bounded slabs so sparse but valid
    gamma_detail files are not mistaken for all-zero output.
    """
    if not HAS_H5PY:
        return True

    try:
        with h5py.File(gamma_file, "r") as f:
            if "gamma_detail" not in f:
                return False
            ds = f["gamma_detail"]
            if ds.size == 0:
                return False

            # A full file can be tens of MB. Read along the leading axis first,
            # then fall back to triplet slabs for the common gamma_detail shapes.
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


def set_phono3py_forces_with_mesh_fc_cache(
    ph3: Phono3py,
    fc2_forces: str,
    fc3_forces: str,
    cache_dir_mesh: str,
    legacy_cache_dir: str = "phonon_cache",
    populate_mesh_cache: bool = True,
    comm: Optional[MPI.Comm] = None,
) -> None:
    """Attach forces and build/load cached fc2/fc3, with legacy fallback."""
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


def compute_gamma_detail_on_mesh_gpu(
    ph3: Phono3py,
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
    comm: Optional[MPI.Comm] = None,
) -> None:
    """Ensure gamma_detail files exist for irreducible grid points, using GPU backend."""
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
        existing = choose_existing_gamma_detail_file(cache_dir, mesh_numbers, gp_i, temperature)
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
                    f"Recomputing grid point {gp_i}: cached file appears invalid/all-zero ({os.path.basename(existing)}).",
                    flush=True,
                )
        elif rank == 0:
            print(f"Skipping grid point {gp_i}: already cached ({os.path.basename(existing)})", flush=True)
    remaining_gps = np.array(remaining_gps, dtype="int64")

    my_grid_points = np.array(remaining_gps[rank::size], dtype="int64")
    if rank == 0:
        print(f"Grid points to compute: {len(remaining_gps)} (after filtering)", flush=True)
        if validate_gamma_detail and invalid_cached_gps:
            print(f"Found {len(invalid_cached_gps)} invalid cached gamma_detail files.", flush=True)
        print(f"Running with {size} MPI processes", flush=True)
        print("GPU backend requires tetrahedron integration (sigma=None).", flush=True)
    print(f"Rank {rank}: assigned {len(my_grid_points)} grid points", flush=True)

    # Ensure no sigma-based smearing is used in GPU path.
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
                    fn = choose_existing_gamma_detail_file(cache_dir, mesh_numbers, int(gp), temperature)
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
                    f"Repairing {len(flat_bad)} gamma_detail files using lang='{fallback_lang.upper()}'...",
                    flush=True,
                )
            my_repair = np.array(flat_bad[rank::size], dtype="int64")
            for start in range(0, len(my_repair), batch_size):
                gp_chunk = my_repair[start : start + batch_size]
                print(
                    f"Rank {rank}: repairing grid points {gp_chunk.tolist()} with lang={fallback_lang.upper()}",
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


def assemble_symmetric_w_matrix(
    cache_dir: str,
    mesh_numbers: List[int],
    ir_grid_points_bzg: np.ndarray,
    ir_grid_map_bzg: np.ndarray,
    n_bands: int,
    include_q2_coupling: bool = True,
) -> np.ndarray:
    """Assemble symmetric W_rate matrix over irreducible q points."""
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
            gamma_detail = f["gamma_detail"][0]  # shape: (T=1, triplet, b0, b1, b2) -> index sigma
            triplet = f["triplet"][:]
            weight = f["weight"][:]

        ir0 = int(ir_grid_map_bzg[gp_i])
        row_slice = slice(ir0 * n_bands, (ir0 + 1) * n_bands)

        # Vectorized accumulation by grouping triplets that map to same irreducible q.
        # This removes the per-triplet Python inner loop.
        ir1_all = ir_grid_map_bzg[triplet[:, 1]].astype(np.int64, copy=False)
        contrib_q1_all = np.sum(gamma_detail, axis=3)  # (n_triplet, b0, b1)
        weighted_q1 = contrib_q1_all * weight[:, None, None]
        for ir1 in np.unique(ir1_all):
            mask = ir1_all == ir1
            col_slice1 = slice(int(ir1) * n_bands, (int(ir1) + 1) * n_bands)
            w_rate[row_slice, col_slice1] += np.sum(weighted_q1[mask], axis=0, dtype=np.float64).astype(
                np.float32, copy=False
            )

        if include_q2_coupling:
            ir2_all = ir_grid_map_bzg[triplet[:, 2]].astype(np.int64, copy=False)
            contrib_q2_all = np.sum(gamma_detail, axis=2)  # (n_triplet, b0, b2)
            weighted_q2 = contrib_q2_all * weight[:, None, None]
            for ir2 in np.unique(ir2_all):
                mask = ir2_all == ir2
                col_slice2 = slice(int(ir2) * n_bands, (int(ir2) + 1) * n_bands)
                w_rate[row_slice, col_slice2] += np.sum(weighted_q2[mask], axis=0, dtype=np.float64).astype(
                    np.float32, copy=False
                )

        if igp % 10 == 0 or igp == n_ir:
            elapsed = time.time() - t_start
            rate = igp / elapsed if elapsed > 0 else 0.0
            print(
                f"[W assembly] processed {igp}/{n_ir} irreducible q-points "
                f"({rate:.2f} q/s, {elapsed:.1f}s elapsed)",
                flush=True,
            )

    w_rate = 0.5 * (w_rate + w_rate.T)
    return w_rate


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
) -> None:
    """Write assembled W_rate / W_tau and metadata to HDF5."""
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

        h5.attrs["mesh_numbers"] = np.array(mesh_numbers, dtype=int)
        h5.attrs["temperature_K"] = float(temperature)
        h5.attrs["qpoints_convention"] = "fractional in primitive reciprocal basis"
        h5.attrs["state_indexing"] = "s = ir_q_index * nband + band_index"
        h5.attrs["nband"] = int(n_bands)
        h5.attrs["Nq_ir"] = int(len(ir_grid_points))
        h5.attrs["Nstates"] = int(w_rate.shape[0])
        h5.attrs["grid_index_convention"] = "phono3py BZ-grid indexing (matches gamma_detail triplet indices)"
        h5.attrs["W_q_indexing_scope"] = "ir"


def run_scattering(
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
    skip_assemble: bool = False,
    max_w_gb: Optional[float] = None,
    validate_gamma_detail: bool = False,
    fallback_lang: str = "NONE",
    force_recompute_gamma_detail: bool = False,
    lang: str = "GPU",
    ir_index_start: Optional[int] = None,
    ir_index_stop: Optional[int] = None,
) -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    setup_gpu_for_rank(rank, size)
    comm.Barrier()

    tag = mesh_tag(mesh_numbers)
    fc_cache_dir = f"phonon_cache_{tag}"
    legacy_fc_cache_dir = "phonon_cache"
    # T-tag the phono3py cache directory so multiple temperatures never share
    # the same gamma_detail files (mirrors the already-correct T-tagged convention
    # used for reduced-gamma output).
    ph3_cache_dir = (
        f"phono3py_cache_{mesh_numbers[0]}x{mesh_numbers[1]}x{mesh_numbers[2]}"
        f"_T{temperature:.1f}K"
    )
    if out_h5 is None:
        out_h5 = f"W_scattering_mesh_{tag}.h5"

    if rank == 0:
        print("\n" + "=" * 78)
        print("GPU COMMENSURATE BILAYER PHONON SCATTERING MATRIX ON FULL MESH")
        print(f"MPI size: {size}")
        print(f"Mesh tag: {tag}")
        print(f"FC cache dir: `{fc_cache_dir}/` (legacy fallback: `{legacy_fc_cache_dir}/`)")
        print(f"Phono3py cache dir: `{ph3_cache_dir}/`")
        print(f"Output file: `{out_h5}`")
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
    ir_grid_map_bzg = np.array([ir_pos_for_grg[int(grg)] for grg in ir_grg_for_bzg], dtype="int64")

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
                f"Invalid ir-index slice [{start_idx}:{stop_idx}] for {n_ir_total} irreducible points."
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

    if skip_assemble:
        print("Skipping W assembly (`--skip-assemble` set).", flush=True)
        return

    frequencies = ph3.get_phonon_data()[0]
    n_bands = frequencies.shape[-1]
    n_ir = len(ir_grid_points_bzg)
    n_states = n_ir * n_bands
    est_gb = (n_states * n_states * 4) / (1024**3)
    print(f"Allocating W_rate with {n_states} states (~{est_gb:.2f} GB at float32)", flush=True)
    if max_w_gb is not None and est_gb > max_w_gb:
        raise RuntimeError(
            f"Estimated W_rate memory ({est_gb:.2f} GB) exceeds --max-w-gb={max_w_gb:.2f}. "
            "Rerun with larger limit or use --skip-assemble."
        )

    w_rate = assemble_symmetric_w_matrix(
        cache_dir=ph3_cache_dir,
        mesh_numbers=mesh_numbers,
        ir_grid_points_bzg=ir_grid_points_bzg,
        ir_grid_map_bzg=ir_grid_map_bzg,
        n_bands=n_bands,
        include_q2_coupling=include_q2_coupling,
    )

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
    )
    print(f"Saved W matrix: `{out_h5}`", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GPU MPI driver for commensurate bilayer phonon scattering matrix on full mesh. "
            "Uses run_imag_self_energy(..., lang='GPU') to write gamma_detail."
        )
    )
    parser.add_argument(
        "--phono3py-yaml",
        default="MoSe2-WSe2_bilayer_relaxed_phono3py_displacements.yaml",
    )
    parser.add_argument(
        "--fc2-forces",
        default="MoSe2-WSe2_bilayer_relaxed_forces_2nd_from_3rd.npy",
    )
    parser.add_argument(
        "--fc3-forces",
        default="MoSe2-WSe2_bilayer_relaxed_forces_3rd.npy",
    )

    parser.add_argument("--mesh", nargs=3, type=int, default=[12, 12, 1], metavar=("MX", "MY", "MZ"))
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--lang",
        default="GPU",
        help="Primary backend language passed to run_imag_self_energy.",
    )
    parser.add_argument("--output", default=None, help="Output HDF5 filename for assembled W matrix.")

    parser.add_argument(
        "--skip-assemble",
        action="store_true",
        help="Only generate gamma_detail cache on GPU; do not assemble W matrix.",
    )
    parser.add_argument(
        "--no-validate-gamma-detail",
        action="store_true",
        help="Skip post-run validation for all-zero gamma_detail files (default behavior).",
    )
    parser.add_argument(
        "--validate-gamma-detail",
        action="store_true",
        help="Enable post-run validation for all-zero gamma_detail files.",
    )
    parser.add_argument(
        "--fallback-lang",
        choices=["C", "NONE", "None", "none"],
        default="NONE",
        help=(
            "Fallback backend used to repair invalid/all-zero gamma_detail files "
            "after GPU generation. Default is NONE (GPU-only flow)."
        ),
    )
    parser.add_argument(
        "--force-recompute-gamma-detail",
        action="store_true",
        help="Ignore existing gamma_detail cache and recompute all irreducible grid points.",
    )
    parser.add_argument(
        "--max-w-gb",
        type=float,
        default=None,
        help="Abort W assembly if estimated dense W_rate memory exceeds this value (GB).",
    )
    parser.add_argument(
        "--no-populate-mesh-fc-cache",
        action="store_true",
        help="Do not copy legacy `phonon_cache/` fc2/fc3 into mesh-tagged cache.",
    )
    parser.add_argument(
        "--make-mesh-tag-alias",
        action="store_true",
        help="Create mesh-tagged gamma_detail alias files next to canonical names.",
    )
    parser.add_argument(
        "--no-q2-coupling",
        action="store_true",
        help="Do not add q0->q2 coupling term (only q0->q1).",
    )
    parser.add_argument(
        "--ir-index-start",
        type=int,
        default=None,
        help="Start index in the irreducible grid-point list for gamma generation subsets.",
    )
    parser.add_argument(
        "--ir-index-stop",
        type=int,
        default=None,
        help="Stop index in the irreducible grid-point list for gamma generation subsets.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_scattering(
        phono3py_yaml=args.phono3py_yaml,
        fc2_forces_npy=args.fc2_forces,
        fc3_forces_npy=args.fc3_forces,
        mesh_numbers=list(args.mesh),
        temperature=args.temperature,
        batch_size=args.batch_size,
        populate_mesh_fc_cache=not args.no_populate_mesh_fc_cache,
        make_mesh_tag_alias=args.make_mesh_tag_alias,
        include_q2_coupling=not args.no_q2_coupling,
        out_h5=args.output,
        skip_assemble=args.skip_assemble,
        max_w_gb=args.max_w_gb,
        validate_gamma_detail=args.validate_gamma_detail and (not args.no_validate_gamma_detail),
        fallback_lang=args.fallback_lang,
        force_recompute_gamma_detail=args.force_recompute_gamma_detail,
        lang=args.lang,
        ir_index_start=args.ir_index_start,
        ir_index_stop=args.ir_index_stop,
    )


if __name__ == "__main__":
    main()
