#!/usr/bin/env python3
"""Build a sparse full-BZ three-phonon collision cache.

The output is designed for arbitrary mode-resolved occupations n[q, band].
It stores physical q indices for each process, while retaining phono3py's
interaction-strength and tetrahedron-weight conventions.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

import h5py
import numpy as np

from .convert_pp_cache_to_csr import convert_cache
from .convert_pp_cache_to_csr import _chunk_shape as _csr_chunk_shape
from .convert_pp_cache_to_csr import _mode_dtype as _csr_mode_dtype


DYNAMICS_PROCESS_NAMES = PROCESS_NAMES = ("abs_plus", "abs_minus", "decay")
LINEWIDTH_PROCESS_NAME = "gamma_detail"
DEFAULT_SUPPORT_POLICY = "existing_support"


def _install_local_phono3py() -> str | None:
    here = Path(__file__).resolve().parent
    candidates = []
    env_path = os.environ.get("PHONO3PY_EINSUM_PATH")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            here.parents[2] / "phonopy_codes" / "phono3py_einsum",
            here.parents[2] / "phonon_codes" / "phono3py_einsum",
            here.parents[1] / "phonopy_codes" / "phono3py_einsum",
            here.parents[1] / "phonon_codes" / "phono3py_einsum",
        ]
    )
    for path in candidates:
        if path.is_dir():
            path_s = str(path)
            if path_s not in sys.path:
                sys.path.insert(0, path_s)
            return path_s
    return None


LOCAL_PHONO3PY = _install_local_phono3py()

try:
    from mpi4py import MPI
except Exception:  # pragma: no cover - exercised only without MPI.
    MPI = None

Phono3py = None
get_grid_point_from_address_py = None
ImagSelfEnergy = None
get_detailed_imag_self_energy_from_g = None
get_triplets_integration_weights = None


def _require_phono3py() -> None:
    global Phono3py
    global get_grid_point_from_address_py
    global ImagSelfEnergy
    global get_detailed_imag_self_energy_from_g
    global get_triplets_integration_weights
    if Phono3py is not None:
        return
    try:
        import phono3py as _Phono3py
        from phonopy.phonon.grid import get_grid_point_from_address_py as _get_grid_point
        from phono3py.phonon3.imag_self_energy import ImagSelfEnergy as _ImagSelfEnergy
        from phono3py.phonon3.imag_self_energy import (
            get_detailed_imag_self_energy_from_g as _get_detailed_imag_self_energy_from_g,
        )
        from phono3py.phonon3.triplets import (
            get_triplets_integration_weights as _get_triplets_integration_weights,
        )
    except Exception as exc:
        raise RuntimeError(
            "phono3py/phonopy dependencies are required to build collision caches. "
            "Pure CSR helper imports do not require them."
        ) from exc
    Phono3py = _Phono3py
    get_grid_point_from_address_py = _get_grid_point
    ImagSelfEnergy = _ImagSelfEnergy
    get_detailed_imag_self_energy_from_g = _get_detailed_imag_self_energy_from_g
    get_triplets_integration_weights = _get_triplets_integration_weights


PROCESS_DESCRIPTIONS = {
    "abs_plus": "q0 + q1 -> q2; delta(w0 + w1 - w2)",
    "abs_minus": "q0 + q1 -> q2 using swapped phono3py legs; delta(w0 - w1 + w2)",
    "decay": "q0 -> q1 + q2; delta(w0 - w1 - w2)",
    LINEWIDTH_PROCESS_NAME: (
        "weighted phono3py gamma_detail contribution for the requested "
        "linewidth temperature; q1/q2 are stored without dynamics inverse mapping"
    ),
}


def _default_symmetrized_output(raw_output: Path) -> Path:
    return raw_output.with_name(f"{raw_output.stem}_symmetrized{raw_output.suffix}")


def _postprocess_symmetrized_cache(args: argparse.Namespace, raw_output: Path, rank: int) -> Path | None:
    """Run the validated full-grid symmetrization postprocessor on rank 0."""
    if not args.symmetrize_full_grid:
        return None
    if rank != 0:
        return None

    from symmetrize_pp_collision_cache import symmetrize_cache

    sym_out = Path(args.symmetrized_output).resolve() if args.symmetrized_output else _default_symmetrized_output(raw_output)
    if sym_out == raw_output:
        raise ValueError(
            "--symmetrized-output must differ from the raw CSR output; "
            "the integration path preserves the raw cache for provenance."
        )

    print(f"Symmetrizing raw CSR cache {raw_output} -> {sym_out}", flush=True)
    summary = symmetrize_cache(
        raw_output,
        Path(args.symmetry_metadata).resolve(),
        sym_out,
        list(DYNAMICS_PROCESS_NAMES),
        degenerate_policy=args.degenerate_policy,
        support_policy=args.support_policy,
        sum_dtype="float64",
        compression=args.symmetrization_compression,
        chunk_rows=args.symmetrization_chunk_rows,
    )
    print(
        "Wrote symmetrized cache "
        f"{sym_out} with support_policy={summary['support_policy']}",
        flush=True,
    )
    return sym_out


def _h5py_mpi_enabled() -> bool:
    try:
        return bool(h5py.get_config().mpi)
    except Exception:
        return False


def _parallel_filter_available(comm, compression: str | None) -> bool:
    if MPI is None:
        raise RuntimeError("Direct CSR output requires mpi4py and MPI-enabled h5py.")
    if not _h5py_mpi_enabled():
        raise RuntimeError(
            "Direct CSR output requires MPI-enabled h5py. On Perlmutter, load "
            "cray-hdf5-parallel/1.14.3.7 and rebuild h5py in the project env with "
            "HDF5_MPI=ON CC=cc HDF5_DIR=$HDF5_DIR python -m pip install --no-binary=h5py "
            "--force-reinstall h5py==3.13.0."
        )

    if compression is None:
        return True

    rank = comm.Get_rank()
    probe = Path(os.environ.get("TMPDIR", "/tmp")) / f"pp_csr_parallel_filter_probe_{os.environ.get('SLURM_JOB_ID', os.getpid())}.h5"
    if rank == 0 and probe.exists():
        probe.unlink()
    comm.Barrier()
    ok = True
    try:
        with h5py.File(probe, "w", driver="mpio", comm=comm) as h5:
            ds = h5.create_dataset(
                "x",
                shape=(comm.Get_size(),),
                chunks=(comm.Get_size(),),
                dtype=np.int64,
                compression=compression,
            )
            with ds.collective:
                ds[rank : rank + 1] = np.asarray([rank], dtype=np.int64)
    except Exception:
        ok = False
    comm.Barrier()
    if rank == 0 and probe.exists():
        probe.unlink()
    comm.Barrier()
    return bool(comm.allreduce(ok, op=MPI.LAND))


def _comm():
    if MPI is None:
        return None, 0, 1
    comm = MPI.COMM_WORLD
    return comm, comm.Get_rank(), comm.Get_size()


def _barrier(comm) -> None:
    if comm is not None:
        comm.Barrier()


def _rank_print(rank: int, *items, **kwargs) -> None:
    if rank == 0:
        print(*items, **kwargs, flush=True)


def _setup_gpu_for_rank(lang: str, rank: int) -> None:
    if not lang.upper().startswith("GPU"):
        return
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError("GPU backend requested but CuPy is not importable.") from exc

    n_devices = cp.cuda.runtime.getDeviceCount()
    if n_devices < 1:
        raise RuntimeError("GPU backend requested but no CUDA devices are visible.")
    local_rank = int(os.environ.get("SLURM_LOCALID", rank))
    dev_id = local_rank % n_devices
    cp.cuda.Device(dev_id).use()
    free_mem, total_mem = cp.cuda.Device(dev_id).mem_info
    print(
        f"[rank {rank}] CUDA device {dev_id}: "
        f"{free_mem / 1e9:.2f} GB free / {total_mem / 1e9:.2f} GB total",
        flush=True,
    )


def _mesh_tag(mesh: Iterable[int]) -> str:
    mx, my, mz = [int(x) for x in mesh]
    return f"{mx}x{my}x{mz}"


def _set_forces_and_fc_cache(
    ph3,
    fc2_forces: Path,
    fc3_forces: Path,
    cache_dir: Path,
    rank: int,
    comm,
) -> None:
    ph3.phonon_forces = np.load(fc2_forces)
    ph3.forces = np.load(fc3_forces)

    fc2_file = cache_dir / "fc2.npy"
    fc3_file = cache_dir / "fc3.npy"
    if rank == 0:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if not (fc2_file.exists() and fc3_file.exists()):
            print(f"Computing force constants in {cache_dir} ...", flush=True)
            ph3.produce_fc3(symmetrize_fc3r=True)
            ph3.produce_fc2(symmetrize_fc2=True)
            np.save(fc3_file, ph3.fc3, allow_pickle=True)
            np.save(fc2_file, ph3.fc2, allow_pickle=True)
    _barrier(comm)

    ph3.fc3 = np.load(fc3_file, allow_pickle=True)
    ph3.fc2 = np.load(fc2_file, allow_pickle=True)


def _build_regular_grid_metadata(grid):
    _require_phono3py()
    num_grg = int(np.prod(grid.D_diag))
    grg2bzg = np.asarray(grid.grg2bzg[:num_grg], dtype=np.int64)
    bzg2grg = np.asarray(grid.bzg2grg, dtype=np.int64)
    addresses_grg = np.asarray(grid.addresses[grg2bzg], dtype=np.int64)
    qpoints_frac = np.dot(addresses_grg, grid.QDinv)
    qpoints_frac = qpoints_frac % 1.0
    qpoints_frac[np.isclose(qpoints_frac, 1.0) | np.isclose(qpoints_frac, 0.0)] = 0.0

    inverse_grg = np.empty(num_grg, dtype=np.int64)
    for grg, address in enumerate(addresses_grg):
        inverse_grg[grg] = get_grid_point_from_address_py(-address, grid.D_diag)

    return {
        "num_grg": num_grg,
        "grg2bzg": grg2bzg,
        "bzg2grg": bzg2grg,
        "addresses_grg": addresses_grg,
        "qpoints_frac": qpoints_frac,
        "inverse_grg": inverse_grg,
    }


def _create_process_datasets(h5: h5py.File, name: str, kernel_dtype: np.dtype) -> None:
    grp = h5.require_group(f"processes/{name}")
    if "index" not in grp:
        grp.create_dataset(
            "index",
            shape=(0, 6),
            maxshape=(None, 6),
            chunks=(min(16384, 1 << 20), 6),
            dtype=np.int64,
            compression="gzip",
        )
        grp.create_dataset(
            "kernel",
            shape=(0,),
            maxshape=(None,),
            chunks=(min(16384, 1 << 20),),
            dtype=kernel_dtype,
            compression="gzip",
        )
        grp.attrs["description"] = PROCESS_DESCRIPTIONS[name]
        grp.attrs["index_columns"] = "q0, band0, q1, band1, q2, band2"


def _append_process_events(
    h5: h5py.File,
    name: str,
    index: np.ndarray,
    kernel: np.ndarray,
    kernel_dtype: np.dtype,
) -> None:
    if len(kernel) == 0:
        return
    _create_process_datasets(h5, name, kernel_dtype)
    grp = h5[f"processes/{name}"]
    old = grp["kernel"].shape[0]
    new = old + len(kernel)
    grp["index"].resize((new, 6))
    grp["kernel"].resize((new,))
    grp["index"][old:new] = index
    grp["kernel"][old:new] = kernel.astype(kernel_dtype, copy=False)


def _event_index(
    q0: np.ndarray,
    b0: np.ndarray,
    q1: np.ndarray,
    b1: np.ndarray,
    q2: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    return np.column_stack((q0, b0, q1, b1, q2, b2)).astype(np.int64, copy=False)


def _append_nonzero_kernel(
    h5: h5py.File,
    name: str,
    kernel: np.ndarray,
    triplets_grg: np.ndarray,
    inverse_grg: np.ndarray,
    band_indices: np.ndarray,
    threshold: float,
    kernel_dtype: np.dtype,
) -> int:
    nz = np.nonzero(np.abs(kernel) > threshold)
    if len(nz[0]) == 0:
        return 0

    ti, b0_local, b1, b2 = nz
    q0 = triplets_grg[ti, 0]
    b0 = band_indices[b0_local]

    if name == "abs_plus":
        q1 = triplets_grg[ti, 1]
        q2 = inverse_grg[triplets_grg[ti, 2]]
    elif name == "abs_minus":
        q1 = triplets_grg[ti, 2]
        q2 = inverse_grg[triplets_grg[ti, 1]]
    elif name == "decay":
        q1 = inverse_grg[triplets_grg[ti, 1]]
        q2 = inverse_grg[triplets_grg[ti, 2]]
    else:  # pragma: no cover
        raise ValueError(name)

    index = _event_index(q0, b0, q1, b1, q2, b2)
    _append_process_events(h5, name, index, kernel[nz], kernel_dtype)
    return len(index)


def _compact_nonzero_kernel(
    name: str,
    kernel: np.ndarray,
    triplets_grg: np.ndarray,
    inverse_grg: np.ndarray,
    band_indices: np.ndarray,
    threshold: float,
    nband: int,
    mode_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nz = np.nonzero(np.abs(kernel) > threshold)
    if len(nz[0]) == 0:
        empty_mode = np.empty(0, dtype=mode_dtype)
        return empty_mode, empty_mode.copy(), empty_mode.copy(), np.empty(0, dtype=kernel.dtype)

    ti, b0_local, b1, b2 = nz
    q0 = triplets_grg[ti, 0]
    b0 = band_indices[b0_local]

    if name == "abs_plus":
        q1 = triplets_grg[ti, 1]
        q2 = inverse_grg[triplets_grg[ti, 2]]
    elif name == "abs_minus":
        q1 = triplets_grg[ti, 2]
        q2 = inverse_grg[triplets_grg[ti, 1]]
    elif name == "decay":
        q1 = inverse_grg[triplets_grg[ti, 1]]
        q2 = inverse_grg[triplets_grg[ti, 2]]
    else:  # pragma: no cover
        raise ValueError(name)

    target = (q0 * nband + b0).astype(mode_dtype, copy=False)
    mode1 = (q1 * nband + b1).astype(mode_dtype, copy=False)
    mode2 = (q2 * nband + b2).astype(mode_dtype, copy=False)
    return target, mode1, mode2, kernel[nz]


def _compact_nonzero_triplet_kernel(
    kernel: np.ndarray,
    triplets_grg: np.ndarray,
    band_indices: np.ndarray,
    threshold: float,
    nband: int,
    mode_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nz = np.nonzero(np.abs(kernel) > threshold)
    if len(nz[0]) == 0:
        empty_mode = np.empty(0, dtype=mode_dtype)
        return empty_mode, empty_mode.copy(), empty_mode.copy(), np.empty(0, dtype=kernel.dtype)

    ti, b0_local, b1, b2 = nz
    q0 = triplets_grg[ti, 0]
    b0 = band_indices[b0_local]
    q1 = triplets_grg[ti, 1]
    q2 = triplets_grg[ti, 2]

    target = (q0 * nband + b0).astype(mode_dtype, copy=False)
    mode1 = (q1 * nband + b1).astype(mode_dtype, copy=False)
    mode2 = (q2 * nband + b2).astype(mode_dtype, copy=False)
    return target, mode1, mode2, kernel[nz]


def _create_compact_spill_datasets(
    h5: h5py.File,
    name: str,
    mode_dtype: np.dtype,
    kernel_dtype: np.dtype,
    chunk_size: int,
) -> None:
    grp = h5.require_group(f"processes/{name}")
    # Use chunk_size directly — _csr_chunk_shape(0, ...) would return (1,) because
    # it caps at n_items=0, bloating shard files with per-element gzip chunks.
    chunk = (chunk_size,)
    for ds_name, dtype in (
        ("target", mode_dtype),
        ("mode1", mode_dtype),
        ("mode2", mode_dtype),
        ("kernel", kernel_dtype),
    ):
        if ds_name not in grp:
            grp.create_dataset(
                ds_name,
                shape=(0,),
                maxshape=(None,),
                chunks=chunk,
                dtype=dtype,
            )


def _append_compact_process_events(
    h5: h5py.File,
    name: str,
    target: np.ndarray,
    mode1: np.ndarray,
    mode2: np.ndarray,
    kernel: np.ndarray,
    mode_dtype: np.dtype,
    kernel_dtype: np.dtype,
    chunk_size: int,
) -> None:
    if len(kernel) == 0:
        return
    _create_compact_spill_datasets(h5, name, mode_dtype, kernel_dtype, chunk_size)
    grp = h5[f"processes/{name}"]
    old = grp["kernel"].shape[0]
    new = old + len(kernel)
    for ds_name in ("target", "mode1", "mode2", "kernel"):
        grp[ds_name].resize((new,))
    grp["target"][old:new] = target.astype(mode_dtype, copy=False)
    grp["mode1"][old:new] = mode1.astype(mode_dtype, copy=False)
    grp["mode2"][old:new] = mode2.astype(mode_dtype, copy=False)
    grp["kernel"][old:new] = kernel.astype(kernel_dtype, copy=False)


def _write_metadata(
    h5: h5py.File,
    ph3,
    metadata: dict,
    mesh: list[int],
    args: argparse.Namespace,
    nband: int,
) -> None:
    system = h5.require_group("system")
    grid = h5.require_group("grid")
    modes = h5.require_group("modes")

    def put(group, name, value, **attrs):
        if name in group:
            del group[name]
        ds = group.create_dataset(name, data=value, compression="gzip" if np.ndim(value) else None)
        for key, val in attrs.items():
            ds.attrs[key] = val

    lattice = np.asarray(ph3.primitive.cell, dtype=np.float64)
    reciprocal_lattice = 2.0 * np.pi * np.linalg.inv(lattice)
    frequencies = ph3.get_phonon_data()[0][metadata["grg2bzg"]]

    put(system, "mesh", np.asarray(mesh, dtype=np.int64))
    put(system, "lattice", lattice, units="Angstrom")
    put(system, "reciprocal_lattice", reciprocal_lattice, units="1/Angstrom, 2pi convention")
    put(grid, "grid_address", metadata["addresses_grg"])
    put(grid, "qpoints_frac", metadata["qpoints_frac"])
    put(grid, "grg2bzg", metadata["grg2bzg"])
    put(grid, "bzg2grg", metadata["bzg2grg"])
    put(grid, "inverse_grg", metadata["inverse_grg"])
    put(modes, "frequency", frequencies.astype(np.float32), units="THz", shape_convention="(num_grid, nband)")

    h5.attrs["format"] = "pp_collision_events_v1"
    h5.attrs["created_by"] = Path(__file__).name
    h5.attrs["local_phono3py_einsum"] = LOCAL_PHONO3PY or ""
    h5.attrs["phono3py_triplet_convention"] = "raw triplets satisfy q0 + q1 + q2 = G"
    h5.attrs["stored_index_convention"] = "regular generalized-grid q index, not BZ-surface duplicate index"
    h5.attrs["occupation_shape"] = f"({metadata['num_grg']}, {nband})"
    h5.attrs["kernel_units"] = "THz-like phono3py Gamma units per occupation polynomial"
    h5.attrs["sigma"] = "None" if args.sigma is None else float(args.sigma)
    h5.attrs["sigma_cutoff"] = "None" if args.sigma_cutoff is None else float(args.sigma_cutoff)
    h5.attrs["interaction_lang"] = args.lang
    h5.attrs["symmetrize_fc3q"] = bool(args.symmetrize_fc3q)
    h5.attrs["kernel_threshold"] = float(args.threshold)
    h5.attrs["nband"] = int(nband)
    h5.attrs["num_grid"] = int(metadata["num_grg"])


def _write_csr_metadata(
    path: Path,
    ph3,
    metadata: dict,
    mesh: list[int],
    args: argparse.Namespace,
    nband: int,
    compression: str | None,
) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with h5py.File(tmp_path, "w") as out:
        _write_metadata(out, ph3, metadata, mesh, args, nband)
        out.attrs["format"] = "pp_collision_events_csr_v1"
        out.attrs["source_format"] = "direct_csr"
        out.attrs["created_by"] = Path(__file__).name
        out.attrs["direct_csr"] = True
        out.attrs["csr_compression"] = "none" if compression is None else compression
        out.attrs["h5py_mpi"] = _h5py_mpi_enabled()
    os.replace(tmp_path, path)


def _copy_attrs(src, dst) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def _copy_dataset_chunked(src: h5py.Dataset, dst: h5py.Dataset) -> None:
    if src.shape == ():
        dst[()] = src[()]
        return
    if src.size == 0:
        return
    if src.chunks is None:
        dst[...] = src[...]
        return
    for chunk_sel in src.iter_chunks():
        dst[chunk_sel] = src[chunk_sel]


def _copy_h5_group(
    src: h5py.Group,
    dst: h5py.Group,
    compression: str | None,
) -> None:
    _copy_attrs(src, dst)
    for name, item in src.items():
        if isinstance(item, h5py.Group):
            child = dst.create_group(name)
            _copy_h5_group(item, child, compression)
            continue

        create_kwargs = {}
        if item.shape != ():
            if item.chunks is not None:
                create_kwargs["chunks"] = item.chunks
            if compression is not None:
                create_kwargs["compression"] = compression
        child = dst.create_dataset(name, shape=item.shape, dtype=item.dtype, **create_kwargs)
        _copy_attrs(item, child)
        _copy_dataset_chunked(item, child)


def _serial_repack_h5(
    src_path: Path,
    dst_path: Path,
    compression: str | None,
    parallel_compression: str | None,
) -> None:
    tmp_path = dst_path.with_suffix(dst_path.suffix + ".repack_tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with h5py.File(src_path, "r") as src, h5py.File(tmp_path, "w") as dst:
        _copy_h5_group(src, dst, compression)
        dst.attrs["csr_compression"] = "none" if compression is None else compression
        dst.attrs["parallel_csr_compression"] = (
            "none" if parallel_compression is None else parallel_compression
        )
        dst.attrs["compression_repacked_serial"] = bool(compression != parallel_compression)
    os.replace(tmp_path, dst_path)


def _merge_shards(final_path: Path, shard_paths: list[Path], ph3, metadata, mesh, args, nband) -> None:
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with h5py.File(tmp_path, "w") as out:
        _write_metadata(out, ph3, metadata, mesh, args, nband)
        kernel_dtype = np.dtype(args.dtype)
        for name in PROCESS_NAMES:
            _create_process_datasets(out, name, kernel_dtype)
            total = 0
            for shard in shard_paths:
                with h5py.File(shard, "r") as src:
                    src_name = f"processes/{name}"
                    if src_name not in src:
                        continue
                    count = src[src_name]["kernel"].shape[0]
                    if count == 0:
                        continue
                    grp = out[src_name]
                    old = grp["kernel"].shape[0]
                    new = old + count
                    grp["index"].resize((new, 6))
                    grp["kernel"].resize((new,))
                    grp["index"][old:new] = src[src_name]["index"][:]
                    grp["kernel"][old:new] = src[src_name]["kernel"][:]
                    total += count
            out[f"processes/{name}"].attrs["num_events"] = int(total)
        out.attrs["num_events_total"] = int(
            sum(out[f"processes/{name}/kernel"].shape[0] for name in PROCESS_NAMES)
        )
    os.replace(tmp_path, final_path)


def _generate_collision_events(
    args: argparse.Namespace,
    rank: int,
    ph3,
    metadata: dict,
    nband: int,
    interaction,
    band_indices: np.ndarray,
    emit: Callable[[str, np.ndarray, np.ndarray], int],
    emit_linewidth: Callable[[np.ndarray, np.ndarray], int] | None = None,
    linewidth_interaction=None,
) -> dict[str, int]:
    ise_unit = ImagSelfEnergy(interaction, with_detail=False, lang=args.lang).unit_conversion_factor
    linewidth_ise_unit = None
    if emit_linewidth is not None:
        if linewidth_interaction is None:
            linewidth_interaction = interaction
        linewidth_ise_unit = ImagSelfEnergy(
            linewidth_interaction, with_detail=False, lang=args.lang
        ).unit_conversion_factor
    my_grg = np.arange(rank, metadata["num_grg"], _comm()[2], dtype=np.int64)
    counts = {name: 0 for name in PROCESS_NAMES}
    if emit_linewidth is not None:
        counts[LINEWIDTH_PROCESS_NAME] = 0

    for local_i, grg in enumerate(my_grg, start=1):
        gp_bzg = int(metadata["grg2bzg"][grg])
        interaction.set_grid_point(gp_bzg)
        freqs = interaction.get_phonons()[0]
        f_points = np.asarray(freqs[gp_bzg, band_indices], dtype=np.float64)
        g, g_zero = get_triplets_integration_weights(
            interaction,
            f_points,
            args.sigma,
            sigma_cutoff=args.sigma_cutoff,
            is_collision_matrix=True,
            lang=args.integration_lang or args.lang,
        )

        interaction.run(lang=args.lang, g_zero=g_zero)
        pp = interaction.interaction_strength
        triplets_bzg, weights, _, _ = interaction.get_triplets_at_q()
        triplets_grg = metadata["bzg2grg"][triplets_bzg]
        weights = weights.astype(np.float64, copy=False)

        g_decay = g[0]
        g_diff = g[1]
        g_sum = g[2]
        g_abs_plus = 0.5 * (g_sum - g_decay + g_diff)
        g_abs_minus = 0.5 * (g_sum - g_decay - g_diff)
        weight_shape = (len(weights),) + (1,) * (pp.ndim - 1)
        base = pp * weights.reshape(weight_shape) * float(ise_unit)

        kernels = {
            "abs_plus": base * g_abs_plus,
            "abs_minus": base * g_abs_minus,
            "decay": base * g_decay,
        }
        for name, kernel in kernels.items():
            counts[name] += emit(name, kernel, triplets_grg)

        if emit_linewidth is not None:
            linewidth_interaction.set_grid_point(gp_bzg)
            linewidth_freqs = linewidth_interaction.get_phonons()[0]
            linewidth_band_indices = np.asarray(
                linewidth_interaction.band_indices, dtype=np.int64
            )
            linewidth_f_points = np.asarray(
                linewidth_freqs[gp_bzg, linewidth_band_indices], dtype=np.float64
            )
            gamma_g, gamma_g_zero = get_triplets_integration_weights(
                linewidth_interaction,
                linewidth_f_points,
                args.sigma,
                sigma_cutoff=args.sigma_cutoff,
                is_collision_matrix=False,
                lang=args.integration_lang or args.lang,
            )
            linewidth_interaction.run(lang=args.lang, g_zero=gamma_g_zero)
            linewidth_pp = linewidth_interaction.interaction_strength
            linewidth_triplets_bzg, linewidth_weights, _, _ = (
                linewidth_interaction.get_triplets_at_q()
            )
            linewidth_triplets_grg = metadata["bzg2grg"][linewidth_triplets_bzg]
            linewidth_weights = linewidth_weights.astype(np.float64, copy=False)
            gamma_detail = get_detailed_imag_self_energy_from_g(
                linewidth_interaction,
                args.linewidth_temperature,
                gamma_g,
                gamma_g_zero,
                pp_strength=linewidth_pp,
                unit_conversion_factor=linewidth_ise_unit,
            )
            weighted_gamma_detail = gamma_detail * linewidth_weights.reshape(
                (len(linewidth_weights),) + (1,) * (gamma_detail.ndim - 1)
            )
            counts[LINEWIDTH_PROCESS_NAME] += emit_linewidth(
                weighted_gamma_detail,
                linewidth_triplets_grg,
            )
            linewidth_interaction.delete_interaction_strength()
        interaction.delete_interaction_strength()
        if local_i % args.progress_every == 0 or local_i == len(my_grg):
            print(
                f"[rank {rank}] q {local_i}/{len(my_grg)} grg={int(grg)} "
                + " ".join(f"{k}={v}" for k, v in counts.items()),
                flush=True,
            )
    return counts


def _create_csr_process_group_parallel(
    h5: h5py.File,
    name: str,
    offsets: np.ndarray,
    n_events: int,
    mode_dtype: np.dtype,
    kernel_dtype: np.dtype,
    compression: str | None,
    chunk_size: int,
) -> h5py.Group:
    grp = h5.require_group(f"processes/{name}")
    offsets_ds = grp.create_dataset(
        "offsets",
        shape=offsets.shape,
        chunks=_csr_chunk_shape(len(offsets), chunk_size),
        dtype=np.int64,
        compression=compression,
    )
    with offsets_ds.collective:
        offsets_ds[:] = offsets
    grp.create_dataset(
        "mode1",
        shape=(n_events,),
        chunks=_csr_chunk_shape(n_events, chunk_size),
        dtype=mode_dtype,
        compression=compression,
    )
    grp.create_dataset(
        "mode2",
        shape=(n_events,),
        chunks=_csr_chunk_shape(n_events, chunk_size),
        dtype=mode_dtype,
        compression=compression,
    )
    grp.create_dataset(
        "kernel",
        shape=(n_events,),
        chunks=_csr_chunk_shape(n_events, chunk_size),
        dtype=kernel_dtype,
        compression=compression,
    )
    grp.attrs["description"] = PROCESS_DESCRIPTIONS[name]
    grp.attrs["layout"] = "target_csr"
    grp.attrs["num_events"] = int(n_events)
    return grp


def _write_empty_collective(ds: h5py.Dataset) -> None:
    _write_collective_slice(ds, 0, np.empty(0, dtype=ds.dtype))


def _write_collective_slice(ds: h5py.Dataset, start: int, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=ds.dtype)
    count = len(values)
    file_space = ds.id.get_space()
    if count:
        file_space.select_hyperslab((int(start),), (int(count),))
    else:
        file_space.select_none()
    mem_space = h5py.h5s.create_simple((int(count),))
    dxpl = h5py.h5p.create(h5py.h5p.DATASET_XFER)
    dxpl.set_dxpl_mpio(h5py.h5fd.MPIO_COLLECTIVE)
    ds.id.write(mem_space, file_space, values, dxpl=dxpl)


def _write_spill_to_csr_group(
    comm,
    rank: int,
    spill_grp: h5py.Group,
    out_grp: h5py.Group,
    offsets: np.ndarray,
    prior_counts: np.ndarray,
    mode_dtype: np.dtype,
    kernel_dtype: np.dtype,
    chunk_size: int,
) -> None:
    n_local = int(spill_grp["kernel"].shape[0])
    num_modes = len(offsets) - 1
    if n_local:
        target = spill_grp["target"][:].astype(np.int64, copy=False)
        order = np.argsort(target, kind="stable")
        target_sorted = target[order]
        mode1_sorted = spill_grp["mode1"][:][order].astype(mode_dtype, copy=False)
        mode2_sorted = spill_grp["mode2"][:][order].astype(mode_dtype, copy=False)
        kernel_sorted = spill_grp["kernel"][:][order].astype(kernel_dtype, copy=False)
        unique, group_start, group_counts = np.unique(
            target_sorted, return_index=True, return_counts=True
        )
        target_slices = {
            int(target_id): (int(local_start), int(local_start + count))
            for target_id, local_start, count in zip(unique, group_start, group_counts)
        }
    else:
        mode1_sorted = np.empty(0, dtype=mode_dtype)
        mode2_sorted = np.empty(0, dtype=mode_dtype)
        kernel_sorted = np.empty(0, dtype=kernel_dtype)
        target_slices = {}

    # global_counts[t] > 0 iff any rank has events for target t.
    # All ranks share the same offsets array (built from AllReduce/Exscan), so
    # skipping zero-count targets is safe without extra MPI communication.
    global_counts = np.diff(offsets)

    for target_id in range(num_modes):
        if global_counts[target_id] == 0:
            continue

        local_slice = target_slices.get(target_id)
        if local_slice is None:
            _write_empty_collective(out_grp["mode1"])
            _write_empty_collective(out_grp["mode2"])
            _write_empty_collective(out_grp["kernel"])
            continue

        local_start, local_stop = local_slice
        count = local_stop - local_start
        out_start = int(offsets[target_id] + prior_counts[target_id])
        out_stop = out_start + count
        _write_collective_slice(
            out_grp["mode1"], out_start, mode1_sorted[local_start:local_stop]
        )
        _write_collective_slice(
            out_grp["mode2"], out_start, mode2_sorted[local_start:local_stop]
        )
        _write_collective_slice(
            out_grp["kernel"], out_start, kernel_sorted[local_start:local_stop]
        )


def _write_direct_csr_from_spills(
    out: Path,
    spill_paths: list[Path],
    local_counts: dict[str, np.ndarray],
    ph3,
    metadata: dict,
    mesh: list[int],
    args: argparse.Namespace,
    nband: int,
    comm,
    rank: int,
) -> None:
    requested_compression = None if args.csr_compression == "none" else args.csr_compression
    parallel_filter_ok = _parallel_filter_available(comm, requested_compression)
    if requested_compression is not None and not parallel_filter_ok:
        parallel_compression = None
        parallel_out = out.with_suffix(out.suffix + ".parallel_uncompressed_tmp")
        if rank == 0:
            print(
                f"Parallel HDF5 filtered writes with compression={requested_compression!r} "
                "are unavailable; writing direct CSR uncompressed collectively, then "
                "serially repacking the final HDF5 with the requested compression.",
                flush=True,
            )
            if parallel_out.exists():
                parallel_out.unlink()
    else:
        parallel_compression = requested_compression
        parallel_out = out
    comm.Barrier()

    kernel_dtype = np.dtype(args.dtype)
    num_modes = int(metadata["num_grg"]) * int(nband)
    mode_dtype = _csr_mode_dtype(num_modes)

    if rank == 0:
        _write_csr_metadata(
            parallel_out,
            ph3,
            metadata,
            mesh,
            args,
            nband,
            requested_compression,
        )
    comm.Barrier()

    process_names = tuple(local_counts)
    total_events = 0
    total_linewidth_events = 0
    global_offsets: dict[str, np.ndarray] = {}
    prior_counts_by_process: dict[str, np.ndarray] = {}
    for name in process_names:
        counts = local_counts[name].astype(np.int64, copy=False)
        global_counts = np.empty_like(counts)
        comm.Allreduce(counts, global_counts, op=MPI.SUM)
        prior = np.zeros_like(counts)
        comm.Exscan(counts, prior, op=MPI.SUM)
        if rank == 0:
            prior.fill(0)
        offsets = np.empty(num_modes + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(global_counts, out=offsets[1:])
        global_offsets[name] = offsets
        prior_counts_by_process[name] = prior
        if name in PROCESS_NAMES:
            total_events += int(offsets[-1])
        else:
            total_linewidth_events += int(offsets[-1])

    with h5py.File(parallel_out, "r+", driver="mpio", comm=comm) as h5:
        h5.attrs["num_events_total"] = int(total_events)
        if total_linewidth_events:
            h5.attrs["num_linewidth_events_total"] = int(total_linewidth_events)
            h5.attrs["linewidth_temperature_K"] = float(args.linewidth_temperature)
            h5.attrs["linewidth_uses_mesh_symmetry"] = True
            h5.attrs["linewidth_symmetrize_fc3q"] = bool(args.linewidth_symmetrize_fc3q)
            h5.attrs["linewidth_kernel_units"] = (
                "weighted phono3py gamma_detail THz for W-style linewidth assembly"
            )
        h5.attrs["parallel_csr_compression"] = (
            "none" if parallel_compression is None else parallel_compression
        )
        for name in process_names:
            _create_csr_process_group_parallel(
                h5,
                name,
                global_offsets[name],
                int(global_offsets[name][-1]),
                mode_dtype,
                kernel_dtype,
                parallel_compression,
                args.csr_chunk_size,
        )
        with h5py.File(spill_paths[rank], "r") as spill:
            for name in process_names:
                _write_spill_to_csr_group(
                    comm,
                    rank,
                    spill[f"processes/{name}"],
                    h5[f"processes/{name}"],
                    global_offsets[name],
                    prior_counts_by_process[name],
                    mode_dtype,
                    kernel_dtype,
                    args.csr_chunk_size,
                )
    comm.Barrier()
    if rank == 0 and parallel_out != out:
        _serial_repack_h5(
            parallel_out,
            out,
            requested_compression,
            parallel_compression,
        )
        parallel_out.unlink()
    comm.Barrier()


def build_cache(args: argparse.Namespace) -> None:
    _require_phono3py()
    comm, rank, size = _comm()
    _setup_gpu_for_rank(args.lang, rank)

    mesh = [int(x) for x in args.mesh]
    out = Path(args.output).resolve()
    shard_dir = Path(args.shard_dir or f"{out}.shards").resolve()

    _rank_print(rank, f"Using local phono3py path: {LOCAL_PHONO3PY}")
    _rank_print(rank, f"MPI ranks: {size}")
    _rank_print(rank, f"Output: {out}")
    _rank_print(rank, f"Output format: {args.output_format}")

    ph3 = Phono3py.load(
        args.phono3py_yaml,
        is_mesh_symmetry=False,
        log_level=1 if rank == 0 else 0,
    )
    ph3.mesh_numbers = mesh
    fc_cache = Path(args.fc_cache_dir or f"phonon_cache_{_mesh_tag(mesh)}").resolve()
    _set_forces_and_fc_cache(
        ph3,
        Path(args.fc2_forces),
        Path(args.fc3_forces),
        fc_cache,
        rank,
        comm,
    )
    ph3.init_phph_interaction(symmetrize_fc3q=args.symmetrize_fc3q)

    _rank_print(rank, "Running harmonic phonon solver on full mesh...")
    t0 = time.time()
    ph3.run_phonon_solver()
    _rank_print(rank, f"Harmonic solve done in {time.time() - t0:.1f}s")

    metadata = _build_regular_grid_metadata(ph3.grid)
    nband = ph3.get_phonon_data()[0].shape[1]
    interaction = ph3.phph_interaction
    band_indices = np.asarray(interaction.band_indices, dtype=np.int64)
    if len(band_indices) != nband or not np.array_equal(band_indices, np.arange(nband)):
        raise RuntimeError("This cache builder currently requires all phonon bands.")

    linewidth_interaction = None
    if args.linewidth_temperature is not None:
        _rank_print(rank, "Preparing mesh-symmetry linewidth interaction...")
        linewidth_ph3 = Phono3py.load(
            args.phono3py_yaml,
            is_mesh_symmetry=True,
            log_level=1 if rank == 0 else 0,
        )
        linewidth_ph3.mesh_numbers = mesh
        _set_forces_and_fc_cache(
            linewidth_ph3,
            Path(args.fc2_forces),
            Path(args.fc3_forces),
            fc_cache,
            rank,
            comm,
        )
        linewidth_ph3.init_phph_interaction(
            symmetrize_fc3q=args.linewidth_symmetrize_fc3q
        )
        t0 = time.time()
        linewidth_ph3.run_phonon_solver()
        _rank_print(rank, f"Linewidth harmonic solve done in {time.time() - t0:.1f}s")
        linewidth_interaction = linewidth_ph3.phph_interaction
        linewidth_band_indices = np.asarray(linewidth_interaction.band_indices, dtype=np.int64)
        if len(linewidth_band_indices) != nband or not np.array_equal(
            linewidth_band_indices, np.arange(nband)
        ):
            raise RuntimeError("Linewidth gamma_detail cache currently requires all phonon bands.")

    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"{out.name}.rank{rank:04d}.h5"
    if shard_path.exists():
        shard_path.unlink()

    if args.output_format == "csr":
        num_modes = int(metadata["num_grg"]) * int(nband)
        mode_dtype = _csr_mode_dtype(num_modes)
        process_names = (
            PROCESS_NAMES + (LINEWIDTH_PROCESS_NAME,)
            if args.linewidth_temperature is not None
            else PROCESS_NAMES
        )
        local_target_counts = {
            name: np.zeros(num_modes, dtype=np.int64) for name in process_names
        }
        with h5py.File(shard_path, "w") as spill:
            for name in process_names:
                _create_compact_spill_datasets(
                    spill,
                    name,
                    mode_dtype,
                    np.dtype(args.dtype),
                    args.csr_chunk_size,
                )

            def emit_compact(name: str, kernel: np.ndarray, triplets_grg: np.ndarray) -> int:
                target, mode1, mode2, values = _compact_nonzero_kernel(
                    name,
                    kernel,
                    triplets_grg,
                    metadata["inverse_grg"],
                    band_indices,
                    args.threshold,
                    nband,
                    mode_dtype,
                )
                if len(values):
                    local_target_counts[name] += np.bincount(
                        target.astype(np.int64, copy=False), minlength=num_modes
                    )
                    _append_compact_process_events(
                        spill,
                        name,
                        target,
                        mode1,
                        mode2,
                        values,
                        mode_dtype,
                        np.dtype(args.dtype),
                        args.csr_chunk_size,
                    )
                return len(values)

            def emit_linewidth_compact(kernel: np.ndarray, triplets_grg: np.ndarray) -> int:
                target, mode1, mode2, values = _compact_nonzero_triplet_kernel(
                    kernel,
                    triplets_grg,
                    band_indices,
                    args.threshold,
                    nband,
                    mode_dtype,
                )
                if len(values):
                    local_target_counts[LINEWIDTH_PROCESS_NAME] += np.bincount(
                        target.astype(np.int64, copy=False), minlength=num_modes
                    )
                    _append_compact_process_events(
                        spill,
                        LINEWIDTH_PROCESS_NAME,
                        target,
                        mode1,
                        mode2,
                        values,
                        mode_dtype,
                        np.dtype(args.dtype),
                        args.csr_chunk_size,
                    )
                return len(values)

            _generate_collision_events(
                args,
                rank,
                ph3,
                metadata,
                nband,
                interaction,
                band_indices,
                emit_compact,
                emit_linewidth_compact if args.linewidth_temperature is not None else None,
                linewidth_interaction=linewidth_interaction,
            )

        _barrier(comm)
        spill_paths = [shard_dir / f"{out.name}.rank{r:04d}.h5" for r in range(size)]
        missing = [str(path) for path in spill_paths if not path.exists()]
        if missing:
            raise RuntimeError(f"Missing spill files: {missing}")
        _write_direct_csr_from_spills(
            out,
            spill_paths,
            local_target_counts,
            ph3,
            metadata,
            mesh,
            args,
            nband,
            comm,
            rank,
        )
        if rank == 0:
            print(f"Wrote {out}", flush=True)
            _postprocess_symmetrized_cache(args, out, rank)
            if args.remove_shards:
                shutil.rmtree(shard_dir)
        _barrier(comm)
        return

    counts = {name: 0 for name in PROCESS_NAMES}
    with h5py.File(shard_path, "w") as shard:
        for name in PROCESS_NAMES:
            _create_process_datasets(shard, name, np.dtype(args.dtype))

        def emit_event_list(name: str, kernel: np.ndarray, triplets_grg: np.ndarray) -> int:
            added = _append_nonzero_kernel(
                shard,
                name,
                kernel,
                triplets_grg,
                metadata["inverse_grg"],
                band_indices,
                args.threshold,
                np.dtype(args.dtype),
            )
            counts[name] += added
            return added

        _generate_collision_events(
            args, rank, ph3, metadata, nband, interaction, band_indices, emit_event_list
        )

    _barrier(comm)
    if rank == 0:
        shard_paths = [shard_dir / f"{out.name}.rank{r:04d}.h5" for r in range(size)]
        missing = [str(path) for path in shard_paths if not path.exists()]
        if missing:
            raise RuntimeError(f"Missing shard files: {missing}")
        _merge_shards(out, shard_paths, ph3, metadata, mesh, args, nband)
        print(f"Wrote {out}", flush=True)
        if args.write_csr:
            csr_out = Path(args.csr_output or out.with_name(f"{out.stem}_csr{out.suffix}")).resolve()
            compression = None if args.csr_compression == "none" else args.csr_compression
            print(f"Converting {out} to CSR cache {csr_out}", flush=True)
            convert_cache(
                out,
                csr_out,
                chunk_size=args.csr_chunk_size,
                compression=compression,
                force=args.force_csr,
            )
            print(f"Wrote {csr_out}", flush=True)
            _postprocess_symmetrized_cache(args, csr_out, rank)
        if args.remove_shards:
            shutil.rmtree(shard_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phono3py-yaml", default="../1-MoSe2/MoSe2_relaxed_phono3py_displacements.yaml")
    parser.add_argument("--fc2-forces", default="../1-MoSe2/MoSe2_relaxed_forces_2nd_from_3rd.npy")
    parser.add_argument("--fc3-forces", default="../1-MoSe2/MoSe2_relaxed_forces_3rd.npy")
    parser.add_argument("--mesh", nargs=3, type=int, default=[36, 36, 1])
    parser.add_argument("--output", default="pp_collision_events_m36x36x1.h5")
    parser.add_argument("--fc-cache-dir", default=None)
    parser.add_argument("--shard-dir", default=None)
    parser.add_argument("--lang", default="GPU", choices=["C", "Fast", "GPU", "GPU_phase", "Hybrid"])
    parser.add_argument("--integration-lang", default=None, help="Backend for integration weights; defaults to --lang.")
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--sigma-cutoff", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument(
        "--linewidth-temperature",
        type=float,
        default=None,
        help=(
            "Also store a same-layout processes/gamma_detail CSR group containing "
            "weighted phono3py gamma_detail kernels for this temperature. The "
            "standard abs_plus/abs_minus/decay kernel datasets remain the "
            "arbitrary-occupation dynamics kernels."
        ),
    )
    parser.add_argument("--symmetrize-fc3q", action="store_true")
    # TODO(cleanup, not yet scheduled): this flag is independent of
    # --symmetrize-fc3q above (separate Interaction object for the linewidth
    # branch) and every production collision.slurm passes --symmetrize-fc3q
    # without also passing this one -- looks like an oversight, though
    # confirmed harmless (bit-identical gamma_detail) on this project's data.
    # See dynamics/docs/specs/2026-07-01-linewidth-symmetrize-fc3q-companion-noop.md.
    # Consider defaulting this to track --symmetrize-fc3q, or warning when
    # they diverge and --linewidth-temperature is set.
    parser.add_argument(
        "--linewidth-symmetrize-fc3q",
        action="store_true",
        help=(
            "Use symmetrize_fc3q=True only for the optional gamma_detail "
            "linewidth companion. This matches the W-reference gamma_detail "
            "workflow without changing the standard dynamics kernels. NOTE: "
            "independent of --symmetrize-fc3q above -- passing that flag does "
            "NOT also enable this one (see TODO above this argument)."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--remove-shards", action="store_true")
    parser.add_argument(
        "--output-format",
        choices=["csr", "event-list"],
        default=None,
        help=(
            "Cache format to write. Defaults to csr. If omitted with --write-csr, "
            "uses the legacy event-list-plus-conversion compatibility path."
        ),
    )
    parser.add_argument(
        "--write-csr",
        action="store_true",
        help=(
            "Deprecated compatibility path: write a legacy event-list cache, then convert "
            "a target-CSR companion cache."
        ),
    )
    parser.add_argument("--csr-output", default=None, help="Output path for --write-csr; defaults to *_csr.h5.")
    parser.add_argument("--csr-chunk-size", type=int, default=1_000_000)
    parser.add_argument("--csr-compression", default="gzip", choices=["gzip", "none"])
    parser.add_argument("--force-csr", action="store_true", help="Overwrite an existing CSR output path.")
    parser.add_argument(
        "--symmetrize-full-grid",
        action="store_true",
        help=(
            "After writing a raw CSR cache, run the validated full-grid "
            "symmetrization postprocessor as an opt-in rank-0 stage. The raw "
            "cache is preserved; use --symmetrized-output to choose the output path."
        ),
    )
    parser.add_argument(
        "--symmetry-metadata",
        default=None,
        help="Task-2 symmetry metadata HDF5 sidecar required by --symmetrize-full-grid.",
    )
    parser.add_argument(
        "--degenerate-policy",
        default="block_average",
        choices=["block_average"],
        help="Degenerate-subspace scalar policy for --symmetrize-full-grid.",
    )
    parser.add_argument(
        "--support-policy",
        default=DEFAULT_SUPPORT_POLICY,
        choices=["existing_support", "orbit_complete"],
        help=(
            "Support policy for --symmetrize-full-grid. existing_support is the "
            "validated production default; orbit_complete is experimental."
        ),
    )
    parser.add_argument(
        "--write-raw-before-symmetrization",
        action="store_true",
        help=(
            "Accepted for explicit provenance in scripted workflows. The "
            "integrated path always preserves the raw CSR cache before writing "
            "the separate symmetrized output."
        ),
    )
    parser.add_argument(
        "--symmetrized-output",
        default=None,
        help="Output path for --symmetrize-full-grid; defaults to *_symmetrized.h5.",
    )
    parser.add_argument(
        "--symmetrization-compression",
        default="none",
        choices=["gzip", "none"],
        help="Compression for the symmetrized cache written by the postprocessor.",
    )
    parser.add_argument(
        "--symmetrization-chunk-rows",
        type=int,
        default=4096,
        help="CSR row chunk size used by the symmetrization postprocessor.",
    )
    args = parser.parse_args()
    if args.output_format is None:
        args.output_format = "event-list" if args.write_csr else "csr"
    if args.write_csr and args.output_format == "csr":
        parser.error("--write-csr is only valid with --output-format event-list.")
    if args.linewidth_temperature is not None and args.output_format != "csr":
        parser.error("--linewidth-temperature is currently supported only with --output-format csr.")
    if args.symmetrize_full_grid:
        if not args.symmetry_metadata:
            parser.error("--symmetrize-full-grid requires --symmetry-metadata.")
        if args.output_format == "event-list" and not args.write_csr:
            parser.error("--symmetrize-full-grid requires CSR output.")
        raw_output = (
            Path(args.csr_output or Path(args.output).with_name(f"{Path(args.output).stem}_csr{Path(args.output).suffix}")).resolve()
            if args.output_format == "event-list"
            else Path(args.output).resolve()
        )
        sym_output = (
            Path(args.symmetrized_output).resolve()
            if args.symmetrized_output
            else _default_symmetrized_output(raw_output)
        )
        if sym_output == raw_output:
            parser.error("--symmetrized-output must differ from the raw CSR output.")
    return args


if __name__ == "__main__":
    build_cache(_parse_args())
