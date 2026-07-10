#!/usr/bin/env python3
"""Convert event-list pp-collision caches to a target-CSR layout."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np


PROCESS_NAMES = ("abs_plus", "abs_minus", "decay")
COPY_GROUPS = ("system", "grid", "modes")


def _copy_root_attrs(src: h5py.File, out: h5py.File) -> None:
    for key, value in src.attrs.items():
        out.attrs[key] = value
    out.attrs["format"] = "pp_collision_events_csr_v1"
    out.attrs["source_format"] = src.attrs.get("format", "pp_collision_events_v1")
    out.attrs["created_by"] = Path(__file__).name


def _copy_metadata_groups(src: h5py.File, out: h5py.File) -> None:
    for name in COPY_GROUPS:
        if name in src:
            src.copy(name, out)


def _mode_dtype(num_modes: int) -> np.dtype:
    if num_modes <= np.iinfo(np.uint32).max:
        return np.dtype("uint32")
    return np.dtype("uint64")


def _chunk_shape(n_events: int, chunk_size: int) -> tuple[int,]:
    return (max(1, min(int(chunk_size), max(1, int(n_events)))),)


def _source_mode_arrays(index: np.ndarray, nband: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = index[:, 0] * nband + index[:, 1]
    mode1 = index[:, 2] * nband + index[:, 3]
    mode2 = index[:, 4] * nband + index[:, 5]
    return target, mode1, mode2


def _build_counts(src_grp: h5py.Group, num_modes: int, nband: int, chunk_size: int) -> np.ndarray:
    counts = np.zeros(num_modes, dtype=np.int64)
    n_events = src_grp["kernel"].shape[0]
    for start in range(0, n_events, chunk_size):
        stop = min(start + chunk_size, n_events)
        target, _, _ = _source_mode_arrays(src_grp["index"][start:stop], nband)
        counts += np.bincount(target, minlength=num_modes)
    return counts


def _create_process_group(
    out: h5py.File,
    name: str,
    src_grp: h5py.Group,
    offsets: np.ndarray,
    mode_dtype: np.dtype,
    compression: str | None,
    chunk_size: int,
) -> h5py.Group:
    n_events = int(src_grp["kernel"].shape[0])
    grp = out.require_group(f"processes/{name}")
    grp.create_dataset("offsets", data=offsets, dtype=np.int64, compression=compression)
    grp.create_dataset(
        "mode1",
        shape=(n_events,),
        chunks=_chunk_shape(n_events, chunk_size),
        dtype=mode_dtype,
        compression=compression,
    )
    grp.create_dataset(
        "mode2",
        shape=(n_events,),
        chunks=_chunk_shape(n_events, chunk_size),
        dtype=mode_dtype,
        compression=compression,
    )
    grp.create_dataset(
        "kernel",
        shape=(n_events,),
        chunks=_chunk_shape(n_events, chunk_size),
        dtype=src_grp["kernel"].dtype,
        compression=compression,
    )
    for key, value in src_grp.attrs.items():
        grp.attrs[key] = value
    grp.attrs["layout"] = "target_csr"
    grp.attrs["num_events"] = n_events
    return grp


def _fill_process_group(
    src_grp: h5py.Group,
    out_grp: h5py.Group,
    offsets: np.ndarray,
    nband: int,
    mode_dtype: np.dtype,
    chunk_size: int,
) -> None:
    cursor = offsets[:-1].copy()
    n_events = src_grp["kernel"].shape[0]
    for start in range(0, n_events, chunk_size):
        stop = min(start + chunk_size, n_events)
        target, mode1, mode2 = _source_mode_arrays(src_grp["index"][start:stop], nband)
        if len(target) == 0:
            continue

        order = np.argsort(target, kind="stable")
        target_sorted = target[order]
        mode1_sorted = mode1[order].astype(mode_dtype, copy=False)
        mode2_sorted = mode2[order].astype(mode_dtype, copy=False)
        kernel_sorted = src_grp["kernel"][start:stop][order]

        unique, group_start, group_counts = np.unique(
            target_sorted, return_index=True, return_counts=True
        )
        positions = np.empty(len(target_sorted), dtype=np.int64)
        for target_id, local_start, count in zip(unique, group_start, group_counts):
            base = cursor[target_id]
            positions[local_start : local_start + count] = base + np.arange(count, dtype=np.int64)
            cursor[target_id] = base + count

        out_grp["mode1"][positions] = mode1_sorted
        out_grp["mode2"][positions] = mode2_sorted
        out_grp["kernel"][positions] = kernel_sorted


def convert_cache(
    source_cache: Path,
    output_cache: Path,
    chunk_size: int = 1_000_000,
    compression: str | None = "gzip",
    force: bool = False,
) -> None:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    if output_cache.exists() and not force:
        raise FileExistsError(f"{output_cache} exists; pass --force to overwrite.")

    tmp_path = output_cache.with_suffix(output_cache.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    with h5py.File(source_cache, "r") as src, h5py.File(tmp_path, "w") as out:
        source_format = src.attrs.get("format", "pp_collision_events_v1")
        if isinstance(source_format, bytes):
            source_format = source_format.decode()
        if source_format != "pp_collision_events_v1":
            raise ValueError(f"Expected pp_collision_events_v1 source, found {source_format!r}.")

        num_grid = int(src.attrs["num_grid"])
        nband = int(src.attrs["nband"])
        num_modes = num_grid * nband
        mode_dtype = _mode_dtype(num_modes)

        _copy_root_attrs(src, out)
        _copy_metadata_groups(src, out)

        total_events = 0
        for process in PROCESS_NAMES:
            group_name = f"processes/{process}"
            if group_name not in src:
                continue
            src_grp = src[group_name]
            counts = _build_counts(src_grp, num_modes, nband, chunk_size)
            offsets = np.empty(num_modes + 1, dtype=np.int64)
            offsets[0] = 0
            np.cumsum(counts, out=offsets[1:])
            out_grp = _create_process_group(
                out, process, src_grp, offsets, mode_dtype, compression, chunk_size
            )
            _fill_process_group(src_grp, out_grp, offsets, nband, mode_dtype, chunk_size)
            total_events += int(offsets[-1])

        out.attrs["num_events_total"] = total_events

    os.replace(tmp_path, output_cache)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_cache", type=Path)
    parser.add_argument("output_cache", type=Path)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--compression", default="gzip", choices=["gzip", "none"])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    compression = None if args.compression == "none" else args.compression
    convert_cache(
        args.source_cache,
        args.output_cache,
        chunk_size=args.chunk_size,
        compression=compression,
        force=args.force,
    )
    print(f"Wrote {args.output_cache}")


if __name__ == "__main__":
    main()
