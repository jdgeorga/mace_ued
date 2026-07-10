"""Build or load the cached force constants used by the linewidth workflow."""

from __future__ import annotations

import argparse

import phono3py

from .gpu_scattering_W_phonons_bilayer_comm_mesh import set_phono3py_forces_with_mesh_fc_cache


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for force-constant cache construction."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phono3py-yaml", required=True, help="Phono3py displacement YAML file.")
    parser.add_argument("--fc2-forces", required=True, help="Second-order-from-third force array (.npy).")
    parser.add_argument("--fc3-forces", required=True, help="Third-order force array (.npy).")
    parser.add_argument("--cache-dir", required=True, help="Mesh-tagged force-constant cache directory.")
    parser.add_argument("--mesh", nargs=3, type=int, default=None, help="Optional mesh dimensions for the cache.")
    parser.add_argument("--populate-mesh-cache", action="store_true", help="Accepted for workflow compatibility.")
    return parser.parse_args()


def main() -> None:
    """Attach force arrays and build/load the mesh force-constant cache."""

    args = parse_args()
    ph3 = phono3py.load(args.phono3py_yaml, log_level=1)
    set_phono3py_forces_with_mesh_fc_cache(
        ph3,
        fc2_forces=args.fc2_forces,
        fc3_forces=args.fc3_forces,
        cache_dir_mesh=args.cache_dir,
        legacy_cache_dir="phonon_cache",
        populate_mesh_cache=True,
    )


if __name__ == "__main__":
    main()
