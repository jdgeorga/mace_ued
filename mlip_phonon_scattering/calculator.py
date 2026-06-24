"""Calculator construction for MACE foundation-model workflows."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MACECalculatorConfig:
    """Configuration for a MACE ASE calculator."""

    model: str = "medium"
    model_path: str | None = None
    device: str = "cpu"
    default_dtype: str = "float32"
    dispersion: bool = False
    interlayer: bool = False
    intralayer_models: tuple[str, ...] = ()
    interlayer_models: tuple[str, ...] = ()
    layer_symbols: tuple[tuple[str, ...], ...] | None = None


def build_mace_foundation_calculator(config: MACECalculatorConfig):
    """Build a MACE ASE calculator.

    If ``model_path`` is supplied, the local model is loaded through
    ``MACECalculator``. Otherwise this uses MACE's foundation-model helper
    ``mace_mp`` with the configured model size/name.
    """

    try:
        from mace.calculators import MACECalculator, mace_mp
    except ImportError as exc:
        raise ImportError(
            "Could not import MACE. Activate the phonon environment or install "
            "the `mace-torch` package before running this workflow."
        ) from exc

    if config.model_path:
        model_path = Path(config.model_path).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"MACE model file does not exist: {model_path}")
        # The trained TMD models are older compiled TorchScript modules whose
        # ``forward`` does not accept the newer kwargs (``compute_edge_forces`` etc.)
        # that mace-torch 0.3.16's stock ``MACECalculator`` passes. Prefer the
        # compatibility shim, which retries without those kwargs; fall back to the
        # stock calculator if the helper is unavailable.
        try:
            from macewrapper import CompatMACECalculator as _LocalCalculator
        except ImportError:
            _LocalCalculator = MACECalculator
        return _LocalCalculator(
            model_paths=str(model_path),
            device=config.device,
            default_dtype=config.default_dtype,
        )

    kwargs: dict[str, Any] = {
        "model": config.model,
        "device": config.device,
        "default_dtype": config.default_dtype,
    }
    if config.dispersion:
        kwargs["dispersion"] = True
    return mace_mp(**kwargs)


def build_nlayer_calculator(config: MACECalculatorConfig, atoms):
    """Build a stacked ``NLayerCalculator`` from per-layer and interlayer models.

    The calculator is bound to ``atoms``: per-layer and interlayer ``MACEWCalculator``
    instances are constructed from subsets of ``atoms`` selected by contiguous
    ``atom_types`` value ranges, so ``atoms`` must share the topology/size of every
    structure the returned calculator will evaluate.
    """

    try:
        from macewrapper import MACEWCalculator
        from n_layer import NLayerCalculator
    except ImportError as exc:
        raise ImportError(
            "Could not import the interlayer helpers (macewrapper / n_layer). Ensure "
            "`repos/mace-interlayer/examples/interlayer_helpers` is on PYTHONPATH "
            "(source load_mace_phonon_env.sh)."
        ) from exc

    import numpy as np

    if config.layer_symbols is None:
        raise ValueError("Interlayer mode requires --layer-symbols to be set.")
    if len(config.intralayer_models) != len(config.layer_symbols):
        raise ValueError(
            "Number of intralayer models "
            f"({len(config.intralayer_models)}) must match number of layers "
            f"({len(config.layer_symbols)})."
        )
    if len(config.interlayer_models) != len(config.layer_symbols) - 1:
        raise ValueError(
            "Number of interlayer models "
            f"({len(config.interlayer_models)}) must be one fewer than the number "
            f"of layers ({len(config.layer_symbols)})."
        )

    layer_symbols = [list(layer) for layer in config.layer_symbols]

    if "atom_types" not in atoms.arrays:
        raise KeyError(
            "Bilayer/interlayer inputs require `atom_types` (and `layer_ids`) arrays "
            "on the atoms, but `atom_types` is missing."
        )
    atom_types = atoms.arrays["atom_types"]

    intralayer_calcs = []
    lower = 0
    for i, syms in enumerate(layer_symbols):
        upper = lower + len(syms) - 1
        mask = np.logical_and(atom_types >= lower, atom_types <= upper)
        layer_atoms = atoms[mask]
        intralayer_calcs.append(
            MACEWCalculator(
                layer_atoms,
                syms,
                model_file=str(config.intralayer_models[i]),
                device=config.device,
                default_dtype=config.default_dtype,
                is_interlayer_calc=False,
            )
        )
        lower = upper + 1

    interlayer_calcs = []
    for i in range(len(layer_symbols) - 1):
        n_i = len(layer_symbols[i])
        n_j = len(layer_symbols[i + 1])
        start_i = sum(len(s) for s in layer_symbols[:i])
        end_j = start_i + n_i + n_j - 1
        mask = np.logical_and(atom_types >= start_i, atom_types <= end_j)
        pair_atoms = atoms[mask]
        interlayer_calcs.append(
            MACEWCalculator(
                pair_atoms,
                [layer_symbols[i], layer_symbols[i + 1]],
                model_file=str(config.interlayer_models[i]),
                device=config.device,
                default_dtype=config.default_dtype,
                is_interlayer_calc=True,
            )
        )

    return NLayerCalculator([atoms], intralayer_calcs, interlayer_calcs, layer_symbols)


def build_calculator(config: MACECalculatorConfig, atoms=None):
    """Dispatch to the interlayer or single foundation-model calculator builder."""

    if config.interlayer:
        if atoms is None:
            raise ValueError(
                "Interlayer mode requires an `atoms` structure to build the "
                "stacked calculator."
            )
        return build_nlayer_calculator(config, atoms)
    return build_mace_foundation_calculator(config)


def add_mace_arguments(parser) -> None:
    """Add common MACE calculator arguments to an argparse parser."""

    parser.add_argument(
        "--mace-model",
        default="medium",
        help="MACE foundation model name/size passed to mace_mp when --mace-model-path is not set.",
    )
    parser.add_argument(
        "--mace-model-path",
        default=None,
        help="Optional local MACE model file. Overrides --mace-model.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device for MACE, e.g. cpu or cuda.")
    parser.add_argument(
        "--default-dtype",
        default="float32",
        choices=("float32", "float64"),
        help="Default floating point dtype for MACE.",
    )
    parser.add_argument(
        "--dispersion",
        action="store_true",
        help="Enable MACE-MP dispersion support when available.",
    )


def add_interlayer_arguments(parser) -> None:
    """Add interlayer multi-model calculator arguments to an argparse parser."""

    parser.add_argument(
        "--interlayer",
        action="store_true",
        help="Use a stacked NLayerCalculator with separate intra/interlayer models.",
    )
    parser.add_argument(
        "--intralayer-models",
        nargs="+",
        default=[],
        help="One intralayer MACE model path per layer.",
    )
    parser.add_argument(
        "--interlayer-model",
        action="append",
        dest="interlayer_models",
        default=[],
        help="One interlayer MACE model path per adjacent layer pair (repeatable).",
    )
    parser.add_argument(
        "--layer-symbols",
        type=str,
        default=None,
        help="Per-layer chemical symbols, e.g. \"[['Mo','Se','Se'],['W','Se','Se']]\".",
    )


def config_from_args(args) -> MACECalculatorConfig:
    """Create calculator config from parsed argparse arguments."""

    if getattr(args, "interlayer", False):
        parsed_symbols = ast.literal_eval(args.layer_symbols)
        layer_symbols = tuple(tuple(str(s) for s in layer) for layer in parsed_symbols)
        return MACECalculatorConfig(
            model=args.mace_model,
            model_path=args.mace_model_path,
            device=args.device,
            default_dtype=args.default_dtype,
            dispersion=bool(args.dispersion),
            interlayer=True,
            intralayer_models=tuple(args.intralayer_models),
            interlayer_models=tuple(args.interlayer_models),
            layer_symbols=layer_symbols,
        )

    return MACECalculatorConfig(
        model=args.mace_model,
        model_path=args.mace_model_path,
        device=args.device,
        default_dtype=args.default_dtype,
        dispersion=bool(args.dispersion),
    )
