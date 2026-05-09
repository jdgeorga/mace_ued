"""Calculator construction for MACE foundation-model workflows."""

from __future__ import annotations

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
        return MACECalculator(
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


def config_from_args(args) -> MACECalculatorConfig:
    """Create calculator config from parsed argparse arguments."""

    return MACECalculatorConfig(
        model=args.mace_model,
        model_path=args.mace_model_path,
        device=args.device,
        default_dtype=args.default_dtype,
        dispersion=bool(args.dispersion),
    )
