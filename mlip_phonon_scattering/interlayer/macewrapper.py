from ase.calculators.calculator import (
    Calculator,
    CalculatorError,
    CalculatorSetupError,
    all_changes,
)
from ase.data import atomic_masses, atomic_numbers, chemical_symbols
from typing import List, Dict, Union
from pathlib import Path
from ase.atoms import Atoms
import numpy as np
import ctypes


def _preload_ijit_notify() -> None:
    """
    PyTorch builds that include oneDNN JIT profiling may reference iJIT_* symbols
    without linking an ittnotify library. Preload a provider (or a local stub)
    so importing torch doesn't fail with an undefined symbol.
    """
    for lib in ("libiJITProfiling.so", "libittnotify.so", "libjitprofiling.so"):
        try:
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            pass

    stub_dir = Path(__file__).resolve().parent / "ittnotify_stub"
    for p in (
        stub_dir / "libittnotify.so",
        stub_dir / "libittnotify_stub.so",
        stub_dir / "libiJITProfiling.so",
    ):
        if p.exists():
            ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
            return


_preload_ijit_notify()
import torch
from mace.calculators import MACECalculator
from mace.calculators.mace import full_3x3_to_voigt_6_stress


class CompatMACECalculator(MACECalculator):
    def _call_model(self, model, batch_dict, compute_stress, oeq_compile):
        model_kwargs = {
            "compute_stress": compute_stress,
            "training": self.use_compile and not oeq_compile,
            "compute_edge_forces": self.compute_atomic_stresses,
            "compute_atomic_stresses": self.compute_atomic_stresses,
        }
        try:
            return model(batch_dict, **model_kwargs)
        except RuntimeError as exc:
            if "Unknown keyword argument" not in str(exc):
                raise
            # Older TorchScript MACE models do not accept the newer atomic-stress
            # kwargs used by mace-torch 0.3.16. Forces only need this older call.
            model_kwargs.pop("compute_edge_forces", None)
            model_kwargs.pop("compute_atomic_stresses", None)
            return model(batch_dict, **model_kwargs)

    # pylint: disable=dangerous-default-value
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms)

        batch_base = self._atoms_to_batch(atoms)
        num_real_atoms = len(atoms)
        is_padded = self.pad_num_atoms > 0 or self.pad_num_edges > 0
        stress_properties = {"stress", "stresses", "virials"}
        compute_stress = (
            self.model_type in ["MACE", "EnergyDipoleMACE", "PolarMACE"]
            and properties is not None
            and any(prop in stress_properties for prop in properties)
        )
        oeq_compile = self.use_compile and self._enable_oeq

        ret_tensors = None
        node_e0 = None
        for i, model in enumerate(self.models):
            batch = self._clone_batch(batch_base)
            model_dtype = next(model.parameters()).dtype
            for key in batch.keys:
                value = batch[key]
                if torch.is_tensor(value) and torch.is_floating_point(value):
                    batch[key] = value.to(dtype=model_dtype)
            batch_dict = batch.to_dict()

            if oeq_compile and compute_stress:
                positions = batch_dict["positions"]
                num_graphs = int(batch_dict["ptr"].numel() - 1)
                displacement = torch.zeros(
                    (num_graphs, 3, 3),
                    dtype=positions.dtype,
                    device=positions.device,
                )
                displacement = displacement + positions.sum() * 0.0
                batch_dict["displacement"] = displacement

            out = self._call_model(model, batch_dict, compute_stress, oeq_compile)
            if is_padded:
                out = self._slice_real_outputs(out, num_real_atoms)
            if i == 0:
                ret_tensors, node_e0 = self._create_result_tensors(
                    self.num_models, num_real_atoms, batch, out
                )
            for key, val in ret_tensors.items():
                if out.get(key) is not None:
                    val[i] = out[key].detach()

        self.results = {}
        scalar_tensors = set(["energy"])
        results_store_ensemble = set(["energy", "forces", "stress", "dipole"])
        results_map = [
            ("energy", "energy", self.energy_units_to_eV),
            ("node_energy", "node_energy", self.energy_units_to_eV),
            ("forces", "forces", self.energy_units_to_eV / self.length_units_to_A),
            ("stress", "stress", self.energy_units_to_eV / self.length_units_to_A**3),
            (
                "stresses",
                "atomic_stresses",
                self.energy_units_to_eV / self.length_units_to_A**3,
            ),
            (
                "virials",
                "atomic_virials",
                self.energy_units_to_eV / self.length_units_to_A**3,
            ),
            ("dipole", "dipole", 1.0),
            ("charges", "charges", 1.0),
            ("polarizability", "polarizability", 1.0),
            ("polarizability_sh", "polarizability_sh", 1.0),
        ]
        for results_key, ret_key, unit_conv in results_map:
            if ret_tensors.get(ret_key) is not None:
                data = torch.mean(ret_tensors[ret_key], dim=0).cpu()
                if ret_key in scalar_tensors:
                    data = data.item()
                else:
                    data = data.numpy()
                self.results[results_key] = data * unit_conv

                if self.num_models > 1 and results_key in results_store_ensemble:
                    data = ret_tensors[results_key].cpu().numpy()
                    data *= unit_conv
                    self.results[results_key + "_comm"] = data

                    data = torch.var(
                        ret_tensors[results_key], dim=0, unbiased=False
                    ).cpu()
                    if ret_key in scalar_tensors:
                        data = data.item()
                    else:
                        data = data.numpy()
                    data *= unit_conv
                    self.results[results_key + "_var"] = data

        if self.results.get("energy") is not None:
            self.results["free_energy"] = self.results["energy"]
        if self.results.get("node_energy") is not None:
            self.results["energies"] = self.results["node_energy"].copy()
            self.results["node_energy"] -= node_e0
        if self.results.get("stress") is not None:
            self.results["stress"] = full_3x3_to_voigt_6_stress(self.results["stress"])
        if self.results.get("stresses") is not None:
            self.results["stresses"] = np.asarray(
                [
                    full_3x3_to_voigt_6_stress(stress)
                    for stress in self.results["stresses"]
                ]
            )


class InterlayerCompatMACECalculator(CompatMACECalculator):
    """MACE calculator variant that keeps only cross-layer graph edges."""

    def __init__(self, *args, **kwargs):
        arrays_keys = dict(kwargs.pop("arrays_keys", {}) or {})
        arrays_keys.setdefault("layer_ids", "layer_ids")
        super().__init__(*args, arrays_keys=arrays_keys, **kwargs)

    def _atoms_to_batch(self, atoms):
        batch = super()._atoms_to_batch(atoms)
        layer_ids = batch["layer_ids"].view(-1)
        edge_index = batch["edge_index"]
        edge_mask = layer_ids[edge_index[0]] != layer_ids[edge_index[1]]
        num_edges = edge_index.shape[1]

        batch["edge_index"] = edge_index[:, edge_mask]
        for key in list(batch.keys):
            value = batch[key]
            if (
                key != "edge_index"
                and torch.is_tensor(value)
                and value.ndim > 0
                and value.shape[0] == num_edges
            ):
                batch[key] = value[edge_mask]

        return batch


class MACEWCalculator(Calculator):
    # Define the properties that the calculator can handle
    implemented_properties = ["energy", "energies", "forces", "free_energy"]

    def __init__(self,
                 atoms: Atoms,
                 layer_symbols: list[str],
                 model_file: str,
                 device='cpu',
                 default_dtype='float32',
                 is_interlayer_calc=False,
                 **kwargs):
        """
        Initializes the MACEWCalculator with a given set of atoms, layer symbols, model file, and device.

        :param atoms: ASE atoms object.
        :param layer_symbols: List of symbols representing different layers in the structure.
        :param model_file: Path to the file containing the trained model.
        :param device: Device to run the calculations on, default is 'cpu'.
        :param default_dtype: Default data type for calculations ('float32' or 'float64').
        :param kwargs: Additional keyword arguments for the base class.
        """
        self.atoms = atoms  # ASE atoms object
        self.atom_types = atoms.arrays['atom_types']  # Extract atom types from atoms object
        self.device = device  # Device for computations
        self.layer_ids = atoms.arrays['layer_ids']
        self.is_interlayer_calc = is_interlayer_calc
        # Flatten the layer symbols list
        self.layer_symbols = [symbol for sublist in layer_symbols for symbol in (sublist if isinstance(sublist, list) else [sublist])]

        # Determine unique atom types and their indices
        unique_types, inverse = np.unique(self.atom_types, return_inverse=True)

        # Map atom types to their relative positions in the unique_types array
        self.relative_layer_types = inverse

        # Ensure the number of unique atom types matches the number of layer symbols provided
        if len(unique_types) != len(self.layer_symbols):
            raise ValueError("Mismatch between the number of atom types and provided layer symbols.")

        # Initialize the MACE calculator
        mace_calculator_cls = InterlayerCompatMACECalculator if is_interlayer_calc else CompatMACECalculator
        self.mace_calc = mace_calculator_cls(
            model_paths=model_file,
            device=device,
            default_dtype=default_dtype,
            **kwargs
        )

        # Initialize the base Calculator class with any additional keyword arguments
        Calculator.__init__(self, **kwargs)

    def calculate(self,
                 atoms: Atoms = None,
                 properties=None,
                 system_changes=all_changes):
        """
        Performs the calculation for the given atoms and properties.

        :param atoms: ASE atoms object to calculate properties for.
        :param properties: List of properties to calculate. If None, uses implemented_properties.
        :param system_changes: List of changes that have been made to the system since last calculation.
        """
        # Default to implemented properties if none are specified
        if properties is None:
            properties = self.implemented_properties

        # Create a temporary copy of the atoms object
        tmp_atoms = atoms.copy()[:]
        tmp_atoms.calc = None  # Remove any attached calculator

        # Backup original atomic numbers and set new atomic numbers based on relative layer types
        original_atom_numbers = tmp_atoms.numbers.copy()
        # tmp_atoms.set_atomic_numbers(self.relative_layer_types + 1)
        tmp_atoms.arrays['atom_types'] = self.relative_layer_types
        tmp_atoms.arrays['layer_ids'] = self.layer_ids

        # Set the MACE calculator for the temporary atoms
        tmp_atoms.calc = self.mace_calc

        # Calculate properties using MACE
        self.mace_calc.calculate(tmp_atoms, properties, system_changes)

        # Restore the original atomic numbers and types
        tmp_atoms.set_atomic_numbers(original_atom_numbers)
        tmp_atoms.arrays['atom_types'] = self.atom_types

        # Copy results from MACE calculator
        self.results = self.mace_calc.results.copy() 
