# from ase.calculators.calculator import (
#     Calculator,
#     CalculatorError,
#     CalculatorSetupError,
#     all_changes,
# )
# from ase import Atoms
# import numpy as np


# class NLayerCalculator(Calculator):
#     implemented_properties = ['energy', 'energies', 'forces']  # Define the properties this calculator can compute
    
#     def __init__(self,
#                  atoms,
#                  intralayer_calculators: list[Calculator],
#                  interlayer_calculators: list[Calculator],
#                  layer_symbols: list[str],
#                  **kwargs):
#         Calculator.__init__(self, **kwargs)
#         self.intra_calc_list = intralayer_calculators
#         self.inter_calc_list = interlayer_calculators
#         self.layer_symbols = layer_symbols
        

#     def calculate(self,
#                   atoms: Atoms,
#                   properties=None,
#                   system_changes=all_changes):
        
#         if properties is None:
#             properties = self.implemented_properties

#         self.atoms_layer_list = []

#         Calculator.calculate(self, atoms, properties, system_changes)

#         self.get_atoms_layer_list(self.atoms)

#         self.results['layer_energy'] = np.zeros((len(self.intra_calc_list), len(self.intra_calc_list)))
#         self.results['layer_forces'] = np.zeros((len(self.intra_calc_list),
#                                                  len(self.intra_calc_list),
#                                                  atoms.get_global_number_of_atoms(), 3))
#         # self.results['layer_energies'] = np.zeros((len(self.intra_calc_list),
#         #                                            len(self.intra_calc_list),
#         #                                            atoms.get_global_number_of_atoms()))
#         for layer_index_i in range(len(self.intra_calc_list)):
#             for layer_index_j in range(len(self.intra_calc_list)):    
#                 if layer_index_i == layer_index_j:
#                     self.calculate_intralayer(atoms, layer=layer_index_i)
#                 elif layer_index_i < layer_index_j:
#                     self.calculate_interlayer(atoms,
#                                               layer_1=layer_index_i,
#                                               layer_2=layer_index_j)
#                 elif layer_index_i > layer_index_j:
#                     self.results['layer_energy'][layer_index_i, layer_index_j] = 0
#                     self.results['layer_forces'][layer_index_i, layer_index_j] = 0
#                     # self.results['layer_energies'][layer_index_i, layer_index_j] = self.results['layer_energies'][layer_index_j, layer_index_i]
#         self.results["energy"] = self.results["layer_energy"].sum()
#         self.results["forces"] = self.results["layer_forces"].sum(axis=(0,1))
#         # self.results['energies'] = 1./2 * (self.results["layer_energies"].sum(axis=(0,1)) + self.results["layer_energies"].trace(axis1 = 0, axis2 = 1))
     
#     def calculate_intralayer(self, atoms, layer: int):

#         if layer <= len(self.intra_calc_list):
#             calc = self.intra_calc_list[layer]
#             atoms_L = self.atoms_layer_list[layer]
#         else:
#             raise ValueError("layer number muls")
        
#         atoms_L.calc = calc
#         atoms_L.calc.calculate(atoms_L)

#         lower_layers_num_atoms = sum([len(layer_atoms) for layer_atoms in self.atoms_layer_list[:(layer)]])
#         layer_atom_indices = [lower_layers_num_atoms,
#                               lower_layers_num_atoms + len(self.atoms_layer_list[layer])]
        
#         self.results[f"layer_energy"][layer,layer] = atoms_L.calc.results['energy']
#         self.results[f"layer_forces"][layer,layer][layer_atom_indices[0]:layer_atom_indices[1]] = atoms_L.calc.results['forces']
#         # self.results['layer_energies'][layer,layer][layer_atom_indices[0]:layer_atom_indices[1]] = atoms_L.calc.results['energies']

#     def calculate_interlayer(self, atoms, layer_1: int, layer_2: int):
#         # JDG: code only considers adjacent layers
#         if layer_2 > layer_1 + 1:
#             return 0
        
#         atoms_L = self.atoms_layer_list[layer_1].copy() + self.atoms_layer_list[layer_2].copy()
#         calc = self.inter_calc_list[layer_1]
#         atoms_L.calc = calc
#         atoms_L.calc.calculate(atoms_L)

#         lower_layers_num_atoms = sum([len(layer_atoms) for layer_atoms in self.atoms_layer_list[:(layer_1)]])
#         layer_atom_indices = [lower_layers_num_atoms,
#                               lower_layers_num_atoms + len(self.atoms_layer_list[layer_1]) + len(self.atoms_layer_list[layer_2])]
        
#         self.results["layer_energy"][layer_1,layer_2] = atoms_L.calc.results['energy']
#         self.results["layer_forces"][layer_1,layer_2][layer_atom_indices[0]:layer_atom_indices[1]] = atoms_L.calc.results['forces']
#         # self.results['layer_energies'][layer_1,layer_2][layer_atom_indices[0]:layer_atom_indices[1]] = atoms_L.get_potential_energies()

#     def get_atoms_layer_list(self, atoms):

#         for layer in range(len(self.intra_calc_list)):
        
#             lower_layer_num_atoms = sum([len(layer_atoms) for layer_atoms in self.layer_symbols[:(layer)]])
#             layer_atom_indices = [lower_layer_num_atoms,
#                                   lower_layer_num_atoms + len(self.layer_symbols[layer]) - 1]
            
#             atoms_L = atoms.copy()[np.logical_and(atoms.arrays["atom_types"] <= layer_atom_indices[1],
#                                                   atoms.arrays["atom_types"] >= layer_atom_indices[0])]
            
#             self.atoms_layer_list.append(atoms_L)


from ase.calculators.calculator import Calculator, all_changes
from ase import Atoms
import numpy as np

class NLayerCalculator(Calculator):
    """
    A calculator designed to handle calculations for materials with multiple layers, 
    using separate intra-layer and inter-layer calculators.
    """
    # Properties that the calculator can compute
    implemented_properties = ['energy', 'energies', 'forces']
    
    def __init__(self, atoms, intralayer_calculators: list[Calculator], interlayer_calculators: list[Calculator], layer_symbols: list[str], **kwargs):
        """
        Initializes the NLayerCalculator.

        :param atoms: The ASE Atoms object representing the system.
        :param intralayer_calculators: A list of calculators for intra-layer interactions.
        :param interlayer_calculators: A list of calculators for inter-layer interactions.
        :param layer_symbols: A list of symbols (or any identifier) for each layer in the system.
        """
        super().__init__(**kwargs)  # Initialize the parent Calculator class
        self.intra_calc_list = intralayer_calculators  # List of intra-layer calculators
        self.inter_calc_list = interlayer_calculators  # List of inter-layer calculators
        self.layer_symbols = layer_symbols  # Symbols for different layers

    def calculate(self, atoms: Atoms, properties=None, system_changes=all_changes):
        """
        Performs the calculation for the specified atoms.

        :param atoms: The ASE Atoms object to calculate properties for.
        :param properties: The list of properties to calculate. Uses implemented_properties by default.
        :param system_changes: The list of changes that have been made to the system since the last calculation.
        """
        # Set to default properties if none are specified
        if properties is None:
            properties = self.implemented_properties

        # Initialize or reset the layer list for atoms
        self.atoms_layer_list = []

        # Perform the base class calculation (primarily for handling system changes)
        super().calculate(atoms, properties, system_changes)

        # Populate the atoms_layer_list based on the provided atoms and layer_symbols
        self.get_atoms_layer_list(atoms)

        # Initialize result arrays for energies and forces
        num_layers = len(self.intra_calc_list)
        num_atoms = atoms.get_global_number_of_atoms()
        self.results['layer_energy'] = np.zeros((num_layers, num_layers))
        self.results['layer_forces'] = np.zeros((num_layers, num_layers, num_atoms, 3))

        # Calculate intra-layer and inter-layer interactions
        for i in range(num_layers):
            for j in range(num_layers):    
                if i == j:
                    # Intra-layer calculation
                    self.calculate_intralayer(atoms, layer=i)
                elif i < j:
                    # Inter-layer calculation for layers i and j
                    self.calculate_interlayer(atoms, layer_1=i, layer_2=j)
                else:
                    # No calculation needed, ensure energy and forces are set to zero
                    self.results['layer_energy'][i, j] = 0
                    self.results['layer_forces'][i, j] = 0

        # Aggregate the energies and forces from all layers to the top-level results
        self.results["energy"] = self.results["layer_energy"].sum()
        self.results["forces"] = self.results["layer_forces"].sum(axis=(0, 1))

    def calculate_intralayer(self, atoms, layer: int):
        """
        Calculates intra-layer properties for a specified layer.

        :param atoms: The ASE Atoms object.
        :param layer: The index of the layer for intra-layer calculation.
        """
        if layer < len(self.intra_calc_list):
            # Assign the corresponding calculator and atoms for the layer
            calc = self.intra_calc_list[layer]
            atoms_L = self.atoms_layer_list[layer]
        else:
            raise ValueError("Invalid layer index for intralayer calculation.")
        
        # Perform the calculation using the assigned intra-layer calculator
        atoms_L.calc = calc
        calc.calculate(atoms_L)

        # Update the results with energies and forces computed for this layer
        lower_layers_num_atoms = sum([len(layer_atoms) for layer_atoms in self.atoms_layer_list[:layer]])
        layer_atom_indices = [lower_layers_num_atoms, lower_layers_num_atoms + len(atoms_L)]
        self.results['layer_energy'][layer, layer] = atoms_L.calc.results['energy']
        self.results['layer_forces'][layer, layer][layer_atom_indices[0]:layer_atom_indices[1]] = atoms_L.calc.results['forces']

    def calculate_interlayer(self, atoms, layer_1: int, layer_2: int):
        """
        Calculates inter-layer properties between two specified layers.

        :param atoms: The ASE Atoms object.
        :param layer_1: The index of the first layer.
        :param layer_2: The index of the second layer.
        """
        if layer_2 <= layer_1 + 1:
            # Combine atoms from both layers for the calculation
            atoms_L = self.atoms_layer_list[layer_1].copy() + self.atoms_layer_list[layer_2].copy()
            calc = self.inter_calc_list[layer_1]
            atoms_L.calc = calc
            calc.calculate(atoms_L)
            # print(f"Unrelaxed from NLAYER NLAYER: Total_energy {atoms_L.calc.results['energy']:.3f} eV")
            
            # Update the results with energies and forces computed between these layers
            lower_layers_num_atoms = sum([len(layer_atoms) for layer_atoms in self.atoms_layer_list[:layer_1]])
            layer_atom_indices = [lower_layers_num_atoms, lower_layers_num_atoms + len(self.atoms_layer_list[layer_1]) + len(self.atoms_layer_list[layer_2])]
            self.results['layer_energy'][layer_1, layer_2] = atoms_L.calc.results['energy']
            self.results['layer_forces'][layer_1, layer_2][layer_atom_indices[0]:layer_atom_indices[1]] = atoms_L.calc.results['forces']

    def get_atoms_layer_list(self, atoms):
        """
        Separates the provided atoms object into different layers based on the 'atom_types' array and layer_symbols.

        :param atoms: The ASE Atoms object to be separated into layers.
        """
        for layer in range(len(self.intra_calc_list)):
            # Determine the range of atom types for the current layer
            lower_layer_num_atoms = sum([len(layer_atoms) for layer_atoms in self.layer_symbols[:layer]])
            upper_layer_num_atoms = lower_layer_num_atoms + len(self.layer_symbols[layer]) - 1
            
            # Select atoms within the specified range of atom types
            atoms_L = atoms.copy()[np.logical_and(atoms.arrays["atom_types"] <= upper_layer_num_atoms,
                                                  atoms.arrays["atom_types"] >= lower_layer_num_atoms)]
            
            # Add the selected atoms to the layer list
            self.atoms_layer_list.append(atoms_L)
            
