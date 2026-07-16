"""Packaging for mlip_phonon_scattering.

Install editable for development:

    pip install -e .

NOTE ON MACE: this package imports ``mace`` (mace-torch) but it is intentionally
left out of ``install_requires``. In the interlayer workflow ``mace`` must be the
forked ``mace-interlayer`` build (installed editable *before* this package), and
listing ``mace-torch`` here would let pip pull the upstream PyPI build instead,
which lacks the interlayer (``is_interlayer_calc`` / ``layer_ids``) support. Install
``mace-interlayer`` first; see SETUP_MACE_PHONON.md.
"""

from setuptools import find_packages, setup

setup(
    name="mlip_phonon_scattering",
    version="0.1.0",
    description=(
        "MACE foundation-model phonon band-structure and UED-intensity workflow, "
        "with optional interlayer (NLayerCalculator) support for bilayers."
    ),
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/jdgeorga/mlip_phonon_scattering",
    packages=find_packages(exclude=("tests", "examples", "docs")),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "h5py",
        "matplotlib",
        "ase",
        "phonopy",
        # mace-torch: provided by the mace-interlayer fork — see note above.
    ],
    extras_require={
        "test": ["pytest"],
    },
    # Install the stage scripts as command-line tools (they carry a python3 shebang
    # and a main(); they remain runnable by path too, as the example run.sh does).
    scripts=[
        "scripts/relax_structure.py",
        "scripts/generate_displacements.py",
        "scripts/compute_mace_forces.py",
        "scripts/solve_phonons.py",
        "scripts/extract_uq_intensities.py",
        "scripts/relax_structure_qe.py",
        "scripts/compute_qe_forces.py",
    ],
    entry_points={
        "console_scripts": [
            "mlip-linewidth-relax = mlip_phonon_scattering.linewidth.relax:main",
            "mlip-linewidth-phonopy-yaml = mlip_phonon_scattering.linewidth.displacements:main_phonopy",
            "mlip-linewidth-phono3py-yaml = mlip_phonon_scattering.linewidth.displacements:main_phono3py",
            "mlip-linewidth-forces2 = mlip_phonon_scattering.linewidth.forces3:main_forces2",
            "mlip-linewidth-forces3 = mlip_phonon_scattering.linewidth.forces3:main_forces3",
            "mlip-linewidth-forces2-from3 = mlip_phonon_scattering.linewidth.forces3:main_forces2_from3",
            "mlip-linewidth-cache-fc = mlip_phonon_scattering.linewidth.fc_cache:main",
            "mlip-linewidth-scatter-gpu = mlip_phonon_scattering.linewidth.gpu_scattering_W_phonons_bilayer_comm_mesh:main",
            "mlip-linewidth-extract-gamma = mlip_phonon_scattering.linewidth.extract_gamma_example:main",
            "mlip-linewidth-plot = mlip_phonon_scattering.linewidth.plot_linewidth_example:main",
            "mlip-linewidth-validate = mlip_phonon_scattering.linewidth.validate:main",
            "mlip-linewidth-read-dfpt = mlip_phonon_scattering.linewidth.dfpt_read:main",
            "mlip-linewidth-matdyn-modes = mlip_phonon_scattering.linewidth.matdyn_modes:main",
            "mlip-linewidth-compare = mlip_phonon_scattering.linewidth.compare_dispersion:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
    ],
)
