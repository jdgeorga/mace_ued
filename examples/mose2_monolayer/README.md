# MoSe2 monolayer linewidths and lifetimes

From a Perlmutter login node, request the validated GPU geometry and run the entry point:

```bash
salloc -N4 -C gpu --gpus-per-node=4 -q interactive
bash run_salloc_pipeline.sh
```

Validated results: new single-model path (fc3 supercell 4x4x1 / phonon supercell 8x8x1, mesh 36x36x1); physics-validated via self-contained checks (there is no golden reference for monolayer — it is physics-validated, not "reproduces a reference"); gamma shape (1296,9); gamma_max 0.01741 THz; fc2 clean (min +3.2e-8, 9 bands); Gamma-point acoustic modes have gamma=0; tau>0 everywhere; produces 4 figures.

The pipeline HANGS at `srun -n 16 --gpus-per-task=1` (16-rank mpi4py/PMI wireup deadlock, device-independent). Must run at `srun --overlap -n 4 --gpus-per-task=1` (`MPI_RANKS=4`); `--overlap` is required so successive `srun` steps within one allocation don't deadlock on GPU-slice accounting.
