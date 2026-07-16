# MoSe2/WSe2 bilayer linewidths and lifetimes

From a Perlmutter login node, request the validated GPU geometry and run the entry point:

```bash
salloc -N4 -C gpu --gpus-per-node=4 -q interactive
bash run_salloc_pipeline.sh
```

Validated results: reproduces golden reference; gamma shape (1296,18); gamma_max 0.020164 THz vs golden 0.020166 THz; max|Δ|=6.7e-4 THz, mean|Δ|=7.2e-7 THz; fc2 clean (min -5e-8); interlayer shear mode 0.575 THz (doubly-degenerate pair), breathing mode 0.853 THz; produces 4 figures.

The pipeline HANGS at `srun -n 16 --gpus-per-task=1` (16-rank mpi4py/PMI wireup deadlock, device-independent). Must run at `srun --overlap -n 4 --gpus-per-task=1` (`MPI_RANKS=4`); `--overlap` is required so successive `srun` steps within one allocation don't deadlock on GPU-slice accounting.
