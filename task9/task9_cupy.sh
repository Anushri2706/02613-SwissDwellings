#!/bin/bash
#BSUB -J jacobi_cupy
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 0:50
#BSUB -o logs/cupy_%J.out
#BSUB -e logs/cupy_%J.err

# === Number of floorplans to process ===
N=2


source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

# Run, redirect CSV to file, timing goes to stderr -> .err log
python3 -u task9_cupy.py $N > results_cupy_$LSB_JOBID.csv