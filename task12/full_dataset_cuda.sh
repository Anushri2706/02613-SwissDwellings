#!/bin/bash
#BSUB -J full_jacobi_cuda
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 6:00
#BSUB -o logs/full_cuda_%J.out
#BSUB -e logs/full_cuda_%J.err

# === Number of floorplans to process ===
N=4571


source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

# Run, redirect CSV to file, timing goes to stderr -> .err log
python3 -u task8.py $N > full_results_cuda_$LSB_JOBID.csv