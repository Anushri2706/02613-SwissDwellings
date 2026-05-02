#!/bin/bash
#BSUB -J jacobi_cpu
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 1:00
#BSUB -o logs/ref_cpu_%J.out
#BSUB -e logs/ref_cpu_%J.err

N=40
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026
python3 -u ref_simulate.py $N > ref_cpu_$LSB_JOBID.csv