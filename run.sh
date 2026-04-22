#!/bin/bash
#BSUB -J simulation
#BSUB -q hpc
#BSUB -W 2
#BSUB -n 1
#BSUB -R "rusage[mem=512MB]"
#BSUB -o simulation_%J.out
#BSUB -e simulation_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

python -u simulate.py