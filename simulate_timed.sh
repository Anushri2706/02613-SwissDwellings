#!/bin/bash
#BSUB -J simulation_timed
#BSUB -q hpc
#BSUB -W 10
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2048MB]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -o simulation_timed_%J.out
#BSUB -e simulation_timed_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

python -u simulate_timed.py 10