#!/bin/bash
#BSUB -J simulation_saved
#BSUB -q hpc
#BSUB -W 5
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2048MB]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -o simulation_saved_%J.out
#BSUB -e simulation_saved_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

python -u simulate_save.py 5