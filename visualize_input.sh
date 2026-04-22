#!/bin/bash
#BSUB -J visualize_input
#BSUB -q hpc
#BSUB -W 3
#BSUB -n 1
#BSUB -R "rusage[mem=512MB]"
#BSUB -o visualize_input_%J.out
#BSUB -e visualize_input_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

python visualize_input.py 10