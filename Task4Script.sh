#! /bin/bash
#BSUB -J "mini_project_job"
#BSUB -q hpc
#BSUB -W 2
#BSUB -n 1
#BSUB -R "rusage[mem=1024MB]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -o batch_output/mini_project_%J.out
#BSUB -e batch_output/mini_project_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613
kernprof -l -v simulate.py