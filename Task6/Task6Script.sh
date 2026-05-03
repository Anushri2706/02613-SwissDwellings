#!/bin/bash
#BSUB -J "mini_project_job"
#BSUB -q hpc
#BSUB -W 60
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1024MB]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -o batch_output/mini_project_%J.out
#BSUB -e batch_output/mini_project_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

echo "workers,time" > timing_Task6.csv

N_PLANS=100

for W in 1 2 4
do
    echo "Running with $W workers..."
    python3 Task6.py $N_PLANS $W >> timing_Task6.csv
done