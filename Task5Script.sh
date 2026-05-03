#!/bin/bash
#BSUB -J "mini_project_job"
#BSUB -q hpc
#BSUB -W 60
#BSUB -n 1
#BSUB -R "rusage[mem=1024MB]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -o batch_output/mini_project_%J.out
#BSUB -e batch_output/mini_project_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

echo "workers,time" > timing_results.csv

N_PLANS=40

# We loop from 1 worker up to the 16 we requested
for W in 1 2 4 8 16
do
    echo "Running with $W workers..."
    # The -n 16 above ensures the OS has 16 slots available
    python3 Task5.py $N_PLANS $W >> timing_results.csv
done