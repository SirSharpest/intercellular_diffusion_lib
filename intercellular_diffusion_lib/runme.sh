#!/bin/bash
#SBATCH --job-name="diffusion_modelling_short"
#SBATCH --output=results_diffusion_modelling-short.txt
#SBATCH --partition=jic-medium
#SBATCH --cpus-per-task=8
#SBATCH --mem 200G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nathan.hughes@jic.ac.uk


source python-3.5.1
source ~/modelling/bin/activate
srun python3 find_dendra_q_value.py
