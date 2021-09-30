#!/bin/bash -l
#SBATCH --job-name=reproduce
# specify number of nodes 
#SBATCH -N 1

# specify number of tasks/cores per node required
#SBATCH --ntasks-per-node 8

# specify the walltime e.g 20 mins
#SBATCH -t 72:00:00

# set to email at start,end and failed jobs
# SBATCH --mail-type=ALL
# SBATCH --mail-user=robert.mccarthy@ucdconnect.ie

# run from current directory
cd $SLURM_SUBMIT_DIR

module load singularity/3.5.2

# The following commands are informed by:
# https://people.tuebingen.mpg.de/felixwidmaier/rrc2021/singularity.html#build-and-run-code-in-singularity

singularity run /home/people/16304643/robochallenge/user_image.sif mpirun -np 8 python3 train.py --exp-dir='reproduce' --n-epochs=300 2>&1 | tee reproduce.log

