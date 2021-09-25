#!/bin/bash -l
#SBATCH --job-name=normalHER
# specify number of nodes 
#SBATCH -N 1

# specify number of tasks/cores per node required
#SBATCH --ntasks-per-node 8

# specify the walltime e.g 20 mins
#SBATCH -t 48:00:00

# set to email at start,end and failed jobs
# SBATCH --mail-type=ALL
# SBATCH --mail-user=robert.mccarthy@ucdconnect.ie

# run from current directory
cd $SLURM_SUBMIT_DIR

module load singularity/3.5.2

# The following commands are informed by:
# https://people.tuebingen.mpg.de/felixwidmaier/rrc2021/singularity.html#build-and-run-code-in-singularity

singularity run /home/people/16304643/robochallenge/user_image.sif mpirun -np 8 python3 train.py --noisy-resets=1 --noise-level=1 --difficulty=3 --trajectory-aware=0 --ep-len=90 --steps-per-goal=30 --step-size=50 --z-reward=0 --z-scale=20 --xy-only=0 --simtoreal=1 --domain-randomization=0 --obs-type='default' --exp-dir='normalHER' --n-epochs=300 2>&1 | tee normalHER.log

