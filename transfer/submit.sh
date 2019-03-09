#!/bin/bash -l
#
#SBATCH --job-name="transLorenz"   # the name of your job
#SBATCH --time=7-00:00:00           # time in hh:mm:ss you want to reserve for the job
#SBATCH --nodes=1                 # the number of nodes you want to use for the job, 1 node contains 8 processors, in total there are 16 nodes
#SBATCH  -c 8       # the number of processors you want to use per node, when you use more than 1 node always set to 8
#SBATCH --output=job_output/trans.%j.o  # the name of the file where the standard output will be written to
#SBATCH --error=job_output/trans.%j.e   # the name of the file where errors will be written to (if there are errors)
# Set OMP_NUM_THREADS to the same value as -c
# with a fallback in case it isn't set.
# SLURM_CPUS_PER_TASK is set to the value of -c, but only if -c is explicitly set
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  omp_threads=$SLURM_CPUS_PER_TASK
else
  omp_threads=8
fi
export OMP_NUM_THREADS=$omp_threads

# start the executable with srun
#srun time ./transfer.out ../cfg/Lorenz63.cfg  
srun time ./simTransfer.out ../cfg/Lorenz63.cfg

