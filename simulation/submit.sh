#!/bin/bash -l
#
#SBATCH --job-name="sim63"   # the name of your job
#SBATCH --time=7-00:00:00           # time in hh:mm:ss you want to reserve for the job
#SBATCH --nodes=1                 # the number of nodes you want to use for the job, 1 node contains 8 processors, in total there are 16 nodes
#SBATCH --ntasks-per-node=1       # the number of processors you want to use per node, when you use more than 1 node always set to 8
#SBATCH --output=job_output/zc.%j.o  # the name of the file where the standard output will be written to
#SBATCH --error=job_output/zc.%j.e   # the name of the file where errors will be written to (if there are errors)

srun time ./simLorenz63.out ../cfg/Lorenz63.cfg                      # start the executable with srun

