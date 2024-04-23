#!/bin/sh
#sbatch -p el8 -N 2 --gres=gpu:6 -t 2 -o sim64.stdout -e sim64.stderr ./sim64.sh

module load xl_r spectrum-mpi cuda

taskset -c 0-159:4 mpirun -N 32 /gpfs/u/home/PCPE/PCPEpwrr/scratch/a.out -c