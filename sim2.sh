#!/bin/sh
#sbatch -p el8 -N 1 --gres=gpu:6 -t 2 -o sim2.stdout -e sim2.stderr ./sim2.sh

module load xl_r spectrum-mpi cuda

taskset -c 0-159:4 mpirun -N 2 /gpfs/u/home/PCPC/PCPCtttl/scratch/a.out -c