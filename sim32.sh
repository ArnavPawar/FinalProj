#!/bin/sh
#sbatch -p el8 -N 1 --gres=gpu:6 -t 2 -o sim32.stdout -e sim32.stderr ./sim32.sh

module load xl_r spectrum-mpi cuda

taskset -c 0-159:4 mpirun -N 32 /gpfs/u/home/PCPC/PCPCtttl/scratch/a.out -c