#!/bin/sh
#sbatch -p el8 -N 4 --gres=gpu:6 -t 2 -o sim128.stdout -e sim128.stderr ./sim128.sh

module load xl_r spectrum-mpi cuda

taskset -c 0-159:4 mpirun -N 32 /gpfs/u/home/PCPC/PCPCtttl/scratch/a.out -c