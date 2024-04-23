#!/bin/sh
#sbatch -p el8 -N 1 --gres=gpu:6 -t 2 -o sim16.stdout -e sim16.stderr ./sim16.sh

module load xl_r spectrum-mpi cuda

taskset -c 0-159:4 mpirun -N 16 /gpfs/u/home/PCPC/PCPCtttl/scratch/a.out -c