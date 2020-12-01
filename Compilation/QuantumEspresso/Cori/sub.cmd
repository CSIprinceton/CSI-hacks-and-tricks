#!/bin/bash
#SBATCH -N 2
#SBATCH -C knl
#SBATCH -p regular 
#SBATCH -J TestQE
#SBATCH -t 1:00:00
#SBATCH -A ***CHANGE***
#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

PW=/PATH/TO/QE/pw.x

srun -n 136 -N 2 -c 2 --cpu_bind=cores $PW < 01.in > 01.out

