#!/bin/bash
#SBATCH --job-name=h2o      #(-J, --job-name)
#SBATCH --account=mxxxx_g      #(-A, --account)
#SBATCH --constraint=gpu       #(-C, --constraint)
#SBATCH --qos=debug            #(-Q, --qos)
#SBATCH --time=00:10:00        #(-t, --time)
#SBATCH --nodes=1              #(-N, --nodes)
#SBATCH --ntasks=1           #(-n, --ntasks)
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1               #(-G, --gpus) 
#SBATCH --image=docker:deepmodeling/deepmd-kit:2.2.7_cuda11.6_gpu

export SLURM_CPU_BIND="cores"

srun --mpi=pmi2 shifter --image=docker:deepmodeling/deepmd-kit:2.2.7_cuda11.6_gpu --module gpu --volume="$(pwd):/workspace" --workdir="/workspace" bash -c "lmp -in in.lammps -log log"
