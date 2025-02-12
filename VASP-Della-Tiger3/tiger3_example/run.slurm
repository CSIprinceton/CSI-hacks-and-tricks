#!/bin/bash
#SBATCH --account=car
#SBATCH --job-name=vasp          # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=8      # total number of tasks per node
#SBATCH --cpus-per-task=2        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=00:01:00          # total run time limit (HH:MM:SS)
##SBATCH --mail-type=begin        # send email when job begins
##SBATCH --mail-type=end          # send email when job ends
##SBATCH --mail-user=<YourNetID>@princeton.edu
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PATH=$PATH:/scratch/gpfs/CAR/Packages/vasp.6.5.0/bin
module purge
module load intel-oneapi/2024.2 intel-mpi/oneapi/2021.13 intel-mkl/2024.2
srun vasp_std
