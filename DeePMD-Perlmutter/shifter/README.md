# Running MD with Deep Potential on Perlmutter using Shifter

Updated: 2024-01-19\
Author: Yifan Li

To run DeePMD on Perlmutter, the most convienient way is to use DeePMD-kit's docker image and run it with Shifter.
\
The advantages of running DeePMD-kit with Shifter are as follows:
1. You don't need to compile DeePMD-kit manually on Perlmutter.
2. You have the flexibility to use any version of DeePMD-kit. Each release version of DeePMD-kit has a docker image on dockerhub: https://hub.docker.com/r/deepmodeling/deepmd-kit/tags. Besides, you can also access the docker image of each commit of DeePMD-kit from GitHub's container registry: https://github.com/deepmodeling/deepmd-kit/pkgs/container/deepmd-kit. 
3. You can also set up a docerfile to build your own docker image of DeePMD-kit and run it with Shifter. This can be useful if you want to develop and test new features or fix bugs for DeePMD-kit (not covered in this tutorial).

**This tutorial demonstrates how to run classical and path integral MD with the Deep Potential model using Shifter.** The example jobs are in folder `examples`.

## 1. Pull the image
You can choose the image you want to use from [dockerhub](https://hub.docker.com/r/deepmodeling/deepmd-kit/tags) or [GitHub's container registry](https://github.com/deepmodeling/deepmd-kit/pkgs/container/deepmd-kit). For example, if you want to use the 2.2.7 release version of DeePMD-kit, you can pull it with:
```bash
shifterimg pull deepmodeling/deepmd-kit:2.2.7_cuda11.6_gpu
```

## 2. Classical MD
You can find an example job in `examples/md`. Before running it, copy the `frozen_model_compressed.pb` file into this folder. 

The `run.slurm` file is the job script which can be differ from using the other machines. Let me explain it a little bit:
```bash
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
 ```

*Note: remember to change the account name in line 3 to your own account name.*\
To run this job, you can use `sbatch run.slurm`.

The line `#SBATCH --image=docker:deepmodeling/deepmd-kit:2.2.7_cuda11.6_gpu` loads the image which we pulled in step 1.\
The line `srun --mpi=pmi2 shifter --image=docker:deepmodeling/deepmd-kit:2.2.7_cuda11.6_gpu --module gpu --volume="$(pwd):/workspace" --workdir="/workspace" bash -c "lmp -in in.lammps -log log"` is the command to run the job. The `shifter` command is used to run the image. The `--volume` option is used to mount the current folder to the container. The `--workdir` option is used to set the working directory of the container. The `bash -c` command is used to run the LAMMPS command in the container.

## 3. Path integral MD
An example job for PIMD is in `examples/pimd`. The major difference between classical MD and PIMD is that PIMD requires to run multiple replicas. Therefore, one is recommended to use [Nvidia's MPS feature](https://docs.nvidia.com/deploy/mps/index.html) to run multiple replicas on one GPU. To use MPS, [NSRSC provide a script `mps-wrapper.sh`](https://docs.nersc.gov/systems/perlmutter/running-jobs/):
```bash
#!/bin/bash
# Example mps-wrapper.sh usage:
# > srun [srun args] mps-wrapper.sh [cmd] [cmd args]
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
# Launch MPS from a single rank per node
if [ $SLURM_LOCALID -eq 0 ]; then
    CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS nvidia-cuda-mps-control -d
fi
# Wait for MPS to start
sleep 5
# Run the command
"$@"
# Quit MPS control daemon before exiting
if [ $SLURM_LOCALID -eq 0 ]; then
    echo quit | nvidia-cuda-mps-control
fi
```
The line `"$@"` takes the command following this script as the command. By using this script, you can use the MPS feature and therefore run multiple DP jobs simultaneously on one GPU. The `run.slurm` script for a 4-bead PIMD job using 1 GPU looks like:
```bash
#!/bin/bash                                                                                                                                                                 
#SBATCH --job-name=h2o4b      #(-J, --job-name)
#SBATCH --account=mxxxx_g      #(-A, --account)
#SBATCH --constraint=gpu       #(-C, --constraint)
#SBATCH --qos=debug            #(-Q, --qos)
#SBATCH --time=00:10:00        #(-t, --time)
#SBATCH --nodes=1              #(-N, --nodes)
#SBATCH --ntasks=4           #(-n, --ntasks)
#SBATCH --ntasks-per-node=4
#SBATCH --gpus=1               #(-G, --gpus) 
#SBATCH --image=docker:deepmodeling/deepmd-kit:2.2.7_cuda11.6_gpu

export SLURM_CPU_BIND="cores"

srun --mpi=pmi2 shifter --image=docker:deepmodeling/deepmd-kit:2.2.7_cuda11.6_gpu --module gpu --volume="$(pwd):/workspace" --workdir="/workspace" bash -c "./mps-wrapper.sh lmp -in in.lammps -p 4x1 -log log"
```

*Note: remember to change the account name in line 3 to your own account name.*\
To run this job, you can use `sbatch run.slurm`. Before running it, copy the `frozen_model_compressed.pb` file into this folder. 

The main difference is that we use `./mps-wrapper.sh lmp -in in.lammps -p 4x1 -log log` to run the LAMMPS job. The `-p 4x1` option is used to run 4 beads for PIMD. 

## 4. Interactive job
To run interactive jobs, you can modify and use the following command:
```bash
srun --nodes 1 -n 1 --tasks-per-node=1 --cpus-per-task=1 --qos interactive --time 01:59:00 --constraint gpu --gpus 1 --account=mxxxx_g --cpu-bind=cores --mpi=pmi2 shifter --image=docker:deepmodeling/deepmd-kit:2.2.7_cuda11.6_gpu --module gpu --volume="$(pwd):/workspace" --workdir="/workspace" bash -c "lmp -in in.lammps"
```