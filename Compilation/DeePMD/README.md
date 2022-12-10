# Compilation of DeePMD and Lammps

## TigerGPU


This script follows these guidelines:
 - https://github.com/deepmodeling/deepmd-kit/tree/r0.12
 - https://github.com/deepmodeling/deepmd-kit/blob/r0.12/doc/multi-gpu-support.md
 - https://github.com/deepmodeling/deepmd-kit/blob/r0.12/doc/install-tf.1.14-gpu.md

and a little bit from here:
 - https://www.tensorflow.org/install/source
 - https://github.com/tensorflow/tensorflow/issues/30703

Script provided by Tom Gartner

```
folder_name=Software-deepmd-kit-1.0
num_cores=12

module purge
module load rh/devtoolset/4
module load cudatoolkit/10.0
module load cudnn/cuda-10.0/7.6.3
module load openmpi/gcc/3.1.3/64
#module load anaconda3/2019.10
#module load fftw/gcc/openmpi-3.1.4/3.3.8

new_folder=`pwd`/$folder_name
mkdir $new_folder
cd $new_folder
echo "Installing in $new_folder"

######################################################
# Install python package of tensorflow
######################################################

tensorflow_venv=$new_folder/tensorflow-venv
python3 -m venv $tensorflow_venv
source $tensorflow_venv/bin/activate
pip install --upgrade pip
pip install --upgrade tensorflow-gpu==1.14.0

read -rsp $'Press any key to continue...\n' -n1 key

######################################################
# Build bazel
######################################################

cd $new_folder
wget https://github.com/bazelbuild/bazel/releases/download/0.24.1/bazel-0.24.1-dist.zip
mkdir bazel-0.24.1
cd bazel-0.24.1
unzip ../bazel-0.24.1-dist.zip
rm ../bazel-0.24.1-dist.zip
# Add line --host_javabase=@local_jdk//:jdk before compiling !!!!
# goes right after   --platforms=@bazel_tools//platforms:target_platform
sed -i '/build Bazel/i \ \ --host_javabase=@local_jdk//:jdk \\' compile.sh
./compile.sh
export PATH=`pwd`/output:$PATH
rm -rf ~/.cache/bazel/_bazel_tgartner/*

######################################################
# Build tensorflow
######################################################

cd $new_folder
git clone https://github.com/tensorflow/tensorflow tensorflow -b v1.14.0 --depth=1
cd tensorflow
tensorflow_root=`pwd`
echo "Tensor flow folder is $tensorflow_root"
# The following environment variables can be used to avoid user input
export TF_CUDA_VERSION="10.0"
export CUDA_TOOLKIT_PATH="/usr/local/cuda-10.0"
export TF_CUDNN_VERSION="7.6.3"
export CUDNN_INSTALL_PATH="/usr/local/cudnn/cuda-10.0/7.6.3"
#export PYTHON_BIN_PATH="/home/tgartner/Programs/DeepMD/tensorflow-venv/bin/python"
#export PYTHON_LIB_PATH="/home/tgartner/Programs/DeepMD/tensorflow-venv/lib64/python3.6/site-packages"
#export OMP_NUM_THREADS="1"
#export TF_NEED_OPENCL_SYCL="0"
#export TF_NEED_ROCM="0"
#export TF_NEED_CUDA="1"
#export NCCL_INSTALL_PATH="/usr/local/cuda-10.0/targets/ppc64le-linux"
#export TF_NCCL_VERSION="2.4.7"
#export TF_CUDA_COMPUTE_CAPABILITIES="7.0"
#export TF_CUDA_CLANG="0"
#export GCC_HOST_COMPILER_PATH="/usr/bin/gcc"
# Change lib to lib64 for MPI compilation
# This hack is needed in some cases, I leave it for future use
#sed -i "s/mpi_home, 'lib'/mpi_home, 'lib64'/g" configure.py
#sed -i "s/lib\/libmpi/lib64\/libmpi/g" configure.py
#export MPI_HOME="/usr/local/openmpi/4.0.1/gcc/ppc64le"
./configure
bazel build -c opt --verbose_failures //tensorflow:libtensorflow_cc.so

read -rsp $'Press any key to continue...\n' -n1 key

cd $tensorflow_root

mkdir $tensorflow_root/lib
cp -d bazel-bin/tensorflow/libtensorflow_cc.so* $tensorflow_root/lib/
cp -d bazel-bin/tensorflow/libtensorflow_framework.so* $tensorflow_root/lib/
cp -d $tensorflow_root/lib/libtensorflow_framework.so.1 $tensorflow_root/lib/libtensorflow_framework.so

mkdir -p $tensorflow_root/include/tensorflow
cp -r bazel-genfiles/* $tensorflow_root/include/
cp -r tensorflow/cc $tensorflow_root/include/tensorflow
cp -r tensorflow/core $tensorflow_root/include/tensorflow
cp -r third_party $tensorflow_root/include
cp -r bazel-tensorflow/external/eigen_archive/Eigen/ $tensorflow_root/include
cp -r bazel-tensorflow/external/eigen_archive/unsupported/ $tensorflow_root/include
rsync -avzh --include '*/' --include '*.h' --include '*.inc' --exclude '*' bazel-tensorflow/external/protobuf_archive/src/ $tensorflow_root/include/
rsync -avzh --include '*/' --include '*.h' --include '*.inc' --exclude '*' bazel-tensorflow/external/com_google_absl/absl/ $tensorflow_root/include/absl

cd $tensorflow_root/include
find . -name "*.cc" -type f -delete

rm -rf ~/.cache/bazel/_bazel_tgartner/*

read -rsp $'Press any key to continue...\n' -n1 key

######################################################
# Build python package
######################################################

# Not sure when this is needed
#cd $new_folder
#git clone https://github.com/tensorflow/tensorflow tensorflow-python -b v1.14.0 --depth=1
#cd tensorflow-python
## Add the following line:
## include "tensorflow/core/util/gpu_kernel_helper.h"
## to line 23 of
## tensorflow/contrib/mpi_collectives/kernels/ring.cu.cc
## This I found in:
## https://github.com/tensorflow/tensorflow/issues/30703
#./configure
#bazel build --config=cuda //tensorflow/tools/pip_package:build_pip_package
##bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
#./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
#pip install /tmp/tensorflow_pkg/tensorflow-*.whl
##pip install /tmp/tensorflow_pkg/tensorflow-1.14.0-cp36-cp36m-linux_ppc64le.whl
#rm -rf ~/.cache/bazel/_bazel_tgartner/*

read -rsp $'Press any key to continue...\n' -n1 key

######################################################
# Build DeePMD
######################################################

cd $new_folder
deepmd_root=`pwd`
git clone https://github.com/deepmodeling/deepmd-kit.git deepmd-kit-1.0 -b r1.0
cd deepmd-kit-1.0
deepmd_source_dir=`pwd`
# the following line is to fix an error in the python installation of deepmd-kit
# where pip was not locating the tensorflow python package. This line may not be
# necessary in the future if the deepmd-kit setup scripts are updated
sed -i 's/ninja\"\, /ninja", "tensorflow-gpu==1.14.0", /g' pyproject.toml
pip install .
cd $deepmd_source_dir/source
sed -i 's/set (LIB_DEEPMD_NATIVE.*/#set (LIB_DEEPMD_NATIVE  "deepmd_native_md")/g' CMakeLists.txt
sed -i 's/set (LIB_DEEPMD_IPI.*/#set (LIB_DEEPMD_IPI     "deepmd_ipi")/g' CMakeLists.txt
sed -i 's/add_subdirectory (md.*/#add_subdirectory (md\/)/g' CMakeLists.txt
sed -i 's/add_subdirectory (ipi.*/#add_subdirectory (ipi\/)/g' CMakeLists.txt
mkdir build 
cd build
cmake3 -DTENSORFLOW_ROOT=$tensorflow_root -DCMAKE_INSTALL_PREFIX=$deepmd_root -DUSE_CUDA_TOOLKIT=true  ..
make -j $num_cores
make install
make lammps

######################################################
# Install lammps
######################################################
cd $new_folder
wget https://lammps.sandia.gov/tars/lammps-stable.tar.gz
tar -xvf lammps-stable.tar.gz
rm lammps-stable.tar.gz
cd lammps-7Aug19/src/
cp -r $deepmd_source_dir/source/build/USER-DEEPMD .
make yes-user-deepmd
make mpi -j $num_cores
cd $new_folder
```


TO DO: Include the appropriate optimization flags.
Tensorflow gives this warning:
```
Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
```

## Traverse


```
folder_name=DeepMD
num_cores=20

module purge
module load rh/devtoolset/7
module load cudatoolkit/10.0
module load cudnn/cuda-10.0/7.6.1
module load openmpi/gcc/3.1.4/64
module load anaconda3/2019.3

new_folder=`pwd`/$folder_name
mkdir $new_folder
cd $new_folder
echo "Installing in $new_folder"

######################################################
# Prepare python environment
######################################################
conda create --name tensorflow-venv
conda activate tensorflow-venv
conda install pip
pip install -U --user pip six numpy wheel setuptools mock 'future>=0.17.1'
pip install -U --user keras_applications --no-deps
pip install -U --user keras_preprocessing --no-deps

######################################################
# Compile bazel
######################################################
cd $new_folder
wget https://github.com/bazelbuild/bazel/releases/download/0.24.1/bazel-0.24.1-dist.zip
mkdir bazel-0.24.1
cd bazel-0.24.1
unzip ../bazel-0.24.1-dist.zip
rm ../bazel-0.24.1-dist.zip
sed -i '/target_platform/a \ \ --host_javabase=@local_jdk//:jdk \\' compile.sh
./compile.sh
export PATH=`pwd`/output:$PATH

######################################################
# Configure tensorflow
######################################################
cd $new_folder
git clone https://github.com/tensorflow/tensorflow tensorflow -b v1.14.0 --depth=1
cd tensorflow
tensorflow_root=`pwd`
echo "Tensor flow folder is $tensorflow_root"
export TF_CUDA_VERSION="10.0"
export CUDA_TOOLKIT_PATH="/usr/local/cuda-10.0"
export TF_CUDNN_VERSION="7.6.1"
export CUDNN_INSTALL_PATH="/usr/local/cudnn/cuda-10.0/7.6.1"
export NCCL_INSTALL_PATH="/usr/local/nccl/cuda-10.0/2.5.6/"
export TF_NCCL_VERSION="2.5.6"
export TF_CUDA_COMPUTE_CAPABILITIES="7.0"
export MPI_HOME="/usr/local/openmpi/3.1.4/devtoolset-8/ppc64le"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/nccl/cuda-10.0/2.5.6/lib64"
./configure
# Before proceeding replace:
#    nullptr,                     /* tp_print */      
# with                                                 
#    NULL,                        /* tp_print */      
# in the files:                                       
# tensorflow/python/eager/pywrap_tfe_src.cc           
# tensorflow/python/lib/core/bfloat16.cc              
# tensorflow/python/lib/core/ndarray_tensor_bridge.cc 

######################################################
# Compile tensorflow pip package
######################################################
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip install -U --user /tmp/tensorflow_pkg/tensorflow-1.14.0-cp38-cp38-linux_ppc64le.whl

######################################################
# Compile tensorflow
######################################################
bazel build -c opt --verbose_failures //tensorflow:libtensorflow_cc.so

cd $tensorflow_root
mkdir $tensorflow_root/lib
cp -d bazel-bin/tensorflow/libtensorflow_cc.so* $tensorflow_root/lib/
cp -d bazel-bin/tensorflow/libtensorflow_framework.so* $tensorflow_root/lib/
cp -d $tensorflow_root/lib/libtensorflow_framework.so.1 $tensorflow_root/lib/libtensorflow_framework.so

mkdir -p $tensorflow_root/include/tensorflow
cp -r bazel-genfiles/* $tensorflow_root/include/
cp -r tensorflow/cc $tensorflow_root/include/tensorflow
cp -r tensorflow/core $tensorflow_root/include/tensorflow
cp -r third_party $tensorflow_root/include
cp -r bazel-tensorflow/external/eigen_archive/Eigen/ $tensorflow_root/include
cp -r bazel-tensorflow/external/eigen_archive/unsupported/ $tensorflow_root/include
rsync -avzh --include '*/' --include '*.h' --include '*.inc' --exclude '*' bazel-tensorflow/external/protobuf_archive/src/ $tensorflow_root/include/
rsync -avzh --include '*/' --include '*.h' --include '*.inc' --exclude '*' bazel-tensorflow/external/com_google_absl/absl/ $tensorflow_root/include/absl

cd $tensorflow_root/include
find . -name "*.cc" -type f -delete

rm -rf ~/.cache/bazel/_bazel_*

######################################################
# Install DeepMD
######################################################
cd $new_folder
deepmd_root=`pwd`
git clone https://github.com/deepmodeling/deepmd-kit.git deepmd-kit-1.0 -b r1.0
cd deepmd-kit-1.0
deepmd_source_dir=`pwd`
cd $deepmd_source_dir/source
mkdir build
cd build
cmake3 -DTENSORFLOW_ROOT=$tensorflow_root -DCMAKE_INSTALL_PREFIX=$deepmd_root -DUSE_CUDA_TOOLKIT=true  ..
make -j $num_cores
make install
make lammps

######################################################
# Install lammps
######################################################
cd $new_folder
git clone https://github.com/lammps/lammps
cd lammps/src/
cp -r $deepmd_source_dir/source/build/USER-DEEPMD .
make yes-user-deepmd
make mpi -j $num_cores
cd $new_folder

```

## DellaGPU

Following instructions from:

https://github.com/deepmodeling/deepmd-kit/blob/7e534be15448fafbca82b6b71b540dde81edb373/doc/install.md
https://github.com/deepmodeling/deepmd-kit/blob/7e534be15448fafbca82b6b71b540dde81edb373/doc/install-tf.2.3.md

```

#Load modules
module load openmpi/gcc/4.1.0 anaconda3/2020.7 cudatoolkit/11.1 cudnn/cuda-11.x/8.2.0

#Conda environment
conda create --name dpmd python=3.8 cudatoolkit=11 cudnn=8 --channel nvidia
conda activate dpmd
pip install tensorflow-gpu==2.4
#Download deepmd-kit
git clone --recursive https://github.com/deepmodeling/deepmd-kit.git deepmd-kit
cd deepmd-kit
pip install .
deepmd_source=`pwd`

#Tensorflow C++ library
cd 
git clone https://github.com/tensorflow/tensorflow tensorflow -b v2.4.0 --depth=1
##Using Bazel compiled on DellaGPU (version 3.5.0)
./configure
```
 
Here is an example on how to configure tensorflow

```
(dpmd) [mandrade@della-gpu tensorflow]$ ./configure 
You have bazel 3.5.0- (@non-git) installed.
Please specify the location of python. [Default is /home/mandrade/.conda/envs/dpmd/bin/python3]: 


Found possible Python library paths:
  /home/mandrade/.conda/envs/dpmd/lib/python3.8/site-packages
Please input the desired Python library path to use.  Default is [/home/mandrade/.conda/envs/dpmd/lib/python3.8/site-packages]

Do you wish to build TensorFlow with ROCm support? [y/N]: 
No ROCm support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: y
CUDA support will be enabled for TensorFlow.

Do you wish to build TensorFlow with TensorRT support? [y/N]: 
No TensorRT support will be enabled for TensorFlow.

Could not find any cudnn.h, cudnn_version.h matching version '' in any subdirectory:
        ''
        'include'
        'include/cuda'
        'include/*-linux-gnu'
        'extras/CUPTI/include'
        'include/cuda/CUPTI'
        'local/cuda/extras/CUPTI/include'
of:
        '/lib'
        '/lib64'
        '/opt/dell/srvadmin/lib64'
        '/opt/dell/srvadmin/lib64/openmanage'
        '/opt/dell/srvadmin/lib64/openmanage/smpop'
        '/opt/mellanox/hcoll/lib'
        '/opt/mellanox/sharp/lib'
        '/usr'
        '/usr/lib64//bind9-export'
        '/usr/lib64/R/lib'
        '/usr/lib64/atlas'
        '/usr/lib64/atlas-corei2'
        '/usr/lib64/dyninst'
        '/usr/local/cuda'
        '/usr/local/cuda-10.2/targets/x86_64-linux/lib'
        '/usr/local/cuda-11.1/targets/x86_64-linux/lib'
        '/usr/local/cuda/targets/x86_64-linux/lib'
        '/usr/local/cudnn'
Asking for detailed CUDA configuration...

Please specify the CUDA SDK version you want to use. [Leave empty to default to CUDA 10]: 11.1


Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7]: 8.2.0


Please specify the locally installed NCCL version you want to use. [Leave empty to use http://github.com/nvidia/nccl]: 


Please specify the comma-separated list of base paths to look for CUDA libraries and headers. [Leave empty to use the default]: /usr/local/cuda-11.1/,/usr/local/cudnn/cuda-11.3/8.2.0/


Found CUDA 11.1 in:
    /usr/local/cuda-11.1/targets/x86_64-linux/lib
    /usr/local/cuda-11.1/targets/x86_64-linux/include
Found cuDNN 8 in:
    /usr/local/cudnn/cuda-11.3/8.2.0/lib64
    /usr/local/cudnn/cuda-11.3/8.2.0/include


Please specify a list of comma-separated CUDA compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus. Each capability can be specified as "x.y" or "compute_xy" to include both virtual and binary GPU code, or as "sm_xy" to only include the binary code.
Please note that each additional compute capability significantly increases your build time and binary size, and that TensorFlow only supports compute capabilities >= 3.5 [Default is: 3.5,7.0]: 8.0


Do you want to use clang as CUDA compiler? [y/N]: 
nvcc will be used as CUDA compiler.

Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]: 


Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native -Wno-sign-compare]: 


Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: 
Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See .bazelrc for more details.
	--config=mkl         	# Build with MKL support.
	--config=mkl_aarch64 	# Build with oneDNN support for Aarch64.
	--config=monolithic  	# Config for mostly static monolithic build.
	--config=ngraph      	# Build with Intel nGraph support.
	--config=numa        	# Build with NUMA support.
	--config=dynamic_kernels	# (Experimental) Build kernels into separate shared objects.
	--config=v2          	# Build TensorFlow 2.x instead of 1.x.
Preconfigured Bazel build configs to DISABLE default on features:
	--config=noaws       	# Disable AWS S3 filesystem support.
	--config=nogcp       	# Disable GCP support.
	--config=nohdfs      	# Disable HDFS support.
	--config=nonccl      	# Disable NVIDIA NCCL support.
Configuration finished
```

Now you can run Bazel to compile the c++ library of tensorflow

```
bazel build -c opt --verbose_failures //tensorflow:libtensorflow_cc.so
mkdir ~/deepmd_root
tensorflow_root=`realpath ~/deepmd_root`
mkdir -p $tensorflow_root/lib
cp -d bazel-bin/tensorflow/libtensorflow_cc.so* $tensorflow_root/lib/
cp -d bazel-bin/tensorflow/libtensorflow_framework.so* $tensorflow_root/lib/
cp -d $tensorflow_root/lib/libtensorflow_framework.so.2 $tensorflow_root/lib/libtensorflow_framework.so
mkdir -p $tensorflow_root/include/tensorflow
rsync -avzh --exclude '_virtual_includes/' --include '*/' --include '*.h' --include '*.inc' --exclude '*' bazel-bin/ $tensorflow_root/include/
rsync -avzh --include '*/' --include '*.h' --include '*.inc' --exclude '*' tensorflow/cc $tensorflow_root/include/tensorflow/
rsync -avzh --include '*/' --include '*.h' --include '*.inc' --exclude '*' tensorflow/core $tensorflow_root/include/tensorflow/
rsync -avzh --include '*/' --include '*' --exclude '*.cc' third_party/ $tensorflow_root/include/third_party/
rsync -avzh --include '*/' --include '*' --exclude '*.txt' bazel-tensorflow/external/eigen_archive/Eigen/ $tensorflow_root/include/Eigen/
rsync -avzh --include '*/' --include '*' --exclude '*.txt' bazel-tensorflow/external/eigen_archive/unsupported/ $tensorflow_root/include/unsupported/
rsync -avzh --include '*/' --include '*.h' --include '*.inc' --exclude '*' bazel-tensorflow/external/com_google_protobuf/src/google/ $tensorflow_root/include/google/
rsync -avzh --include '*/' --include '*.h' --include '*.inc' --exclude '*' bazel-tensorflow/external/com_google_absl/absl/ $tensorflow_root/include/absl/
```

Finally, you can compile the C++ library of DeepMD-Kit

```
cd $deepmd_source/source
deepmd_root=$tensorflow_root
mkdir build
cd build
cmake -DUSE_CUDA_TOOLKIT=true -DTENSORFLOW_ROOT=$tensorflow_root -DCMAKE_INSTALL_PREFIX=$deepmd_root ..
make -j 4
make install
make lammps
```

You're now ready to compile Lammps with DeepMD-Kit

```
cd
wget https://github.com/lammps/lammps/archive/stable_29Oct2020.tar.gz
tar -xf stable_29Oct2020.tar.gz
cd lammps-stable_29Oct2020/src
cp -r $deepmd_source/source/build/USER-DEEPMD .
make yes-deepmd yes-kspace
make -j 12 mpi
```

## Perlmutter

Compilation performed on 10/09/2021.

Load the following modules:

```
module load gcc/9.3.0 cudnn/8.2.0 python/3.8-anaconda-2020.11
```

The following modules were up:

```
  1) craype-x86-rome          5) xpmem/2.2.40-7.0.1.0_2.7__g1d7a24d.shasta   9) PrgEnv-nvidia/8.1.0        13) xalt/2.10.2
  2) libfabric/1.11.0.4.79    6) craype/2.7.10                              10) cray-pmi/6.0.13            14) darshan/3.2.1    (io)
  3) craype-network-ofi       7) cray-dsmml/0.2.1                           11) cray-pmi-lib/6.0.13        15) gcc/9.3.0        (c)
  4) perftools-base/21.09.0   8) cray-libsci/21.08.1.2                      12) cuda/11.3.0         (g,c)  16) cray-mpich/8.1.9 (mpi)
```
Download Bazel binary (version 3.7.2):
```
mkdir ~/bin
wget https://github.com/bazelbuild/bazel/releases/download/3.7.2/bazel-3.7.2-linux-x86_64 ~/bin/
cd ~/bin
chmod +x bazel-3.7.2-linux-x86_64
ln -s bazel-3.7.2-linux-x86_64 bazel
cd
```
```
mkdir DeepMD
cd DeepMD
git clone https://github.com/tensorflow/tensorflow tensorflow -b v2.4.0 --depth=1
cd tensorflow
```

```
./configure
You have bazel 3.7.2 installed.
Please specify the location of python. [Default is /global/common/software/nersc/cos1.3/python/3.8-anaconda-2020.11/bin/python3]: 


Found possible Python library paths:
  /global/common/software/nersc/cos1.3/python/3.8-anaconda-2020.11/lib/python3.8/site-packages
Please input the desired Python library path to use.  Default is [/global/common/software/nersc/cos1.3/python/3.8-anaconda-2020.11/lib/python3.8/site-packages]

Do you wish to build TensorFlow with ROCm support? [y/N]: 
No ROCm support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: Y
CUDA support will be enabled for TensorFlow.

Do you wish to build TensorFlow with TensorRT support? [y/N]: 
No TensorRT support will be enabled for TensorFlow.

Could not find any cuda.h matching version '' in any subdirectory:
        ''
        'include'
        'include/cuda'
        'include/*-linux-gnu'
        'extras/CUPTI/include'
        'include/cuda/CUPTI'
        'local/cuda/extras/CUPTI/include'
of:
        '/lib'
        '/lib64'
        '/opt/cray/pe/lib64'
        '/opt/cray/pe/lib64/cce'
        '/opt/cray/xpmem/default/lib64'
        '/usr'
        '/usr/lib'
        '/usr/lib/tls'
        '/usr/lib64'
        '/usr/lib64/tls'
Asking for detailed CUDA configuration...

Please specify the CUDA SDK version you want to use. [Leave empty to default to CUDA 10]: 11.3


Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7]: 8.2


Please specify the locally installed NCCL version you want to use. [Leave empty to use http://github.com/nvidia/nccl]: 


Please specify the comma-separated list of base paths to look for CUDA libraries and headers. [Leave empty to use the default]: /global/common/software/nersc/cos1.3/cuda/11.3.0,/global/common/software/nersc/cos1.3/cudnn/8.2.0/cuda/11.3/


Found CUDA 11.3 in:
    /global/common/software/nersc/cos1.3/cuda/11.3.0/targets/x86_64-linux/lib
    /global/common/software/nersc/cos1.3/cuda/11.3.0/targets/x86_64-linux/include
Found cuDNN 8 in:
    /global/common/software/nersc/cos1.3/cudnn/8.2.0/cuda/11.3/lib64
    /global/common/software/nersc/cos1.3/cudnn/8.2.0/cuda/11.3/include


Please specify a list of comma-separated CUDA compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus. Each capability can be specified as "x.y" or "compute_xy" to include both virtual and binary GPU code, or as "sm_xy" to only include the binary code.
Please note that each additional compute capability significantly increases your build time and binary size, and that TensorFlow only supports compute capabilities >= 3.5 [Default is: 8.0]: 


Do you want to use clang as CUDA compiler? [y/N]: 
nvcc will be used as CUDA compiler.

Please specify which gcc should be used by nvcc as the host compiler. [Default is /opt/cray/pe/gcc/9.3.0/bin/gcc]: 


Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native -Wno-sign-compare]: 


Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: 
Not configuring the WORKSPACE for Android builds.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See .bazelrc for more details.
	--config=mkl         	# Build with MKL support.
	--config=mkl_aarch64 	# Build with oneDNN support for Aarch64.
	--config=monolithic  	# Config for mostly static monolithic build.
	--config=ngraph      	# Build with Intel nGraph support.
	--config=numa        	# Build with NUMA support.
	--config=dynamic_kernels	# (Experimental) Build kernels into separate shared objects.
	--config=v2          	# Build TensorFlow 2.x instead of 1.x.
Preconfigured Bazel build configs to DISABLE default on features:
	--config=noaws       	# Disable AWS S3 filesystem support.
	--config=nogcp       	# Disable GCP support.
	--config=nohdfs      	# Disable HDFS support.
	--config=nonccl      	# Disable NVIDIA NCCL support.
Configuration finished
```

Build tensorflow
```
bazel build -c opt --verbose_failures //tensorflow:libtensorflow_cc.so
tensorflow_root=`realpath ..`
mkdir -p $tensorflow_root/lib
cp -d bazel-bin/tensorflow/libtensorflow_cc.so* $tensorflow_root/lib/
cp -d bazel-bin/tensorflow/libtensorflow_framework.so* $tensorflow_root/lib/
cp -d $tensorflow_root/lib/libtensorflow_framework.so.2 $tensorflow_root/lib/libtensorflow_framework.so
mkdir -p $tensorflow_root/include/tensorflow
rsync -avzh --exclude '_virtual_includes/' --include '*/' --include '*.h' --include '*.inc' --exclude '*' bazel-bin/ $tensorflow_root/include/
rsync -avzh --include '*/' --include '*.h' --include '*.inc' --exclude '*' tensorflow/cc $tensorflow_root/include/tensorflow/
rsync -avzh --include '*/' --include '*.h' --include '*.inc' --exclude '*' tensorflow/core $tensorflow_root/include/tensorflow/
rsync -avzh --include '*/' --include '*' --exclude '*.cc' third_party/ $tensorflow_root/include/third_party/
rsync -avzh --include '*/' --include '*' --exclude '*.txt' bazel-tensorflow/external/eigen_archive/Eigen/ $tensorflow_root/include/Eigen/
rsync -avzh --include '*/' --include '*' --exclude '*.txt' bazel-tensorflow/external/eigen_archive/unsupported/ $tensorflow_root/include/unsupported/
rsync -avzh --include '*/' --include '*.h' --include '*.inc' --exclude '*' bazel-tensorflow/external/com_google_protobuf/src/google/ $tensorflow_root/include/google/
rsync -avzh --include '*/' --include '*.h' --include '*.inc' --exclude '*' bazel-tensorflow/external/com_google_absl/absl/ $tensorflow_root/include/absl/
```

Compile DeepMD-Kit
```
git clone --recursive https://github.com/deepmodeling/deepmd-kit.git deepmd-kit
git checkout v1.3.3
cd source
mkdir build
cd build
deepmd_root=$tensorflow_root
CC=gcc cmake -DUSE_CUDA_TOOLKIT=true -DTENSORFLOW_ROOT=$tensorflow_root -DCMAKE_INSTALL_PREFIX=$deepmd_root ..
make -j 4 
make install 
make lammps
```

Compile Lammps

```
cd $tensorflow_root
wget https://github.com/lammps/lammps/archive/stable_29Oct2020.tar.gz
tar -xf stable_29Oct2020.tar.gz
cd lammps-stable_29Oct2020/src
cp -r ../../../deepmd-kit/source/build/USER-DEEPMD/ .
sed -i 's/mpicxx/CC/g' MPI/Makefile.mpi
make yes-user-deepmd yes-kspace
#One could also make yes-gpu. Before doing so, go to ../lib/gpu and compile this library first
make -j 4 mpi
```

## SUMMIT

Code compiled on 12/10/2022
DeepMD-Kit commit: 6e3d4a626af965e951298f1bce9a9d0a2bbda317
DeepMD-Kit version: 2.1.5
Lammps version: stable_23Jun2022

Load modules:

```
module load gcc cuda cmake spectrum-mpi/10.4.0.3-20210112 open-ce/1.2.0-py38-0
```

The following modules were loaded during compilation:

```
  1) lsf-tools/2.0                4) xalt/1.2.1             7) nsight-compute/2021.2.1     10) gcc/9.1.0
  2) hsi/5.0.2.p5                 5) DefApps                8) nsight-systems/2021.3.1.54  11) spectrum-mpi/10.4.0.3-20210112
  3) darshan-runtime/3.3.0-lite   6) open-ce/1.2.0-py38-0   9) cuda/11.0.3                 12) cmake/3.23.2
```

Go to the DeepMD-kit source folder and execute the commands below. Make sure you have the variable "deepmd_root" set to some existing path.

```
mkdir build
cd build
CC=gcc cmake -DUSE_CUDA_TOOLKIT=true -DCMAKE_INSTALL_PREFIX=$deepmd_root -DTENSORFLOW_ROOT=/sw/summit/open-ce/anaconda-base/envs/open-ce-1.2.0-py38-0/lib/python3.8/site-packages/tensorflow/ ..
make -j 8
make install
make lammps
```

Copy USER-DEEPMD to the scr folder in LAMMPS. Go to the scr folder in Lammps and execute the following:

```
git checkout stable_23Jun2022
make yes-kspace yes-molecule yes-user-deepmd
make -j 8 mpi
```
