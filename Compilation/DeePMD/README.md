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
git clone https://github.com/deepmodeling/deepmd-kit.git deepmd-kit -b r0.12
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
