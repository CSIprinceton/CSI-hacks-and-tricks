# Compilation of Quantum Espresso with libxc (for SCAN)

## Script for TigerCPU

This script is based on Zach's compilation:

```
module purge
module load intel/18.0/64/18.0.3.222 intel-mpi/intel/2018.3/64

mkdir QuantumEspresso
cd QuantumEspresso
qedir=$(pwd)

# install libxc 4.3.4
mkdir libxc-install # to install later
wget http://www.tddft.org/programs/libxc/down.php?file=4.3.4/libxc-4.3.4.tar.gz
tar -xf down.php?file=4.3.4%2Flibxc-4.3.4.tar.gz
cd libxc-4.3.4
./configure --prefix=${qedir}/libxc-install CC=icc FC=ifort

# install quantum espresso 6.4.1 - 6.5 doesn't seen to work with these instructions
cd ../
wget https://github.com/QEF/q-e/archive/qe-6.4.1.tar.gz
tar -xf qe-6.4.1.tar.gz
cd q-e-qe-6.4.1

./configure --with-libxc --with-libxc-include=${qedir}/libxc-install/include/ --with-libxc-prefix=${qedir}/libxc-install/
make -j 10 all

```

## Script for Della

This script is very similar to the one of TigerCPU

```
module purge
module load intel intel-mpi intel-mkl

mkdir QuantumEspresso
cd QuantumEspresso
qedir=$(pwd)

# install libxc 4.3.4
mkdir libxc-install # to install later
wget http://www.tddft.org/programs/libxc/down.php?file=4.3.4/libxc-4.3.4.tar.gz
tar -xf down.php?file=4.3.4%2Flibxc-4.3.4.tar.gz
cd libxc-4.3.4
./configure --prefix=${qedir}/libxc-install CC=icc FC=ifort

# install quantum espresso 6.4.1 - 6.5 doesn't seen to work with these instructions
cd ../
wget https://github.com/QEF/q-e/archive/qe-6.4.1.tar.gz
tar -xf qe-6.4.1.tar.gz
cd q-e-qe-6.4.1

./configure --with-scalapack=intelmpi --enable-openmp --with-libxc --with-libxc-include=${qedir}/libxc-install/include/ --with-libxc-prefix=${qedir}/libxc-install/
make -j 10 all

```
