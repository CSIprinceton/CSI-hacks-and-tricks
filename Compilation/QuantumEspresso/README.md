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

### Scritp for SUMMIT 

```
module load pgi/20.4 hdf5 essl netlib-lapack cuda spectrum-mpi/10.3.1.2-20200121

#Using QE commit number: 4dec0ccc59bb1cea8e9e146347c7e713d13ac8d7
git clone https://gitlab.com/QEF/q-e.git
cd q-e
git clone https://gitlab.com/libxc/libxc.git
cd libxc
git checkout 4.2.3 #I could not make libxc versions 5.x.x work

#GNU autotools
libtoolize
aclocal -I m4\
  && autoheader \
  && automake --add-missing \
  && autoconf \
  && autoreconf -i

mkdir ../libxc_lib
./configure --prefix=`realpath ../libxc_lib`
make && make install

cd ..

export BLAS_LIBS="-L$OLCF_ESSL_ROOT/lib64 -lessl"
export LAPACK_LIBS="-L$OLCF_ESSL_ROOT/lib64 -lessl $OLCF_NETLIB_LAPACK_ROOT/lib64/liblapack.a"

./configure --enable-openmp --with-hdf5=$OLCF_HDF5_ROOT \
            --with-cuda=$OLCF_CUDA_ROOT --with-cuda-runtime=10.1 --with-cuda-cc=70 \
            --with-libxc=yes --with-libxc-prefix=`realpath ./libxc_lib` --with-libxc-include=`realpath ./libxc_lib/include`

sed -i "/DFLAGS/s/__FFTW/__LINUX_ESSL/" make.inc
sed -i "/CFLAGS/s/= /= -c11 /" make.inc

make -j 8 pw

```
