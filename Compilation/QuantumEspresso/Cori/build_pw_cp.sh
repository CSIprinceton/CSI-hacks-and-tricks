#!/bin/bash
#Note: this script was slightly adapted from a build script available at NERSC

day=`date +%Y%m%d`
suffix=${day}-v2

export CRAYPE_LINK_TYPE=static

SOFTWAREPATH=/usr/common/software

#hdf5
module load cray-hdf5-parallel/1.10.5.2
module unload cray-libsci

espresso_version="6.6"
version=scalapack
arch=knl

echo "Please provide the path to libxc root folder (this folder should contain lib and include folders):"
read libxc_path
libxc_path=`readlink -f ${libxc_path}`

if [ "${arch}" == "knl" ]; then
#            ARCHFLAGS=" -xMIC-AVX512"  # do not set this otherwise KNL cross compile will fail
        if [ "${CRAY_CPU_TARGET}" == "haswell" ]; then
          echo "Current CRAY_CPU_TARGET=${CRAY_CPU_TARGET}, need to swap to knl"
          module swap craype-haswell craype-mic-knl
        fi
elif [ "${arch}" == "hsw" ]; then
#            ARCHFLAGS=" -xCORE-AVX2"
        if [ "${CRAY_CPU_TARGET}" == "knl" ]; then
          echo "Current CRAY_CPU_TARGET=${CRAY_CPU_TARGET}, need to swap to haswell"
          module swap craype-mic-knl craype-haswell
        fi
      else
          ARCHFLAGS=" -xCORE-AVX-I"
      fi

      out=build-${arch}-out${suffix}
      date > ${out}
      module list &>> ${out}
      echo >> ${out}

#scalapack linker flags, cannot be found automatically
export scalapackflags="${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.a -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_lp64.a -Wl,--end-group -liomp5 -lpthread -lm -ldl"

#clean everything up so that no libraries with wrong arch are around
echo "make veryclean" >> ${out}
make veryclean
export FC=ifort
export F90=ifort
export MPIF90=ftn
#	export FCFLAGS="-mkl -O3 -qopenmp"
export FCFLAGS="-mkl -O3 -qopenmp${ARCHFLAGS}"
export CC=icc
#	export CFLAGS="-mkl -O3 -qopenmp"
export CFLAGS="-mkl -O3 -qopenmp${ARCHFLAGS}"
export LDFLAGS="${FCFLAGS}"
export F90FLAGS="${FCFLAGS}"
export MPIF90FLAGS="${F90FLAGS}"

#configure
echo "./configure \
            --enable-openmp \
            --enable-parallel \
            --disable-shared \
            --with-scalapack=intel \
            --with-libxc=yes \
            --with-libxc-prefix=${libxc_path} \
            --with-hdf5=${HDF5_DIR}" &>> ${out}
./configure \
            --enable-openmp \
            --enable-parallel \
            --disable-shared \
            --with-scalapack=intel \
            --with-libxc=yes \
            --with-libxc-prefix=${libxc_path} \
	    --with-hdf5=${HDF5_DIR} | tee ${out}
date >> ${out}


#some makefile hacking
cp -p make.inc make.inc-prev
sed -i "s|^HDF5_LIB =|HDF5_LIB = -L${HDF5_DIR}/lib -lhdf5|g" make.inc
sed -i "s|^F90FLAGS.*=|F90FLAGS = ${ARCHFLAGS}|g" make.inc
sed -i "s|^FFLAGS.*=|FFLAGS = ${ARCHFLAGS}|g" make.inc
sed -i "s|^CFLAGS.*=|CFLAGS = ${ARCHFLAGS}|g" make.inc
sed -i "s|^MANUAL_DFLAGS  =|MANUAL_DFLAGS  = -D__SCALAPACK|g" make.inc
sed -i "s|^SCALAPACK_LIBS =|SCALAPACK_LIBS = ${scalapackflags}|g" make.inc

#clean up
make clean

#build crap
j=16

date >> ${out}
       echo "make -j ${j} pw cp" &>> ${out}
       make -j ${j} pw cp | tee ${out}
date >> ${out}
