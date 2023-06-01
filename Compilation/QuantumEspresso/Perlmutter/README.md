## Minimal script to compile QE 6.4.1 with libxc on Perlmutter

```
module load PrgEnv-intel

tar -xf qe-6.4.1.tar.gz
tar -xf libxc-4.3.4.tar.gz
mkdir libxc-install
working_dir=`pwd`

cd libxc-4.3.4
./configure --prefix=${working_dir}/libxc-install CC=cc FC=ftn
make -j 8
make install
cd ../

cd q-e-qe-6.4.1
export scalapackflags="${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.a -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_lp64.a -Wl,--end-group -liomp5 -lpthread -lm -ldl"
./configure --with-libxc --with-libxc-prefix=${working_dir}/libxc-install --with-scalapack=intel --enable-openmp CC=cc MPIF90=ftn F90=ftn 
sed -i "s|^MANUAL_DFLAGS  =|MANUAL_DFLAGS  = -D__SCALAPACK|g" make.inc
sed -i "s|^SCALAPACK_LIBS =|SCALAPACK_LIBS = ${scalapackflags}|g" make.inc
make -j8 pw
```            
