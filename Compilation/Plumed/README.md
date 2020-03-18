# Compilation of Plumed and Lammps

## Script for TigerCPU

```
# make sure you are on tigercpu.princeton.edu

module purge
module load intel intel-mpi

cd /home/username
mkdir Programs; cd Programs

# Build plumed

git clone https://github.com/plumed/plumed2 plumed2
cd plumed2
./configure --prefix=/home/username/Programs/plumed-install --enable-modules=all CXX=mpiicpc CXXFLAGS="-Ofast -mtune=skylake-avx512" cross_compiling=yes
make -j 10
make install

export PKG_CONFIG_PATH="$PKG_CONFIG_PATH:/home/username/Programs/Plumed/plumed-install/lib/pkgconfig"
cd ..

# Build lammps

git clone https://github.com/lammps/lammps lammps
cd lammps
mkdir build; cd build

# copy and paste the next 4 lines into the terminal
cmake3 -D CMAKE_INSTALL_PREFIX=$HOME/.local -D LAMMPS_MACHINE=tigerCpu -D ENABLE_TESTING=yes \
-D BUILD_MPI=yes -D BUILD_OMP=yes -D CMAKE_CXX_COMPILER=icpc -D CMAKE_BUILD_TYPE=Release \
-D CMAKE_CXX_FLAGS_RELEASE="-Ofast -mtune=skylake-avx512 -DNDEBUG" -D PKG_USER-OMP=yes \
-D PKG_KSPACE=yes -D PKG_RIGID=yes -D PKG_MANYBODY=yes -D PKG_USER-PLUMED=yes \
-D DOWNLOAD_PLUMED=no -D PLUMED_MODE=static \
-D PKG_MOLECULE=yes -D PKG_USER-INTEL=yes -D INTEL_ARCH=cpu -D INTEL_LRT_MODE=threads ../cmake

make -j 10


```

