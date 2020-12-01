# Steps to compile QE (with LIBXC) on Cori KNL

## 1) Download and compile LIBXC

```
git clone https://gitlab.com/libxc/libxc.git
git checkout 4.0.0
./autoconf.sh #We are providing this code here 
./configure --prefix=/path/to/libxc/
make -j 16
make install
```

## 2) Download and compile QE
```
git clone https://gitlab.com/QEF/q-e.git 
git checkout qe-6.6 #Only version 6.6 was tested here.
#Now you should execute the build code provided in this repo.
./build_pw_cp.sh
#The code will ask for the libxc path (/path/to/libxc/)
```

## 3) Please see an example of a submission script (for Cori KNL) in the file sub.cmd 
