# README

aponteeduardo@gmail.com

## Introduction

mpdcm is a toolbox for efficient integration of the dynamical system defined
by DCM (see citations) on GPU and multicore architectures. Besides the core
 code, several 
routines for statistical inference are defined as part of our toolbox, 
including a map estimator, a mle estimator, variational Bayes hierarchical 
inference, thermodynamic integration, and population MCMC.


## Compilation

Currently only LINUX is supported for compilation. We have compiled in a few
different platforms successfully. Most likely it will compile in MAC and 
Windows but we haven't yet tested those environments. 

Cuda version supported: 4.2 onward (testest until CUDA 9.0)

For compilation you require a matlab running environment, a cuda installation
and the supported compiler. If your standard compiler is not compatible with
your cuda installation, then, although it''s possible to compile, it might
require some manual work. For compilation you should use

 cd src
 ./configure
 make

The cuda installation is now optional. To activate it use:

./configure --enable-cuda

If your cuda installation happens to be in a non-standard location, or you
would like to use a version of cuda installed in your computer but not the 
standard one use

 ./configure --enable-cuda --with-cuda=<LOCATION OF YOUR CUDA INSTALLATION>

For example

 ./configure --enable-cuda --with-cuda=/cluster/apps/cuda/4.2.9/

mpdcm compiles by default in single precision float numbers. Double precision
compilation is possible, but it has an effect on performance.

### Select compiler

To select a compiler use

CC='gcc-4.8' ./configure

### Point to the armadillo library

Armadillo is a template library so you don't need to install it in the proper
sense. The installer will find it if you export it into you path. You will
need to export it into the path

export CXXFLAGS="-I/cluster/home/username/software/armadillo/include $CXXFLAGS"
   

## Problems in Linux

### libgfortran: version GFORTRAN\_1.4 not found

This error is cause by bad linking of the gfortran libraries. To solve it 
locate the library

locate libfortran

As example output

/usr/lib/x86_64-linux-gnu/libgfortran.so.3
/usr/lib/x86_64-linux-gnu/libgfortran.so.3.0.0
/usr/local/MATLAB/R2014a/sys/os/glnxa64/libgfortran.so.3
/usr/local/MATLAB/R2014a/sys/os/glnxa64/libgfortran.so.3.0.0

Then update matlabs softlink.

sudo ln -sf /usr/lib/x86_64-linux-gnu/libgfortran.so.3.0.0 /usr/local/MATLAB/R2014a/sys/os/glnxa64/libgfortran.so.3

## Compilation in Windows

The purely CPU part of mpdcm can be compiled in Windows. For that you will 
require an installation of armadillo and a version of visual studio that is
compatible with you matlab version.

### Installing armadillo

Installing armadillo consist of the following steps:

Download it.
Unzip it.
Move the include files to your preferred location.

Currently, we do not support linking against blas or lapack, although this 
should be possible if those libraries are linked dynamically or whether they
are available in you path. If you want to link agains these libraries skip the
next step, but you will have to figure out how to link them on your own.

## Proving compiler options.

If a file called tapas_mex_options.xml is found in the directory

    path_to_tapas\tapas\mpdcm\src 

it will be used as the options file for the compiler. Notice that if you add
the compiler option \openmp the whole package will support multithreading.


# Change log:

v2.0.0.1: Fixed compilation issue in ./src/src/mpdcm.hcu.
