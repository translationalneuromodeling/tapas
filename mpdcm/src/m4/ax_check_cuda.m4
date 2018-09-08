##### 
#
# SYNOPSIS
#
# AX_CHECK_CUDA
#
# DESCRIPTION
#
# Figures out if CUDA Driver API/nvcc is available, i.e. existence of:
# 	cuda.h
#   libcuda.so
#   nvcc
#
# If something isn't found, fails straight away.
#
# Locations of these are included in 
#   CUDA_CFLAGS and 
#   CUDA_LDFLAGS
#   CUDA_LIBS.
# Path to nvcc is included as
#   NVCC_PATH
# in config.h
# 
# The author is personally using CUDA such that the .cu code is generated
# at runtime, so don't expect any automake magic to exist for compile time
# compilation of .cu files.
#
# LICENCE
# Public domain
#
# AUTHOR
# wili
#
##### 

AC_DEFUN([AX_CHECK_CUDA], [

# Provide your CUDA path with this		
AC_ARG_WITH(cuda, [  --with-cuda=PREFIX      Prefix of your CUDA installation], [cuda_prefix=$withval], [cuda_prefix="/usr/local/cuda"])

if $(test ! -d "$cuda_prefix"); then
   	AC_MSG_FAILURE([cuda directory $cuda_prefix does not exist]) 
fi


# Checking for nvcc
AC_CHECK_PROG(NVCC_CHECK, yes, "", "$cuda_prefix/bin")

if test x"$NVCC_CHECK" != x"yes" ; then
    NVCC="$cuda_prefix/bin/nvcc"
else
	AC_MSG_FAILURE([nvcc was not found in $cuda_prefix/bin])
fi


# We need to add the CUDA search directories for header and lib searches

# Saving the current flags
ax_save_CFLAGS="${CFLAGS}"
ax_save_LDFLAGS="${LDFLAGS}"
ax_save_LIBS="${LIBS}"

CUDA_CFLAGS="-I$cuda_prefix/include"
CFLAGS="$CUDA_CFLAGS $CFLAGS"
CUDA_LDFLAGS="-L$cuda_prefix/lib64"
LDFLAGS="$CUDA_LDFLAGS $LDFLAGS"

# Get the version of nvcc 

NVCC_VER=$($NVCC -V | tail -n 1 | sed 's%.*release %%' | sed 's%,.*$%%')

# And the header and the lib
AC_CHECK_HEADER([cuda.h], [], AC_MSG_FAILURE([Couldn't find cuda.h]), [#include <cuda.h>])
AC_CHECK_LIB([cuda], [cuInit], [], AC_MSG_FAILURE([Couldn't find libcuda]))

# Reck for the runtime

AC_CHECK_HEADER([cuda_runtime_api.h], [], AC_MSG_FAILURE([Couldn't find cuda_runtime_api.h]), [#include <cuda_runtime_api.h>])
AC_CHECK_LIB([cudart], [cudaDeviceSynchronize], [], AC_MSG_FAILURE([Couldn't find libcudart]))

CUDA_LIBS="${LIBS}"

# Announcing the new variables
AC_SUBST([CUDA_CFLAGS])
AC_SUBST([CUDA_LDFLAGS])
AC_SUBST([CUDA_LIBS])
AC_SUBST([NVCC])


# Returning to the original flags
CFLAGS=${ax_save_CFLAGS}
LDFLAGS=${ax_save_LDFLAGS}
LIBS=${ax_save_LIBS}
])
