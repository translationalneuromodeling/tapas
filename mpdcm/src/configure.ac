# aponteeduardo@gmail.com
# copyright (c) 2015

AC_INIT(mpdcm, version-2.0)
AC_PREREQ(2.5)

AM_INIT_AUTOMAKE(1.1)
AC_CONFIG_MACRO_DIR(m4)

#AC_PROG_CC
AC_PROG_CXX

AC_CONFIG_SRCDIR(src/cuda/fmri_dynamics.cu)
AC_CONFIG_HEADERS([config.h])

AC_OPENMP
# In mac omp is compiled through iomp5. TODO If the compiler is found but the 
# library is not it will fail.
AC_CHECK_LIB(gomp, omp_get_thread_num, , \
    AC_CHECK_LIB(iomp5, omp_get_thread_num))
AC_CHECK_HEADERS([omp.h])

AC_CHECK_LIB(gslcblas,main)
AC_CHECK_LIB(gsl,main)

AC_LANG_PUSH([C++])
AC_CHECK_HEADERS([armadillo],[], [AC_MSG_ERROR("Please install armadillo")])
AC_LANG_POP([C++])

AC_ARG_ENABLE([cuda], 
    AS_HELP_STRING([--enable-cuda], [Compile with cuda]),
    [use_cuda=yes],
    [use_cuda=no])

echo "use cuda $use_cuda"

AS_CASE([$use_cuda],
    [yes], [AX_CHECK_CUDA],
    [no], [],
    [AC_MSG_ERROR("Invalid enable-cuda option")])

AM_CONDITIONAL([USE_CUDA], [test "$use_cuda" == yes])

AC_CHECK_LIB([openblas], [cblas_dgemm], ,
  [ AC_MSG_ERROR([unable to find blas ])
])

AC_DEFINE([ARMA_DONT_USE_WRAPPER], [], ["Don't use the arma runtimes"])
AC_DEFINE([ARMA_BLAS_LONG_LONG], [], ["Use blas long format"])

# Find the matlab compiler

matlabhome=$(matlab -nodisplay -r "fprintf(1, matlabroot); fprintf(1, '\n'); exit;"| tail -n 1)

AC_CANONICAL_HOST

mexext=$($matlabhome/bin/mexext)

AC_SUBST(matlabhome)
AC_SUBST(major)
AC_SUBST(minor)
AC_SUBST(mexext)

AC_OUTPUT(Makefile 
    src/Makefile 
    src/libs/Makefile 
    src/cuda/Makefile 
    src/erp/Makefile  
    src/host/Makefile
    src/test/Makefile 
    src/mex/Makefile
    src/mex_erp/Makefile
    src/mex_utils/Makefile
    src/mex_host/Makefile)
