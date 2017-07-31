# aponteeduardo@gmail.com
# copyright (c) 2015

AC_INIT(sem, version-1.0)
AC_PREREQ(2.5)

AM_INIT_AUTOMAKE(1.1)
AC_CONFIG_MACRO_DIR(m4)

AC_PROG_CC

AC_CONFIG_SRCDIR(antisaccades/antisaccades.h)
AC_CONFIG_HEADERS([config.h])

AC_CHECK_HEADERS(matrix.h)
AC_CHECK_HEADERS(mex.h)
AC_CHECK_HEADERS(math.h)

AC_CHECK_HEADERS(gsl/gsl_sf_gamma.h)
AC_CHECK_HEADERS(gsl/gsl_sys.h)

AC_OPENMP
# In mac omp is compiled through iomp5. TODO If the compiler is found but the 
# library is not it will fail.
AC_CHECK_LIB(gomp, omp_get_thread_num, , \
    AC_CHECK_LIB(iomp5, omp_get_thread_num))

AC_CHECK_HEADERS(omp.h)

# ME OF THE FUTURE: This looks redundant but needs to be done like that.
AC_CHECK_LIB([m],[cos])
AC_CHECK_LIB([gslcblas],[cblas_dgemm], , \
    [ AC_MSG_ERROR([ unable to find gsl ]) ])
AC_CHECK_LIB([gsl],[gsl_blas_dgemm], , \
    [ AC_MSG_ERROR([ unable to find gsl ]) ])

# Find matlab
matlabhome=$(matlab -nodisplay -nosplash -r "fprintf(1, matlabroot); fprintf(1, '\n'); exit;"| tail -n 1)

# Get the platform
mexext=$($matlabhome/bin/mexext)

AC_SUBST(matlabhome)
AC_SUBST(major)
AC_SUBST(minor)
AC_SUBST(mexext)

AC_DEFINE([TAPAS_MATLAB], [], ["Compile for matlab"])

AC_OUTPUT(Makefile 
    antisaccades/Makefile 
    mex/Makefile
    test/Makefile)
