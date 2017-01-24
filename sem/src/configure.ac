# aponteeduardo@gmail.com
# copyright (c) 2015

AC_INIT(sooner, version-0.1)
AC_PREREQ(2.5)

AM_INIT_AUTOMAKE(1.1)
AC_CONFIG_MACRO_DIR(m4)

AC_PROG_CC

AC_CHECK_HEADERS(matrix.h)
AC_CHECK_HEADERS(mex.h)
AC_CHECK_HEADERS(math.h)

AC_CHECK_HEADERS(gsl/gsl_sf_gamma.h)
AC_CHECK_HEADERS(gsl/gsl_sys.h)

AC_OPENMP

LDFLAGS="-L/opt/local/lib/ ${LDFLAGS}"

AC_CHECK_LIB(gomp, omp_get_thread_num)

LDFLAGS="-L/opt/local/lib/ ${LDFLAGS}"

# ME OF THE FUTURE: This looks redundant but needs to be done like that.

AC_CHECK_LIB([m],[cos])
AC_CHECK_LIB([gslcblas],[cblas_dgemm])
AC_CHECK_LIB([gsl],[gsl_blas_dgemm])

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
