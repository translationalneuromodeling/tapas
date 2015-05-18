# aponteeduardo@gmail.com
# copyright (c) 2015

AC_INIT(mpdcm, version-0.1)
AC_PREREQ(2.5)

AM_INIT_AUTOMAKE(1.1)
AC_CONFIG_MACRO_DIR(m4)

AC_PROG_CC

AC_PROG_AWK

AX_CHECK_CUDA

AC_CHECK_HEADERS(matrix.h)
AC_CHECK_HEADERS(mex.h)
AC_CHECK_HEADERS(math.h)
AC_CHECK_HEADERS(curand_kernel.h)
AC_CHECK_HEADERS(curand.h)
AC_CHECK_HEADERS(math.h)


matlabhome=$(matlab -nodisplay -r \ 
  "fprintf(1, matlabroot); fprintf(1, '\n'); exit;"| tail -n 1 | \
  sed -e 's/^[ \t]*//' -e 's/[ \t]*$//')
AC_SUBST(matlabhome)

AC_OUTPUT(Makefile src/Makefile)

