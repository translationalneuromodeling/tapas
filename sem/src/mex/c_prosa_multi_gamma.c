/* aponteeduardo@gmail.com */
/* copyright (C) 2017 */

#include "antisaccades.h"
#include "mexutils.h"
#ifdef HAVE_OMP_H
#include <omp.h>
#endif

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    wrapper_prosa_multi(nlhs, plhs, nrhs, prhs, reparametrize_prosa_gamma);

} 
