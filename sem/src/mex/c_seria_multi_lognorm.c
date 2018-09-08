/* aponteeduardo@gmail.com */
/* copyright (C) 2018 */

#include "mexutils.h"

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    wrapper_seria_multi(nlhs, plhs, nrhs, prhs, reparametrize_seria_lognorm);

}