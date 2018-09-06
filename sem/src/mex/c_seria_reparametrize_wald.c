/* aponteeduardo@gmail.com */
/* copyright (C) 2018 */

#include "antisaccades.h"
#include "mexutils.h"

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    reparametrize_seria(nlhs, plhs, nrhs, prhs, reparametrize_seria_wald);

}