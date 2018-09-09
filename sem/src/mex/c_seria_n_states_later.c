/* aponteeduardo@gmail.com */
/* copyright (C) 2018 */

#include "mexutils.h"

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    wrapper_seria_n_states(nlhs, plhs, nrhs, prhs, reparametrize_seria_later);

}