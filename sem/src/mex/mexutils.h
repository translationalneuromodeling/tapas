/* aponteeduardo@gmail.com */
/* copyright (C) 2018 */

#ifndef MEXUTILS_H
#define MEXUTILS_H

#include <matrix.h>
#include <mex.h>
#include "antisaccades.h"

void
reparametrize_prosa(
        int nlhs,
        mxArray *plhs[], 
        int nrhs,
        const mxArray *prhs[],
        FILL_PARAMETERS_PROSA reparametrize);


void
reparametrize_seria(
        int nlhs,
        mxArray *plhs[], 
        int nrhs,
        const mxArray *prhs[],
        FILL_PARAMETERS_SERIA reparametrize);


#endif // MEXUTILS_H
