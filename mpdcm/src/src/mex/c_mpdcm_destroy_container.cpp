//
// Author: Eduardo Aponte
// Copyright 2015 by Eduardo Aponte <aponteeduardo@gmail.com>
//
// Licensed under GNU General Public License 3.0 or later.
// Some rights reserved. See COPYING, AUTHORS.
//
// Revision log:
//

#include <stdint.h>
#include "matrix.h"
#include "mex.h"
#include "mpdcm.hcu"
#include "libs/utilities.hpp"
#include "libs/FmriContainer.hpp"
#include "libs/MatContainer.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    if (nlhs > 1)
    {
        mexErrMsgIdAndTxt("tapas:mpdcm:collect_results:output",
            "Only one output supported");        
    }

    if (nrhs != 1)
    {
        mexErrMsgIdAndTxt("tapas:mpdcm:collect_results:input",
            "Only one argument supported");        
    }


    if ( nlhs == 0 )
        return;

    auto fmri_container = reinterpret_cast<
        fmri::FmriContainer<
            MatContainer < DataArray, MPFLOAT >, // y
            MatContainer < DataArray, MPFLOAT >, // u
            MatContainer < ThetaFmriArray, ThetaFmri >, // theta
            MatContainer < DataArray, MPFLOAT >, // ackt
            SparseContainer, // B
            SparseContainer > * >(*((uint64_t *) mxGetData(*prhs)));

    delete fmri_container;

    // Return zero status. 
    *plhs = mxCreateDoubleMatrix(1, 1, mxREAL);
    double *out = mxGetPr(*plhs);
    *out = 0;
    return;
}

