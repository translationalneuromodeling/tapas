//
// Author: Eduardo Aponte
// Copyright 2015 by Eduardo Aponte <aponteeduardo@gmail.com>
//
// Licensed under GNU General Public License 3.0 or later.
// Some rights reserved. See COPYING, AUTHORS.
//
// Revision log:
//

#include "matrix.h"
#include "mex.h"
#include "mpdcm.hcu"
#include "libs/FmriContainer.hpp"
#include "libs/FmriStandardRK4Bilinear.hpp"
#include "libs/utilities.hpp"
#include "libs/MatContainer.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    if (nlhs > 1)
    {
        mexErrMsgIdAndTxt("tapas:mpdcm:fmri:rk4:asynchronic:output",
            "Only one output supported");        
    }

    if (nrhs != 3)
    {
        mexErrMsgIdAndTxt("tapas:mpdcm:fmri:rk4:asynchronic:input",
            "Only three arguments supported");        
    }

    if ( nlhs == 0 )
        return;
        
    auto container_fmri = new fmri::FmriContainer< 
        MatContainer < DataArray, MPFLOAT >, // y
        MatContainer < DataArray, MPFLOAT >, // u
        MatContainer < ThetaFmriArray, ThetaFmri >, // theta
        MatContainer < DataArray, MPFLOAT >, // ackt
        SparseContainer, // B
        SparseContainer // D
        >;

    utils::run_asynchronous_kernel<ThetaFmriArray, PThetaFmri, 
        Host::FmriStandardRK4Bilinear>(prhs[0], prhs[1], prhs[2], 
                *container_fmri);
    uint64_t *out; 
    *plhs = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    out = (uint64_t *) mxGetData(*plhs);
    *out = reinterpret_cast< uint64_t >(container_fmri);

}

