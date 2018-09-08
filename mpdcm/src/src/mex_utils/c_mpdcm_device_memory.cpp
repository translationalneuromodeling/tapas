//
// Author: Eduardo Aponte
// Copyright 2015 by Eduardo Aponte <aponteeduardo@gmail.com>
//
// Licensed under GNU General Public License 3.0 or later.
// Some rights reserved. See COPYING, AUTHORS.
//
// Revision log:
//

#include <mex.h>
#include <matrix.h>
#include <cuda_runtime_api.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int nd = 1;
    size_t mfree, mtotal;
    double *out;
    mxArray *ty;

    if (nlhs > 1)
    {
        mexErrMsgIdAndTxt("mpdcm:num_devices:output",
            "Only one output supported");
    }

    if (nrhs != 0)
    {
        mexErrMsgIdAndTxt("mpdcm:num_devices:input",
            "Only zero arguments supported");
    }


    if ( nlhs == 1 )
    {

        cudaError_t error = cudaMemGetInfo(&mfree, &mtotal);
        if (cudaSuccess != error)
            nd = 0; 
        
        *plhs = mxCreateDoubleMatrix(2, 1, mxREAL);
        out = mxGetPr(*plhs);
        out[0] = (double ) mfree;
        out[1] = (double ) mtotal;
    }

}
