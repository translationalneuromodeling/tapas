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
    int nd;
    double *input;
    cudaError_t error;

    if (nlhs > 1)
    {
        mexErrMsgIdAndTxt("mpdcm:num_devices:output",
            "Only one output supported");
    }

    if (nrhs != 1)
    {
        mexErrMsgIdAndTxt("mpdcm:num_devices:input",
            "Only one argument supported");
    }

    input = mxGetPr(*prhs);
    error =  cudaSetDevice((int ) *input);

    if ( cudaSuccess == error )
    {
        mexErrMsgIdAndTxt("mpdcm:set_device:cuda",
            "Error while setting device");
    }
 
    if ( nlhs == 1 )
    {   
       double *out; 
        *plhs = mxCreateDoubleMatrix(1, 1, mxREAL);
        out = mxGetPr(*plhs);
        *out = 0;
    }

}
