/* aponteeduardo@gmail.com */
/* copyright (C) 2014 */

#include <mex.h>
#include <matrix.h>
#include <cuda_runtime_api.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int nd;
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

        cudaError_t error = cudaGetDeviceCount(&nd);
        if (cudaSuccess != error || nd < 0)
            nd = 0; 
        
        *plhs = mxCreateDoubleMatrix(1, 1, mxREAL);
        out = mxGetPr(*plhs);
        *out = nd;
    }

}
