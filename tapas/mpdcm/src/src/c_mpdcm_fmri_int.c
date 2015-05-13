/* aponteeduardo@gmail.com */
//
// Author: Eduardo Aponte
//
// Revision log:
//

#include "c_mpdcm.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    if (nlhs > 1)
    {
        mexErrMsgIdAndTxt("CUDA_DCM:cuda_check_theta:input",
            "Only one output supported");
    }

    if (nrhs != 3)
    {
        mexErrMsgIdAndTxt("CUDA_DCM:cuda_check_theta:input",
            "Only three arguments supported");        
    }


    if ( nlhs == 1 )
    {
        int nx, ny, nu, dp, nt, nb;
 
        c_mpdcm_fmri_euler(plhs, prhs, prhs+1, prhs+2, 
            &nx, &nu, &ny, &dp, &nt, &nb); 
    }

}
 
