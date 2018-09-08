/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */

/* Returns a 0 if the compilation flag is single and 1 if it's double */


#include "mpdcm.hcu"
#include <mex.h>
#include <matrix.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    double *out;
    double nd = 0;

    if (nlhs > 1)
    {
        mexErrMsgIdAndTxt("tapas:mpdcm:compile_flag",
        "Only one output supported");
    }

    if (nrhs != 0)
    {
        mexErrMsgIdAndTxt("tapas:mpdcm:compile_flag",
        "Only 0 arguments supported");
    }


#ifdef MPDOUBLEFLAG
    nd = 1;
#endif 

    if ( nlhs == 1 )
    {

        *plhs = mxCreateDoubleMatrix(1, 1, mxREAL);
        out = mxGetPr(*plhs);
        *out = nd;
    }


}
