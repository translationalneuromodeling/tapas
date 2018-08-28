/* aponteeduardo@gmail.com */
/* copyright (C) 2017 */

#include "antisaccades.h"

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    ANTIS_INPUT svals;
    double *llh;
    SERI_MODEL model;
    int i;
    int nd = 0;
    int *np;
    int na = 1;

    svals.t = mxGetPr(prhs[0]);
    svals.a = mxGetPr(prhs[1]);
    svals.u = mxGetPr(prhs[2]);
    svals.theta = mxGetPr(prhs[3]);

    svals.nt = *mxGetDimensions(prhs[0]);

    nd = mxGetNumberOfDimensions(prhs[3]);
    np = mxGetDimensions(prhs[3]);

    for (i = 0; i < nd; i++)
    {
        // For a weird reason empty dimensions are set to zero.
        na *= (np[i] == 0)? 1: np[i];
    }

    if (nd == 0 )
    {
         mexErrMsgIdAndTxt("tapas:sem:input", "Empty input."); 
    }

    if ( na % DIM_SERI_THETA != 0)
    {
        mexErrMsgIdAndTxt("tapas:sem:input", "Dimensions are not correct"); 
    }

    svals.np = na / DIM_SERI_THETA;

    plhs[0] = mxCreateDoubleMatrix(svals.nt, 1, mxREAL);
    llh = mxGetPr(plhs[0]);
   
    model.llh = seri_llh_abstract;
    model.fill_parameters = reparametrize_seri_lognorm; 
    gsl_set_error_handler_off();
    seri_model_n_states(svals, model, llh);
    gsl_set_error_handler(NULL);
}
