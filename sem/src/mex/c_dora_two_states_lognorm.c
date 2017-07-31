/* aponteeduardo@gmail.com */
/* copyright (C) 2017 */

#include "antisaccades.h"

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    ANTIS_INPUT svals;
    double *llh;
    DORA_MODEL model;

    svals.t = mxGetPr(prhs[0]);
    svals.a = mxGetPr(prhs[1]);
    svals.u = mxGetPr(prhs[2]);
    svals.theta = mxGetPr(prhs[3]);

    svals.nt = *mxGetDimensions(prhs[0]); 
    svals.np = mxGetDimensions(prhs[3])[1];
    
    plhs[0] = mxCreateDoubleMatrix(svals.nt, 1, mxREAL);
    llh = mxGetPr(plhs[0]);
    
    model.llh = dora_llh_lognorm;
    model.fill_parameters = populate_parameters_dora; 
    dora_model_two_states(svals, model, llh);
}