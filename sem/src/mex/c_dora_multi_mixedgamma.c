/* aponteeduardo@gmail.com */
/* copyright (C) 2017 */

#include "antisaccades.h"
#ifdef HAVE_OMP_H
#include <omp.h>
#endif

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    double *llh;
    DORA_MODEL model;
    int k;
    int ns = mxGetDimensions(prhs[1])[0];
    int nc = mxGetDimensions(prhs[1])[1];

    if ( nlhs != 1 )
        HANDLE_CERROR_MSG( 1 , "Wrong left size assignments.");

    if ( nrhs != 2 )
        HANDLE_CERROR_MSG( 1 , "Wrong right size assignments." );

    plhs[0] = mxCreateDoubleMatrix(ns, nc, mxREAL);
    llh = mxGetPr(plhs[0]);

    model.llh = dora_llh_abstract;
    model.fill_parameters = reparametrize_dora_mixedgamma; 
    gsl_set_error_handler_off();

    //#pragma omp parallel for private(i) private(j) collapse(2) schedule(dynamic) 
    #pragma omp parallel for private(k) schedule(dynamic)
    for (k = 0; k < nc * ns; k++) 
    {
        int j = k/ns, i = k%ns;
        double illh = 0;
        
        ANTIS_INPUT svals;
        mxArray *y = mxGetField(prhs[0], i, "y");
        mxArray *u = mxGetField(prhs[0], i, "u");
        mxArray *theta = mxGetCell(prhs[1], k);
        double *tllh;
        int l;

        svals.t = mxGetPr(mxGetField(y, 0, "t"));
        svals.a = mxGetPr(mxGetField(y, 0, "a"));
        svals.u = mxGetPr(mxGetField(u, 0, "tt"));
        
        svals.theta = mxGetPr(theta);
        
        svals.nt = *mxGetDimensions(mxGetField(y, 0, "t")); 
        svals.np = (mxGetDimensions(theta)[0]
            * mxGetDimensions(theta)[1])/DIM_DORA_THETA; 
        tllh = (double *) malloc(svals.nt * sizeof(double));
        
        dora_model_n_states_optimized(svals, model, tllh);

        for (l = 0; l < svals.nt; l++)
        {
            illh += tllh[l];
        }

        if ( abs(illh) == INFINITY || isnan(illh))
            illh = -INFINITY;
        
        llh[k] = illh;
        free(tllh);
        
    }
   gsl_set_error_handler(NULL); 
} 
