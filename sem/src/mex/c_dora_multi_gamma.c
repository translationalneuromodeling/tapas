

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
    int i, j;
    int ns = mxGetDimensions(prhs[1])[0];
    int nc = mxGetDimensions(prhs[1])[1];

    if ( nlhs != 1 )
        HANDLE_CERROR_MSG( 1 , "Wrong left size assignments.");

    if ( nrhs != 2 )
        HANDLE_CERROR_MSG( 1 , "Wrong right size assignments." );

    plhs[0] = mxCreateDoubleMatrix(ns, nc, mxREAL);
    llh = mxGetPr(plhs[0]);

    model.llh = dora_llh_gamma;
    model.nested_integral = ngamma_gslint;
    model.fill_parameters = reparametrize_dora_gamma; 
    gsl_set_error_handler_off();

    #pragma omp parallel for private(i) private(j) collapse(2) schedule(static) 
    for (j = 0; j < nc; j++) 
    {
        for (i = 0; i < ns; i++)
        {
            
            ANTIS_INPUT svals;
            mxArray *y = mxGetField(prhs[0], i, "y");
            mxArray *u = mxGetField(prhs[0], i, "u");
            mxArray *theta = mxGetCell(prhs[1], i + ns * j);
            double *tllh;
            int k;

            
            svals.t = mxGetPr(mxGetField(y, 0, "t"));
            svals.a = mxGetPr(mxGetField(y, 0, "a"));
            svals.u = mxGetPr(mxGetField(u, 0, "tt"));
            
            svals.theta = mxGetPr(theta);
            
            svals.nt = *mxGetDimensions(mxGetField(y, 0, "t")); 
            svals.np = mxGetDimensions(theta)[1];
            
            tllh = (double *) malloc(svals.nt * sizeof(double));
            
            dora_model_two_states_optimized(svals, model, tllh);

            llh[i + ns * j] = 0;
            for (k = 0; k < svals.nt; k++)
            {
                llh[i + ns * j] += tllh[k];
            }
            if ( abs(llh[i + ns * j]) == INFINITY || isnan(llh[i + ns * j]))
                llh[i + ns * j] = -INFINITY;
            free(tllh);
            
        }
    }
   gsl_set_error_handler(NULL); 
} 
