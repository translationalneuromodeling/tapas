/* aponteeduardo@gmail.com */
/* copyright (C) 2018 */

#include "mexutils.h"
#include <stdio.h>

void
verify_input_theta_array(mxArray *theta, int dims_theta)
{
    int i;
    int na = 1;

    if ( !mxIsCell(theta) )
    {
          mexErrMsgIdAndTxt("tapas:sem:input", 
                  "Input parameters should be a cell array");   
    }

    mwSize nd = mxGetNumberOfDimensions(theta);
    mwSize *np = mxGetDimensions(theta);

    for (i = 0; i < nd; i++)
    {
        // For a weird reason empty dimensions are set to zero.
        na *= (np[i] == 0)? 1: np[i];
    }

    if ( nd == 0 )
    {
         mexErrMsgIdAndTxt("tapas:sem:input", "Empty input."); 
    }

    for (i = 0; i < na; i++)
    {

        mxArray *i_params = mxGetCell(theta, i);

        mwSize stheta = mxGetNumberOfDimensions(i_params);
        mwSize *dtheta = mxGetDimensions(i_params);

        if ( stheta == 0 || stheta > 2)
        {
            mexErrMsgIdAndTxt("tapas:sem:input",
                    "Dimensions of the parameters is not adequate."); 
        }

        if ( stheta == 2 )
        {
            if ( dtheta[1] > 1 )
            {
                mexErrMsgIdAndTxt("tapas:sem:input",
                    "Dimensions of the parameters is not adequate.");       
            }
        }

        if ( dims_theta * (dtheta[0] / dims_theta) != dtheta[0] )
        {
            mexErrMsgIdAndTxt("tapas:sem:input",
                "Dimensions of the parameters is not adequate.");       
        }
    
    }  
    
}

void
reparametrize_prosa(
        int nlhs,
        mxArray *plhs[], 
        int nrhs,
        const mxArray *prhs[],
        FILL_PARAMETERS_PROSA reparametrize)
{

    int i;
    int na = 1;

    if ( nrhs != 1 )
    {
         mexErrMsgIdAndTxt("tapas:sem:input", "Invalid number of inputs."); 
    }

    if ( nlhs != 1 )
    {
         mexErrMsgIdAndTxt("tapas:sem:output", "Invalid output."); 
    }

    // Verify input
    
    verify_input_theta_array(prhs[0], DIM_PROSA_THETA);

    mwSize nd = mxGetNumberOfDimensions(prhs[0]);
    mwSize *np = mxGetDimensions(prhs[0]);

    for (i = 0; i < nd; i++)
    {
        // For a weird reason empty dimensions are set to zero.
        na *= ( np[i] == 0 )? 1 : np[i];
    }

    // Output will have the same structure
    plhs[0] = mxCreateCellArray(nd, np);
   
    gsl_set_error_handler_off();
    for (i = 0; i < na; i++)
    {
        int j;

        mxArray *i_params = mxGetCell(prhs[0], i);
        double *d_i_params = mxGetPr(i_params);
        
        mwSize stheta = mxGetNumberOfDimensions(i_params);
        mwSize *dtheta = mxGetDimensions(i_params);

        mxArray *o_params = mxCreateDoubleMatrix(dtheta[0], 1, mxREAL); 
        double *d_o_params = mxGetPr(o_params);

        for (j = 0; j < dtheta[0]; j += DIM_PROSA_THETA)
        {
            PROSA_PARAMETERS sparams;
            reparametrize(d_i_params + j, &sparams);
            linearize_prosa(&sparams, d_o_params + j);
        }

        // Set the parameters
        mxSetCell(plhs[0], i, o_params);
    
    }  
    
    gsl_set_error_handler(NULL);

}

void
reparametrize_seria(
        int nlhs,
        mxArray *plhs[], 
        int nrhs,
        const mxArray *prhs[],
        FILL_PARAMETERS_SERIA reparametrize)
{

    int i;
    int na = 1;

    if ( nrhs != 1 )
    {
         mexErrMsgIdAndTxt("tapas:sem:input", "Invalid number of inputs."); 
    }

    if ( nlhs != 1 )
    {
         mexErrMsgIdAndTxt("tapas:sem:output", "Invalid output."); 
    }

    // Verify input
    
    verify_input_theta_array(prhs[0], DIM_SERIA_THETA);

    mwSize nd = mxGetNumberOfDimensions(prhs[0]);
    mwSize *np = mxGetDimensions(prhs[0]);

    for (i = 0; i < nd; i++)
    {
        // For a weird reason empty dimensions are set to zero.
        na *= ( np[i] == 0 )? 1 : np[i];
    }

    // Output will have the same structure
    plhs[0] = mxCreateCellArray(nd, np);
   
    gsl_set_error_handler_off();
    for (i = 0; i < na; i++)
    {
        int j;

        mxArray *i_params = mxGetCell(prhs[0], i);
        double *d_i_params = mxGetPr(i_params);
        
        mwSize *dtheta = mxGetDimensions(i_params);

        mxArray *o_params = mxCreateDoubleMatrix(dtheta[0], 1, mxREAL); 
        
        double *d_o_params = mxGetPr(o_params);

        for (j = 0; j < dtheta[0]; j += DIM_SERIA_THETA)
        {
            SERIA_PARAMETERS sparams;
            reparametrize(d_i_params + j, &sparams);
            linearize_seria(&sparams, d_o_params + j);
        }
       
        // Set the parameters
        mxSetCell(plhs[0], i, o_params);
    
    }  
    
    gsl_set_error_handler(NULL);

}
