/* aponteeduardo@gmail.com */
/* copyright (C) 2018 */

#include "mexutils.h"

void
verify_input_theta_array(const mxArray *theta, int dims_theta)
{
    int i;
    int na = 1;

    if ( !mxIsCell(theta) )
    {
          mexErrMsgIdAndTxt("tapas:sem:input",
                  "Input parameters should be a cell array");
    }

    const mwSize nd = mxGetNumberOfDimensions(theta);
    const mwSize *np = mxGetDimensions(theta);

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

        const mxArray *i_params = mxGetCell(theta, i);

        const mwSize stheta = mxGetNumberOfDimensions(i_params);
        const mwSize *dtheta = mxGetDimensions(i_params);

        if ( !mxIsNumeric(i_params) )
        {
             mexErrMsgIdAndTxt("tapas:sem:input",
                    "Not all parameters are numeric.");
        }

        if ( !( mxGetClassID(i_params) == mxDOUBLE_CLASS )  )
        {
              mexErrMsgIdAndTxt("tapas:sem:input",
                    "Not all parameters are double.");          
        }

        if ( !mxIsNumeric(i_params) )
        {
             mexErrMsgIdAndTxt("tapas:sem:input",
                    "Not all parameters are numeric.");
        }

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
wrapper_prosa_n_states(
        int nlhs,
        mxArray *plhs[],
        int nrhs,
        const mxArray *prhs[],
        FILL_PARAMETERS_PROSA reparametrize)
{
    ANTIS_INPUT svals;
    double *llh;
    PROSA_MODEL model;
    int i;
    int nd = 0;
    int *np;
    int na = 1;

    svals.t = mxGetPr(prhs[0]);
    svals.a = mxGetPr(prhs[1]);
    svals.u = mxGetPr(prhs[2]);
    svals.theta = mxGetPr(prhs[3]);

    svals.nt = *mxGetDimensions(prhs[0]);
    verify_input_theta_array(prhs[3], DIM_PROSA_THETA);

    nd = mxGetNumberOfDimensions(prhs[3]);
    np = mxGetDimensions(prhs[3]);

    for (i = 0; i < nd; i++)
    {
        // For a weird reason empty dimensions are set to zero.
        na *= (np[i] == 0)? 1: np[i];
    }

    if ( na % DIM_PROSA_THETA != 0)
    {
        mexErrMsgIdAndTxt("tapas:sem:input", "Dimensions are not correct");
    }

    svals.np = na / DIM_PROSA_THETA;

    plhs[0] = mxCreateDoubleMatrix(svals.nt, 1, mxREAL);
    llh = mxGetPr(plhs[0]);

    model.llh = prosa_llh_abstract;
    model.fill_parameters = reparametrize;
    gsl_set_error_handler_off();
    prosa_model_n_states(svals, model, llh);
    gsl_set_error_handler(NULL);
}

void
wrapper_seria_n_states(
        int nlhs,
        mxArray *plhs[],
        int nrhs,
        const mxArray *prhs[],
        FILL_PARAMETERS_SERIA reparametrize)
{
    ANTIS_INPUT svals;
    double *llh;
    SERIA_MODEL model;
    int i;
    int nd = 0;
    int *np;
    int na = 1;

    svals.t = mxGetPr(prhs[0]);
    svals.a = mxGetPr(prhs[1]);
    svals.u = mxGetPr(prhs[2]);
    svals.theta = mxGetPr(prhs[3]);

    verify_input_theta_array(prhs[3], DIM_SERIA_THETA);
    svals.nt = *mxGetDimensions(prhs[0]);

    nd = mxGetNumberOfDimensions(prhs[3]);
    np = mxGetDimensions(prhs[3]);


    for (i = 0; i < nd; i++)
    {
        // For a weird reason empty dimensions are set to zero.
        na *= (np[i] == 0)? 1: np[i];
    }

    svals.np = na / DIM_SERIA_THETA;

    plhs[0] = mxCreateDoubleMatrix(svals.nt, 1, mxREAL);
    llh = mxGetPr(plhs[0]);
   
    model.llh = seria_llh_abstract;
    model.fill_parameters = reparametrize; 
    gsl_set_error_handler_off();
    seria_model_n_states(svals, model, llh);
    gsl_set_error_handler(NULL);
}

void
wrapper_seria_multi(
        int nlhs,
        mxArray *plhs[],
        int nrhs,
        const mxArray *prhs[],
        FILL_PARAMETERS_SERIA reparametrize)
{
    double *llh;
    SERIA_MODEL model;
    int i, j;
    int ns = mxGetDimensions(prhs[1])[0];
    int nc = mxGetDimensions(prhs[1])[1];

    if ( nlhs != 1 )
        HANDLE_CERROR_MSG( 1 , "Wrong left size assignments.");

    if ( nrhs != 2 )
        HANDLE_CERROR_MSG( 1 , "Wrong right size assignments." );

    verify_input_theta_array(prhs[1], DIM_SERIA_THETA);

    plhs[0] = mxCreateDoubleMatrix(ns, nc, mxREAL);
    llh = mxGetPr(plhs[0]);

    model.llh = seria_llh_abstract;
    model.fill_parameters = reparametrize;
    gsl_set_error_handler_off();

    #pragma omp parallel for private(i) private(j) collapse(2) schedule(dynamic)
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
            svals.np = (mxGetDimensions(theta)[0]
                * mxGetDimensions(theta)[1])/DIM_SERIA_THETA;

            tllh = (double *) malloc(svals.nt * sizeof(double));

            seria_model_n_states_optimized(svals, model, tllh);

            llh[i + ns * j] = 0;
            for (k = 0; k < svals.nt; k++)
            {
                llh[i + ns * j] += tllh[k];
            }
            if ( fabs(llh[i + ns * j]) == INFINITY || isnan(llh[i + ns * j]))
                llh[i + ns * j] = -INFINITY;
            free(tllh);

        }
    }
   gsl_set_error_handler(NULL);
}

void
wrapper_prosa_multi(
        int nlhs,
        mxArray *plhs[],
        int nrhs,
        const mxArray *prhs[],
        FILL_PARAMETERS_PROSA reparametrize)
{
    double *llh;
    PROSA_MODEL model;
    int i, j;
    int ns = mxGetDimensions(prhs[1])[0];
    int nc = mxGetDimensions(prhs[1])[1];

    if ( nlhs != 1 )
        HANDLE_CERROR_MSG( 1 , "Wrong left size assignments.");

    if ( nrhs != 2 )
        HANDLE_CERROR_MSG( 1 , "Wrong right size assignments." );

    verify_input_theta_array(prhs[1], DIM_PROSA_THETA);

    plhs[0] = mxCreateDoubleMatrix(ns, nc, mxREAL);
    llh = mxGetPr(plhs[0]);

    model.llh = prosa_llh_abstract;
    model.fill_parameters = reparametrize;
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
            svals.np = (mxGetDimensions(theta)[0]
                * mxGetDimensions(theta)[1])/DIM_PROSA_THETA;

            tllh = (double *) malloc(svals.nt * sizeof(double));

            prosa_model_n_states_optimized(svals, model, tllh);

            llh[i + ns * j] = 0;
            for (k = 0; k < svals.nt; k++)
            {
                llh[i + ns * j] += tllh[k];
            }
            if ( fabs(llh[i + ns * j]) == INFINITY || isnan(llh[i + ns * j]))
                llh[i + ns * j] = -INFINITY;
            free(tllh);

        }
    }
   gsl_set_error_handler(NULL);
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

    const mwSize nd = mxGetNumberOfDimensions(prhs[0]);
    const mwSize *np = mxGetDimensions(prhs[0]);

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

        const mxArray *i_params = mxGetCell(prhs[0], i);
        double *d_i_params = mxGetPr(i_params);

        const mwSize stheta = mxGetNumberOfDimensions(i_params);
        const mwSize *dtheta = mxGetDimensions(i_params);

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

    const mwSize nd = mxGetNumberOfDimensions(prhs[0]);
    const mwSize *np = mxGetDimensions(prhs[0]);

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

        const mwSize *dtheta = mxGetDimensions(i_params);

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

