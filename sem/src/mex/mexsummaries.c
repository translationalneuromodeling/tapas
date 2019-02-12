/* aponteeduardo@gmail.com */
/* copyright (C) 2018 */

#include "mexutils.h"

int
seria_set_summary_fields(
        const SERIA_SUMMARY *summary, 
        int index, 
        mxArray *mstruct)
{
    
    mxArray *late_pro_prob = mxCreateDoubleScalar(summary->late_pro_prob);
    mxSetField(mstruct, index, "late_pro_prob", late_pro_prob);
    
    mxArray *inhib_fail_prob = mxCreateDoubleScalar(summary->inhib_fail_prob);
    mxSetField(mstruct, index, "inhib_fail_prob", inhib_fail_prob);

    mxArray *inhib_fail_rt = mxCreateDoubleScalar(summary->inhib_fail_rt);
    mxSetField(mstruct, index, "inhib_fail_rt", inhib_fail_rt);

    mxArray *late_pro_rt = mxCreateDoubleScalar(summary->late_pro_rt);
    mxSetField(mstruct, index, "late_pro_rt", late_pro_rt);

    mxArray *anti_rt = mxCreateDoubleScalar(summary->anti_rt);
    mxSetField(mstruct, index, "anti_rt", anti_rt);

    mxArray *predicted_anti_rt = 
        mxCreateDoubleScalar(summary->predicted_anti_rt);
    mxSetField(mstruct, index, "predicted_anti_rt", predicted_anti_rt);

    mxArray *predicted_pro_rt = 
        mxCreateDoubleScalar(summary->predicted_pro_rt);
    mxSetField(mstruct, index, "predicted_pro_rt", predicted_pro_rt);

    mxArray *predicted_anti_prob = 
        mxCreateDoubleScalar(summary->predicted_anti_prob);
    mxSetField(mstruct, index, "predicted_anti_prob", predicted_anti_prob);

    mxArray *predicted_pro_prob = 
        mxCreateDoubleScalar(summary->predicted_pro_prob);
    mxSetField(mstruct, index, "predicted_pro_prob", predicted_pro_prob);

    return 0;

}

void
wrapper_seria_summaries(
        int nlhs,
        mxArray *plhs[],
        int nrhs,
        const mxArray *prhs[],
        FILL_PARAMETERS_SERIA reparametrize)
{

    if ( nlhs != 1 )
        HANDLE_CERROR_MSG( 1 , "Wrong left size assignments.");

    if ( nrhs != 1 )
        HANDLE_CERROR_MSG( 1 , "Wrong right size assignments." );

    
    SERIA_MODEL model;
    int i, j;
    // Number of subjects
    int ns = mxGetDimensions(prhs[0])[0];
    // Number of parameters
    int nc = mxGetDimensions(prhs[0])[1];

    // If the second dimensions is zero then 
    if ( nc == 0 && ns > 0 )
    {
        nc = 1;
    }

    // Verify that theta has the right structure
    verify_input_theta_array(prhs[0], DIM_SERIA_THETA);

    // Create output array
    plhs[0] = mxCreateCellMatrix(ns, nc);

    // Initilize model
    model.llh = seria_llh_abstract;
    model.fill_parameters = reparametrize;
    gsl_set_error_handler_off();


    for (j = 0; j < nc; j++)
    {
        for (i = 0; i < ns; i++)
        {

            ANTIS_INPUT svals;
            mxArray *theta = mxGetCell(prhs[0], i + ns * j);
           
            svals.theta = mxGetPr(theta);

            // There is no input (in terms of time)
            svals.nt = 0; 
            svals.np = (mxGetDimensions(theta)[0]
                * mxGetDimensions(theta)[1])/DIM_SERIA_THETA;


            // Allocate memory for the summaries.
            SERIA_SUMMARY *summaries = 
                (SERIA_SUMMARY *) malloc(sizeof(SERIA_SUMMARY) * svals.np);

            seria_model_summary(svals, model, summaries);

            int nfields = 9;
            char *fields[] = {
                "late_pro_prob", "inhib_fail_prob", 
                "late_pro_rt", "anti_rt",
                "inhib_fail_rt",
                "predicted_pro_rt", "predicted_pro_prob",
                "predicted_anti_rt", "predicted_anti_prob",
                };

            // Create the output
            mxArray *results = mxCreateStructMatrix(
                    svals.np, // output first dimension
                    1, // Ouput second dimensions
                    nfields,
                    fields);

            for (int k = 0; k < svals.np; k++)
            {
                seria_set_summary_fields(summaries + k, k, results);
            }
            
            // Populate the input;
            mxSetCell(plhs[0], i + ns * j, results);

            // Clean memory 
            free(summaries);

        }
    }
   gsl_set_error_handler(NULL);
}

int
prosa_set_summary_fields(
        const PROSA_SUMMARY *summary, 
        int index, 
        mxArray *mstruct)
{
        
    mxArray *inhib_fail_prob = mxCreateDoubleScalar(summary->inhib_fail_prob);
    mxSetField(mstruct, index, "inhib_fail_prob", inhib_fail_prob);

    mxArray *inhib_fail_rt = mxCreateDoubleScalar(summary->inhib_fail_rt);
    mxSetField(mstruct, index, "inhib_fail_rt", inhib_fail_rt);

    mxArray *anti_rt = mxCreateDoubleScalar(summary->anti_rt);
    mxSetField(mstruct, index, "anti_rt", anti_rt);

    mxArray *predicted_anti_rt = 
        mxCreateDoubleScalar(summary->predicted_anti_rt);
    mxSetField(mstruct, index, "predicted_anti_rt", predicted_anti_rt);

    mxArray *predicted_pro_rt = 
        mxCreateDoubleScalar(summary->predicted_pro_rt);
    mxSetField(mstruct, index, "predicted_pro_rt", predicted_pro_rt);

    mxArray *predicted_anti_prob = 
        mxCreateDoubleScalar(summary->predicted_anti_prob);
    mxSetField(mstruct, index, "predicted_anti_prob", predicted_anti_prob);

    mxArray *predicted_pro_prob = 
        mxCreateDoubleScalar(summary->predicted_pro_prob);
    mxSetField(mstruct, index, "predicted_pro_prob", predicted_pro_prob);

    return 0;

}

void
wrapper_prosa_summaries(
        int nlhs,
        mxArray *plhs[],
        int nrhs,
        const mxArray *prhs[],
        FILL_PARAMETERS_PROSA reparametrize)
{

    if ( nlhs != 1 )
        HANDLE_CERROR_MSG( 1 , "Wrong left size assignments.");

    if ( nrhs != 1 )
        HANDLE_CERROR_MSG( 1 , "Wrong right size assignments." );

    
    PROSA_MODEL model;
    int i, j;
    // Number of subjects
    int ns = mxGetDimensions(prhs[0])[0];
    // Number of parameters
    int nc = mxGetDimensions(prhs[0])[1];

    // If the second dimensions is zero then 
    if ( nc == 0 && ns > 0 )
    {
        nc = 1;
    }

    // Verify that theta has the right structure
    verify_input_theta_array(prhs[0], DIM_PROSA_THETA);

    // Create output array
    plhs[0] = mxCreateCellMatrix(ns, nc);

    // Initilize model
    model.llh = prosa_llh_abstract;
    model.fill_parameters = reparametrize;
    gsl_set_error_handler_off();


    for (j = 0; j < nc; j++)
    {
        for (i = 0; i < ns; i++)
        {

            ANTIS_INPUT svals;
            mxArray *theta = mxGetCell(prhs[0], i + ns * j);
           
            svals.theta = mxGetPr(theta);

            // There is no input (in terms of time)
            svals.nt = 0; 
            svals.np = (mxGetDimensions(theta)[0]
                * mxGetDimensions(theta)[1])/DIM_PROSA_THETA;


            // Allocate memory for the summaries.
            PROSA_SUMMARY *summaries = 
                (PROSA_SUMMARY *) malloc(sizeof(PROSA_SUMMARY) * svals.np);

            prosa_model_summary(svals, model, summaries);

            int nfields = 7;
            char *fields[] = {
                "inhib_fail_prob", 
                "anti_rt",
                "inhib_fail_rt",
                "predicted_pro_rt", "predicted_pro_prob",
                "predicted_anti_rt", "predicted_anti_prob",
                };

            // Create the output
            mxArray *results = mxCreateStructMatrix(
                    svals.np, // output first dimension
                    1, // Ouput second dimensions
                    nfields,
                    fields);

            for (int k = 0; k < svals.np; k++)
            {
                prosa_set_summary_fields(summaries + k, k, results);
            }
            
            // Populate the input;
            mxSetCell(plhs[0], i + ns * j, results);

            // Clean memory 
            free(summaries);

        }
    }
   gsl_set_error_handler(NULL);
}

