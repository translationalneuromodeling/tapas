/* aponteeduardo@gmail.com */
/* copyright (C) 2019 */

#include "antisaccades.h"

#define INTSTEPS 200;

double
prosa_summary_parameter(
        PROSA_SUMMARY_FUNCTION summary_func,
        PROSA_PARAMETERS *params)
{

    double result;
    double ierror;

    gsl_integration_workspace *wspace = 
        gsl_integration_workspace_alloc( 1000 );

    PROSA_GSL_INT_INPUT input;
    
    input.func = summary_func;
    input.params = params;
    
    gsl_function f;
    f.function = prosa_summary_wrapper;
    f.params = (void *) &input;

    gsl_integration_qagiu(
            &f, // Gsl function
            0.0000000001, // Lower boundary
            SEM_TOL, 
            SEM_RTOL, 
            1000, // Limit the number of iterations
            wspace, 
            &result, &ierror);

    gsl_integration_workspace_free(wspace);

    return result;

}

double
prosa_anti_rt(double t, PROSA_PARAMETERS *params)
{
    double delay = params->t0 + params->da;

    // Below the delays
    if ( t < delay )
        return 0;

    t -= delay;
    double eval = log(t + delay) +
        params->anti.lpdf(t, params->ka, params->ta);

    return exp(eval);

}


double
prosa_inhib_rt(double t, PROSA_PARAMETERS *params)
{
    double delay = params->t0;

    // Below the delays
    if ( t < delay )
        return 0;

    t -= delay;
    
    double eval = log(t + delay) +
        params->early.lpdf(t, params->kp, params->tp) +
        params->stop.lsf(t, params->ks, params->ts);

    t -= params->da;
    // Account for the delay
    if ( 0 < t )
        eval += params->anti.lsf(t, params->ka, params->ta);

    return exp(eval);

}


double
prosa_inhib_fail_prob(double t, PROSA_PARAMETERS *params)
{
    
    t -= params->t0;
    
    if (t <= 0)
        return 0;
    
    double lp = params->early.lpdf(t, params->kp, params->tp) +
        params->stop.lsf(t, params->ks, params->ts);
    
    t -= params->da;

    if ( t > 0 )
        lp += params->anti.lsf(t, params->ka, params->ta);
         
    lp = exp(lp);

    return lp;

}


double
prosa_predicted_anti_prob(double t, PROSA_PARAMETERS *params)
{
    
    // Log likelihood of a prosaccade
    double llh = prosa_llh_abstract(t, ANTISACCADE, *params);
    if ( isnan(llh) )
        llh = GSL_NEGINF;

    return exp(llh); 

}

double
prosa_predicted_pro_prob(double t, PROSA_PARAMETERS *params)
{

    // Log likelihood of a prosaccade
    double llh = prosa_llh_abstract(t, PROSACCADE, *params);
    if ( isnan(llh) )
        llh = GSL_NEGINF;

    return exp(llh); 

}

double
prosa_predicted_anti_rt(double t, PROSA_PARAMETERS *params)
{
    
    // Log likelihood of a prosaccade
    double llh = prosa_llh_abstract(t, ANTISACCADE, *params);
    
    if ( isnan(llh) )
        llh = GSL_NEGINF;

    return exp(log(t) + llh); 

}

double
prosa_predicted_pro_rt(double t, PROSA_PARAMETERS *params)
{

    // Log likelihood of a prosaccade
    double llh = prosa_llh_abstract(t, PROSACCADE, *params);
    
    if ( isnan( llh ) )
        llh = GSL_NEGINF;

    return exp(log(t) + llh); 

}


double
prosa_summary_wrapper(double t, void *gsl_int_pars)
{
    // Stop by the user if necessary
    //signal(SIGINT, intHandler); 
    // Cast the input into a function.
    PROSA_GSL_INT_INPUT *pars = (PROSA_GSL_INT_INPUT *) gsl_int_pars;

    // Compute the function
    double val = pars->func(t, pars->params);

    return val;

};


int
prosa_summary_abstract(PROSA_PARAMETERS *params, PROSA_SUMMARY *summary)
{

    // Inhibition probability
    summary->inhib_fail_prob = prosa_summary_parameter(
            prosa_inhib_fail_prob, params);

    summary->anti_rt = prosa_summary_parameter(prosa_anti_rt, params);

    summary->inhib_fail_rt = prosa_summary_parameter(prosa_inhib_rt, 
            params);
    // Normalize the integral
    summary->inhib_fail_rt /= summary->inhib_fail_prob;

    summary->predicted_pro_prob = 
        prosa_summary_parameter(prosa_predicted_pro_prob, params);
    summary->predicted_pro_rt = 
        prosa_summary_parameter(prosa_predicted_pro_rt, params);
    summary->predicted_pro_rt /= summary->predicted_pro_prob;

    summary->predicted_anti_prob = 1.0 - summary->predicted_pro_prob;
    
    summary->predicted_anti_rt = 
        prosa_summary_parameter(prosa_predicted_anti_rt, params);
    summary->predicted_anti_rt /= summary->predicted_anti_prob;


    return 0;

}
