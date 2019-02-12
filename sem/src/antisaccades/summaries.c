/* aponteeduardo@gmail.com */
/* copyright (C) 2019 */

#include "antisaccades.h"
#include <signal.h>

#define INTSTEPS 200;

void
intHandler(int dummy)
{
    mexErrMsgIdAndTxt("tapas:user_exception", "Stopped by the user");
}

double
seria_summary_parameter(
        SERIA_SUMMARY_FUNCTION summary_func,
        SERIA_PARAMETERS *params)
{

    double result;
    double ierror;

    gsl_integration_workspace *wspace = 
        gsl_integration_workspace_alloc( 1000 );

    SERIA_GSL_INT_INPUT input;
    
    input.func = summary_func;
    input.params = params;
    
    gsl_function f;
    f.function = seria_summary_wrapper;
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
seria_late_pro_rt(double t, SERIA_PARAMETERS *params)
{
    double delay = params->t0 + params->da;

    // Below the delays
    if ( t < delay )
        return 0;

    t -= delay;
    double eval = log(t + delay) + 
        params->late.lpdf(t, params->kl, params->tl) + 
        params->anti.lsf(t, params->ka, params->ta);

    return exp(eval);
}

double
seria_anti_rt(double t, SERIA_PARAMETERS *params)
{
    double delay = params->t0 + params->da;

    // Below the delays
    if ( t < delay )
        return 0;

    t -= delay;
    double eval = log(t + delay) +
        params->anti.lpdf(t, params->ka, params->ta) +
        params->late.lsf(t, params->kl, params->tl);

    return exp(eval);
}


double
seria_inhib_rt(double t, SERIA_PARAMETERS *params)
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
        eval += params->anti.lsf(t, params->ka, params->ta) +
            params->late.lsf(t, params->kl, params->tl);

    return exp(eval);

}


double
seria_inhib_fail_prob(double t, SERIA_PARAMETERS *params)
{
    
    t -= params->t0;
    
    if (t <= 0)
        return 0;
    
    double lp = params->early.lpdf(t, params->kp, params->tp) +
        params->stop.lsf(t, params->ks, params->ts);
    
    t -= params->da;

    if ( t > 0 )
        lp += params->late.lsf(t, params->kl, params->tl) + 
            params->anti.lsf(t, params->ka, params->ta);
         
    lp = exp(lp);

    return lp;

}

double
seria_late_pro_prob(double t, SERIA_PARAMETERS *params)
{ 
    // Include the delay of the late units
    t -= params->da;

    if ( t < 0 )
    {
        return 0;
    }

    double lp = params->late.lpdf(t, params->kl, params->tl) +
        params->anti.lsf(t, params->ka, params->ta);

    lp = exp(lp);

    return lp;

}

double
seria_predicted_anti_prob(double t, SERIA_PARAMETERS *params)
{
    
    // Log likelihood of a prosaccade
    double llh = seria_llh_abstract(t, ANTISACCADE, *params);
    if ( isnan(llh) )
        llh = GSL_NEGINF;

    return exp(llh); 

}

double
seria_predicted_pro_prob(double t, SERIA_PARAMETERS *params)
{

    // Log likelihood of a prosaccade
    double llh = seria_llh_abstract(t, PROSACCADE, *params);
    if ( isnan(llh) )
        llh = GSL_NEGINF;

    return exp(llh); 

}

double
seria_predicted_anti_rt(double t, SERIA_PARAMETERS *params)
{
    
    // Log likelihood of a prosaccade
    double llh = seria_llh_abstract(t, ANTISACCADE, *params);
    
    if ( isnan(llh) )
        llh = GSL_NEGINF;

    return exp(log(t) + llh); 

}

double
seria_predicted_pro_rt(double t, SERIA_PARAMETERS *params)
{

    // Log likelihood of a prosaccade
    double llh = seria_llh_abstract(t, PROSACCADE, *params);
    
    if ( isnan( llh ) )
        llh = GSL_NEGINF;

    return exp(log(t) + llh); 

}


double
seria_summary_wrapper(double t, void *gsl_int_pars)
{
    // Stop by the user if necessary
    //signal(SIGINT, intHandler); 
    // Cast the input into a function.
    SERIA_GSL_INT_INPUT *pars = (SERIA_GSL_INT_INPUT *) gsl_int_pars;

    // Compute the function
    double val = pars->func(t, pars->params);

    return val;

};


int
seria_summary_abstract(SERIA_PARAMETERS *params, SERIA_SUMMARY *summary)
{

    // Inhibition probability
    summary->inhib_fail_prob = seria_summary_parameter(
            seria_inhib_fail_prob, params);
    summary->late_pro_prob = seria_summary_parameter(
            seria_late_pro_prob, params);

    summary->late_pro_rt = seria_summary_parameter(seria_late_pro_rt, params);
    // Normalize the integral
    summary->late_pro_rt /= summary->late_pro_prob;

    summary->anti_rt = seria_summary_parameter(seria_anti_rt, params);
    // Normalize the integral
    summary->anti_rt /= 1.0 - summary->late_pro_prob;

    summary->inhib_fail_rt = seria_summary_parameter(seria_inhib_rt, 
            params);
    // Normalize the integral
    summary->inhib_fail_rt /= summary->inhib_fail_prob;

    summary->predicted_pro_prob = 
        seria_summary_parameter(seria_predicted_pro_prob, params);
    summary->predicted_pro_rt = 
        seria_summary_parameter(seria_predicted_pro_rt, params);
    summary->predicted_pro_rt /= summary->predicted_pro_prob;

    summary->predicted_anti_prob = 1.0 - summary->predicted_pro_prob;
    
    summary->predicted_anti_rt = 
        seria_summary_parameter(seria_predicted_anti_rt, params);
    summary->predicted_anti_rt /= summary->predicted_anti_prob;


    return 0;

}
