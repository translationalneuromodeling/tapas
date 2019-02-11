/* aponteeduardo@gmail.com */
/* copyright (C) 2019 */

#include "antisaccades.h"

#define INTSTEPS 200;

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

    gsl_integration_qagiu(f, 0, SEM_TOL, SEM_RTOL, wspace, &result, &ierror);

    return result;

}


double
seria_late_pro(double t, SERIA_PARAMETERS *params)
{
    
    double lp = params->early.lpdf(t, params->kl, params->tl) +
        params->stop.lsf(t, params->ka, params->ta);
    
    lp = exp(lp);

    return lp;

}


double
seria_inhib_prob(double t, SERIA_PARAMETERS *params)
{
    
    double lp = params->early.lcdf(t, params->kp, params->tp) +
        params->stop.lsf(t, params->ks, params->ts);
    
    // Include the delay of the late units
    t -= params->da;

    if ( t > 0 )
    {
        lp += params->late.lsf(t, params->ka, params->ta) +
            params->anti.lsf(t, params->kl, params->tl);
    }

    lp = exp(lp);

    return lp;

}


double
seria_summary_wrapper(double t, void *gsl_int_pars)
{
    
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
    summary->inhib_prob = seria_summary_parameter(seria_inhib_prob, params);
    summary->late_pro_prob = seria_summary_parameter(seria_late_pro, params);

    return 0;

}
