/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */

#include "antisaccades.h"

int
prosa_model_trial_by_trial(const ANTIS_INPUT svals, PROSA_MODEL model, 
    double *llh)
{
    int i;
    double *t = svals.t;
    double *a = svals.a;
    //double *u = svals.u;
    double *theta = svals.theta;

    int nt = svals.nt;

    gsl_set_error_handler_off();
    
    #pragma omp parallel for private(i)
    for (i = 0; i < nt; i++)
    {
        PROSA_PARAMETERS ptheta;
        model.fill_parameters(theta + i * DIM_PROSA_THETA, &ptheta);
        llh[i] = model.llh(t[i], a[i], ptheta);
    }

    gsl_set_error_handler(NULL);


    return 0;
}

int
prosa_model_two_states(const ANTIS_INPUT svals, PROSA_MODEL model, 
    double *llh)
{
    int i;
    double *t = svals.t;
    double *a = svals.a;
    double *u = svals.u;
    double *theta = svals.theta;

    int nt = svals.nt;

    gsl_set_error_handler_off();
    
    #pragma omp parallel for private(i)
    for (i = 0; i < nt; i++)
    {
        PROSA_PARAMETERS ptheta;
        model.fill_parameters(theta + ((int ) u[i]) * DIM_PROSA_THETA, 
                &ptheta);
        llh[i] = model.llh(t[i], a[i], ptheta);
    }

    gsl_set_error_handler(NULL);

    return 0;
}


int
prosa_auxiliary(double t, double a, PROSA_MODEL model, 
    PROSA_PARAMETERS *ptheta, double *ct)
{

    double t0 = t - ptheta->t0;

    // In case that it has not been initilize, do it now
    
    if ( ptheta->cumint == CUMINT_NO_INIT )
    {
        ptheta->cumint = 0;
    }

    // For now do nothing
    if ( t0 <= ZERO_DISTS )
    {
        ptheta->cumint = 0;
    } else 
    {
        if ( t0 > *ct )
        {
            ptheta->cumint += model.nested_integral(*ct, t0, ptheta->kp,
                   ptheta->ks, ptheta->tp, ptheta->ts);
            *ct = t0;
        }   
    }

    return 0;
}

int
prosa_model_two_states_optimized(const ANTIS_INPUT svals, PROSA_MODEL model, 
        double *llh)
{
    int i;
    double *t = svals.t;
    double ct = 0;
    double *a = svals.a;
    double *u = svals.u;
    double *theta = svals.theta;

    double ot_pro = ZERO_DISTS;
    double ot_anti = ZERO_DISTS;

    int nt = svals.nt;

    PROSA_PARAMETERS ptheta_pro;
    PROSA_PARAMETERS ptheta_anti;

    model.fill_parameters(theta, &ptheta_pro);
    model.fill_parameters(theta + DIM_PROSA_THETA, &ptheta_anti);

    for (i = 0; i < nt; i++)
    {
        switch ( (int ) u[i] )
        {
            case ANTISACCADE:
                // Update if necessary
                prosa_auxiliary(t[i], a[i], model, &ptheta_anti, &ot_anti);
                llh[i] = model.llh(t[i], a[i], ptheta_anti);
                break;
            case PROSACCADE:
                prosa_auxiliary(t[i], a[i], model, &ptheta_pro, &ot_pro);
                llh[i] = model.llh(t[i], a[i], ptheta_pro);
                break;
        }
    }

    return 0;
}

int
seri_model_trial_by_trial(const ANTIS_INPUT svals, SERI_MODEL model, 
        double *llh)
{
    int i;
    double *t = svals.t;
    double *a = svals.a;
    //double *u = svals.u;
    double *theta = svals.theta;

    int nt = svals.nt;

    gsl_set_error_handler_off();
    
    #pragma omp parallel for private(i)
    for (i = 0; i < nt; i++)
    {
        SERI_PARAMETERS ptheta;
        model.fill_parameters(theta + i * DIM_SERI_THETA, &ptheta);
        llh[i] = model.llh(t[i], a[i], ptheta);
    }

    gsl_set_error_handler(NULL);


    return 0;
}

int
seri_model_two_states(const ANTIS_INPUT svals, SERI_MODEL model, double *llh)
{
    int i;
    double *t = svals.t;
    double *a = svals.a;
    double *u = svals.u;
    double *theta = svals.theta;

    int nt = svals.nt;

    SERI_PARAMETERS ptheta_pro;
    SERI_PARAMETERS ptheta_anti;

    model.fill_parameters(theta, &ptheta_pro);
    model.fill_parameters(theta + DIM_SERI_THETA, &ptheta_anti);

    gsl_set_error_handler_off();
    
    #pragma omp parallel for private(i)
    for (i = 0; i < nt; i++)
    {
        llh[i] = model.llh(t[i], a[i], u[i] ? ptheta_anti : ptheta_pro);
    }

    gsl_set_error_handler(NULL);

    return 0;
}

int
seri_auxiliary(double t, double a, SERI_MODEL model, SERI_PARAMETERS *ptheta,
        double *ct)
{

    double t0 = t - ptheta->t0;

    // In case that it has not been initilize, do it now
    
    if ( ptheta->cumint == CUMINT_NO_INIT )
    {
        ptheta->cumint = 0;
    }

    // For now do nothing
    if ( t0 <= ZERO_DISTS )
    {
        ptheta->cumint = 0;
    } else 
    {
        if ( t0 > *ct )
        {
            ptheta->cumint += model.nested_integral(*ct, t0, ptheta->kp,
                   ptheta->ks, ptheta->tp, ptheta->ts);
            *ct = t0;
        }   
    }

    return 0;
}

int
seri_model_two_states_optimized(const ANTIS_INPUT svals, SERI_MODEL model, 
        double *llh)
{
    int i;
    double *t = svals.t;
    double ct = 0;
    double *a = svals.a;
    double *u = svals.u;
    double *theta = svals.theta;

    double ot_pro = ZERO_DISTS;
    double ot_anti = ZERO_DISTS;

    int nt = svals.nt;

    SERI_PARAMETERS ptheta_pro;
    SERI_PARAMETERS ptheta_anti;

    model.fill_parameters(theta, &ptheta_pro);
    model.fill_parameters(theta + DIM_SERI_THETA, &ptheta_anti);

    for (i = 0; i < nt; i++)
    {
        switch ( (int ) u[i] )
        {
            case ANTISACCADE:
                // Update if necessary
                seri_auxiliary(t[i], a[i], model, &ptheta_anti, &ot_anti);
                llh[i] = model.llh(t[i], a[i], ptheta_anti);
                break;
            case PROSACCADE:
                seri_auxiliary(t[i], a[i], model, &ptheta_pro, &ot_pro);
                llh[i] = model.llh(t[i], a[i], ptheta_pro);
                break;
        }
    }

    return 0;
}



int
dora_model_trial_by_trial(const ANTIS_INPUT svals, DORA_MODEL model, 
        double *llh)
{
    int i;
    double *t = svals.t;
    double *a = svals.a;
    //double *u = svals.u;
    double *theta = svals.theta;
    
    int nt = svals.nt;

    gsl_set_error_handler_off();
    
    #pragma omp parallel for private(i)
    for (i = 0; i < nt; i++)
    {
        DORA_PARAMETERS ptheta;
        model.fill_parameters(theta + i * DIM_DORA_THETA, &ptheta);
        llh[i] = model.llh(t[i], a[i], ptheta);
    }

    gsl_set_error_handler(NULL);


    return 0;
}

int
dora_model_two_states(const ANTIS_INPUT svals, DORA_MODEL model, double *llh)
{
    int i;
    double *t = svals.t;
    double *a = svals.a;
    double *u = svals.u;
    double *theta = svals.theta;

    int nt = svals.nt;

    gsl_set_error_handler_off();

    DORA_PARAMETERS ptheta_pro;
    DORA_PARAMETERS ptheta_anti;

    model.fill_parameters(theta, &ptheta_pro);
    model.fill_parameters(theta + DIM_DORA_THETA, &ptheta_anti);

    #pragma omp parallel for private(i)
    for (i = 0; i < nt; i++)
    {
        DORA_PARAMETERS ptheta;
        llh[i] = model.llh(t[i], a[i], u[i] ? ptheta_anti : ptheta_pro);
    }

    gsl_set_error_handler(NULL);

    return 0;
}

int
dora_auxiliary(double t, double a, DORA_MODEL model, DORA_PARAMETERS *ptheta,
        double *ct)
{

    double t0 = t - ptheta->t0;

    // In case that it has not been initilize, do it now
    
    if ( ptheta->cumint == CUMINT_NO_INIT )
    {
        ptheta->cumint = 0;
    }

    // For now do nothing
    if ( t0 <= ZERO_DISTS )
    {
        ptheta->cumint = 0;
    } else 
    {
        if ( t0 > *ct )
        {
            ptheta->cumint += model.nested_integral(*ct, t0, ptheta->kp,
                   ptheta->ks, ptheta->tp, ptheta->ts);
            *ct = t0;
        }   
    }

    return 0;
}

int
dora_model_two_states_optimized(const ANTIS_INPUT svals, DORA_MODEL model, 
        double *llh)
{
    int i;
    double *t = svals.t;
    double ct = 0;
    double *a = svals.a;
    double *u = svals.u;
    double *theta = svals.theta;

    double ot_pro = ZERO_DISTS;
    double ot_anti = ZERO_DISTS;

    int nt = svals.nt;

    DORA_PARAMETERS ptheta_pro;
    DORA_PARAMETERS ptheta_anti;

    model.fill_parameters(theta, &ptheta_pro);
    model.fill_parameters(theta + DIM_DORA_THETA, &ptheta_anti);

    for (i = 0; i < nt; i++)
    {
        switch ( (int ) u[i] )
        {
            case ANTISACCADE:
                // Update if necessary
                dora_auxiliary(t[i], a[i], model, &ptheta_anti, &ot_anti);
                llh[i] = model.llh(t[i], a[i], ptheta_anti);
                break;
            case PROSACCADE:
                dora_auxiliary(t[i], a[i], model, &ptheta_pro, &ot_pro);
                llh[i] = model.llh(t[i], a[i], ptheta_pro);
                break;
        }
    }

    return 0;
}

