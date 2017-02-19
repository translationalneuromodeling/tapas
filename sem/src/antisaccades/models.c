/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */

#include "antisaccades.h"

int
prosa_model_trial_by_trial(const ANTIS_INPUT svals, PROSA_LLH fllh, 
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
        populate_parameters_prosa(theta + i * DIM_PROSA_THETA, &ptheta);
        llh[i] = fllh(t[i], a[i], ptheta);
    }

    gsl_set_error_handler(NULL);


    return 0;
}

int
prosa_model_two_states(const ANTIS_INPUT svals, PROSA_LLH fllh, 
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
        populate_parameters_prosa(theta + ((int ) u[i]) * DIM_PROSA_THETA, 
                &ptheta);
        llh[i] = fllh(t[i], a[i], ptheta);
    }

    gsl_set_error_handler(NULL);

    return 0;
}

    
int
seri_model_trial_by_trial(const ANTIS_INPUT svals, SERI_LLH fllh, double *llh)
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
        populate_parameters_seri(theta + i * DIM_SERI_THETA, &ptheta);
        llh[i] = fllh(t[i], a[i], ptheta);
    }

    gsl_set_error_handler(NULL);


    return 0;
}

int
seri_model_two_states(const ANTIS_INPUT svals, SERI_LLH fllh, double *llh)
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
        SERI_PARAMETERS ptheta;
        populate_parameters_seri(theta + ((int ) u[i]) * DIM_SERI_THETA, 
                &ptheta);
        llh[i] = fllh(t[i], a[i], ptheta);
    }

    gsl_set_error_handler(NULL);

    return 0;
}


int
dora_model_trial_by_trial(const ANTIS_INPUT svals, DORA_LLH fllh, double *llh)
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
        populate_parameters_dora(theta + i * DIM_DORA_THETA, &ptheta);
        llh[i] = fllh(t[i], a[i], ptheta);
    }

    gsl_set_error_handler(NULL);


    return 0;
}

int
dora_model_two_states(const ANTIS_INPUT svals, DORA_LLH fllh, double *llh)
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
        DORA_PARAMETERS ptheta;
        populate_parameters_dora(theta + ((int ) u[i]) * DIM_DORA_THETA, 
                &ptheta);
        llh[i] = fllh(t[i], a[i], ptheta);
    }

    gsl_set_error_handler(NULL);

    return 0;
}


