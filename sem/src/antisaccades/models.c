/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */

#include "antisaccades.h"
#include <stdlib.h>
#include <string.h>


/* Compares between trials */
int cmpfunc(const void *a, const void *b)
{
    // Get the values from the pointers
    double *da = ((MODEL_INPUTS *) a)->t;
    double *db = ((MODEL_INPUTS *) b)->t;
    double diff = (*da) - (*db); 
    return ( (diff > 0) - (diff < 0) );
}

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
prosa_auxiliary(double t, double a, PROSA_MODEL model, 
    PROSA_PARAMETERS *ptheta, double *ct)
{

    double t0 = t - ptheta->t0;

    // In case that it has not been initialize, do it now
    
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
            ptheta->cumint += ptheta->inhibition_race(*ct, t0, ptheta->kp,
                   ptheta->ks, ptheta->tp, ptheta->ts);
            *ct = t0;
        }   
    }

    return 0;
}

int
prosa_model_n_states_optimized(const ANTIS_INPUT svals, PROSA_MODEL model, 
        double *llh)
{
    int i, j;
    double *t = svals.t;
    double *a = svals.a;
    double *u = svals.u;
    double *theta = svals.theta;

    int nt = svals.nt;
    int np = svals.np; /* Sets the number of parameters */

    /* Enter arbitrary number of parameters */
    double *old_times = (double *) malloc(np * sizeof( double ));
    PROSA_PARAMETERS *ptheta = (PROSA_PARAMETERS *) 
        malloc( np * sizeof(PROSA_PARAMETERS) );

    for (j = 0; j < np; j++)
    {
        model.fill_parameters(theta + j * DIM_PROSA_THETA, ptheta + j);
        old_times[j] = ZERO_DISTS;
    }

    for (i = 0; i < nt; i++)
    {
        // Update if necessary
        int trial_type = u[i];
        prosa_auxiliary(t[i], a[i], model, ptheta + trial_type, 
                old_times + trial_type);
        llh[i] = model.llh(t[i], a[i], ptheta[trial_type]);
    }
    
    /* Clean up memory */

    free(ptheta);
    free(old_times);

    return 0;
}

int
prosa_model_n_states(const ANTIS_INPUT svals, PROSA_MODEL model, 
        double *llh)
{
    int i, j, k;
    int nt = svals.nt;
    double buff;
   
    MODEL_INPUTS *inputs = 
        (MODEL_INPUTS *) malloc(nt * sizeof(MODEL_INPUTS));
   
    double *sorted_t = (double *) malloc(nt * sizeof(double));
    double *sorted_a = (double *) malloc(nt * sizeof(double));
    double *sorted_u = (double *) malloc(nt * sizeof(double));

    ANTIS_INPUT tvals;

    tvals.theta = svals.theta;
    tvals.nt = svals.nt;
    tvals.np = svals.np;

    for (i = 0; i < nt; i++)
    {
        inputs[i].a = svals.a + i;
        inputs[i].u = svals.u + i;
        inputs[i].t = svals.t + i;
    }

    /* Sort the times. */
    qsort(inputs, nt, sizeof(MODEL_INPUTS), cmpfunc);

    for (i = 0; i < nt; i++)
    {
        sorted_t[i] = *(inputs[i].t);
        sorted_a[i] = *(inputs[i].a);
        sorted_u[i] = *(inputs[i].u);
    }

    tvals.t = sorted_t;
    tvals.a = sorted_a;
    tvals.u = sorted_u;

    prosa_model_n_states_optimized(tvals, model, llh);

    k = 0;
    while (k < nt)
    {
        if ( inputs[k].t - svals.t == k )
        {
            k++;
            continue;
        }

        i = inputs[k].t - svals.t;
        j = inputs[i].t - svals.t;

        inputs[i].t = i + svals.t;
        inputs[k].t = j + svals.t;

        buff = llh[k];

        llh[k] = llh[i];
        llh[i] = buff;

    }

    free(sorted_t);
    free(sorted_a);
    free(sorted_u);
    free(inputs);

    return 0;
}

/* ---------------------------------------------------------------------*/
int
seria_model_trial_by_trial(const ANTIS_INPUT svals, SERIA_MODEL model, 
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
        SERIA_PARAMETERS ptheta;
        model.fill_parameters(theta + i * DIM_SERIA_THETA, &ptheta);
        llh[i] = model.llh(t[i], a[i], ptheta);
    }

    gsl_set_error_handler(NULL);


    return 0;
}

int
seria_auxiliary(double t, double a, SERIA_PARAMETERS *ptheta, double *ct)
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
            ptheta->cumint += ptheta->inhibition_race(*ct, t0, ptheta->kp,
                   ptheta->ks, ptheta->tp, ptheta->ts);
            /* Update the value. */
            *ct = t0;
        }   
    }

    return 0;
}


int
seria_model_n_states(const ANTIS_INPUT svals, SERIA_MODEL model, 
        double *llh)
{
    int i, j, k;
    int nt = svals.nt;
    double buff;
   
    MODEL_INPUTS *inputs = 
        (MODEL_INPUTS *) malloc(nt * sizeof(MODEL_INPUTS));
   
    double *sorted_t = (double *) malloc(nt * sizeof(double));
    double *sorted_a = (double *) malloc(nt * sizeof(double));
    double *sorted_u = (double *) malloc(nt * sizeof(double));

    ANTIS_INPUT tvals;

    tvals.theta = svals.theta;
    tvals.nt = svals.nt;
    tvals.np = svals.np;

    for (i = 0; i < nt; i++)
    {
        inputs[i].a = svals.a + i;
        inputs[i].u = svals.u + i;
        inputs[i].t = svals.t + i;
    }

    /* Sort the times. */
    qsort(inputs, nt, sizeof(MODEL_INPUTS), cmpfunc);

    for (i = 0; i < nt; i++)
    {
        sorted_t[i] = *(inputs[i].t);
        sorted_a[i] = *(inputs[i].a);
        sorted_u[i] = *(inputs[i].u);
    }

    tvals.t = sorted_t;
    tvals.a = sorted_a;
    tvals.u = sorted_u;

    seria_model_n_states_optimized(tvals, model, llh);

    k = 0;
    while (k < nt)
    {
        if ( inputs[k].t - svals.t == k )
        {
            k++;
            continue;
        }

        i = inputs[k].t - svals.t;
        j = inputs[i].t - svals.t;

        inputs[i].t = i + svals.t;
        inputs[k].t = j + svals.t;

        buff = llh[k];

        llh[k] = llh[i];
        llh[i] = buff;

    }

    free(sorted_t);
    free(sorted_a);
    free(sorted_u);
    free(inputs);

    return 0;
}


int
seria_model_n_states_optimized(const ANTIS_INPUT svals, SERIA_MODEL model, 
        double *llh)
{
    int i, j;
    double *t = svals.t;
    double *a = svals.a;
    double *u = svals.u;
    double *theta = svals.theta;

    int nt = svals.nt;
    int np = svals.np; /* Sets the number of parameters */

    /* Enter arbitrary number of parameters */
    double *old_times = (double *) malloc(np * sizeof( double ));

    SERIA_PARAMETERS *ptheta = (SERIA_PARAMETERS *) 
        malloc( np * sizeof(SERIA_PARAMETERS) );

    for (j = 0; j < np; j++)
    {
        model.fill_parameters(theta + j * DIM_SERIA_THETA, ptheta + j);
        old_times[j] = ZERO_DISTS;
    }

    for (i = 0; i < nt; i++)
    {
        // Update if necessary
        int trial_type = u[i];
        seria_auxiliary(
                t[i], /* Time */
                a[i], /* Action */
                ptheta + trial_type, /* Parameter set */
                old_times + trial_type /* Previous time. */
                );
        llh[i] = model.llh(t[i], a[i], ptheta[trial_type]);
    }
    
    /* Clean up memory */

    free(ptheta);
    free(old_times);

    return 0;
}

int
seria_model_summary(
    const ANTIS_INPUT svals,
    SERIA_MODEL model, 
    SERIA_SUMMARY *summaries)
{

    double *theta = svals.theta;
    
    int np = svals.np; /* Sets the number of parameters */

    SERIA_PARAMETERS params;
    // The parameters
    
    for (int i = 0; i < np; i++)
    {
        // Initilize the parameters.
        model.fill_parameters(theta + i * DIM_SERIA_THETA, &params);
        
        // Generate the summary. 
        seria_summary_abstract(&params, summaries + i);
    }

    return 0;

}

int
prosa_model_summary(
    const ANTIS_INPUT svals,
    PROSA_MODEL model, 
    PROSA_SUMMARY *summaries)
{

    double *theta = svals.theta;
    
    int np = svals.np; /* Sets the number of parameters */

    PROSA_PARAMETERS params;
    // The parameters
    
    for (int i = 0; i < np; i++)
    {
        // Initilize the parameters.
        model.fill_parameters(theta + i * DIM_PROSA_THETA, &params);
        
        // Generate the summary. 
        prosa_summary_abstract(&params, summaries + i);
    }

    return 0;

}
