/* aponteeduardo@gmail.com */
/* copyright (C) 2016 */

#include "antisaccades.h"

double
seria_llh_abstract(double t, int a, SERIA_PARAMETERS params)
{

    double t0 = params.t0;
    double p0 = params.p0;

    double kp = params.kp;
    double tp = params.tp;
    double ka = params.ka;
    double ta = params.ta;
    double kl = params.kl;
    double tl = params.tl;
    double ks = params.ks;
    double ts = params.ts;

    double da = params.da;

    double fllh = GSL_NEGINF;
    double sllh = GSL_NEGINF;

    double cumint = params.cumint;

    t -= t0;

    // Take care of the outliers
    if ( t < 0 )
    {
        switch ( a ){
            case PROSACCADE:
                fllh = LN_P_OUTLIER_PRO;
                break;
            case ANTISACCADE:
                fllh = LN_P_OUTLIER_ANTI;
                break;
        }

        /* This is the log of the sigmoid function
         * 1/2 x - ln 2 - ln (exp(x/2) + exp(-x/2)/2)
         * 1/2 x - ln 2 - ln ( 2*exp(-x/2) / (1 + exp(-x)))
         * -> exp
         *  exp(x/2) * 1/2 * 2 * exp(-x/2)/(1 + exp(-x))
         * 1/ (1 + exp(-x)) 
         * */

        return fllh + 0.5 * p0 - M_LN2 - lcosh(0.5 * p0) - log(t0);
    }


    fllh = params.early.lpdf(t, kp, tp);
    fllh += params.stop.lsf(t, ks, ts);
    
    fllh += (a == PROSACCADE ? LN_P_SERIA_EARLY_PRO : LN_P_SERIA_EARLY_ANTI);

    if ( t > da )
    {
        fllh += params.late.lsf(t - da, kl, tl);
        fllh += params.anti.lsf(t - da, ka, ta);

        // For optimization
        if ( cumint == CUMINT_NO_INIT )
        {
            /* Numerical weird case. */
            if ( t <= ZERO_DISTS )
            {
                cumint = 0;
            } else
            {
                cumint = params.inhibition_race(ZERO_DISTS, t, kp, ks, tp, ts);
            }
        }
        sllh = log(cumint + params.early.sf(t, kp, tp));
   
        switch ( a )
        {
        case PROSACCADE:
                sllh += params.late.lpdf(t - da, kl, tl) +
                    params.anti.lsf(t - da, ka, ta);
            break;
        case ANTISACCADE:
                sllh += params.anti.lpdf(t - da, ka, ta) +
                    params.late.lsf(t - da, kl, tl);
            break;
        }
    
    }

    p0 = -0.5 * p0 - M_LN2 - lcosh(0.5 * p0);

    // Perform this operation very carefully by taking out the largest 
    // component out of the log. Then perform a log( exp () + 1)
    if ( fllh >= sllh )
        fllh = p0 + fllh + log1p(exp(sllh - fllh));
    else
        fllh = p0 + sllh + log1p(exp(fllh - sllh)); 
    
    // Guard against odd values
    if ( fllh == GSL_POSINF )
        fllh = GSL_NAN;

	return fllh;
}

/* Probability of an early prosaccade. */
double
seria_early_llh_abstract(double t, int a, SERIA_PARAMETERS params)
{

    double t0 = params.t0;
    double p0 = params.p0;

    double kp = params.kp;
    double tp = params.tp;
    double ka = params.ka;
    double ta = params.ta;
    double kl = params.kl;
    double tl = params.tl;
    double ks = params.ks;
    double ts = params.ts;

    double da = params.da;

    double fllh = GSL_NEGINF;

    t -= t0;

    // Take care of the outliers
    if ( t < 0 )
    {
        fllh = LN_P_OUTLIER_PRO;
        return fllh + 0.5 * p0 - M_LN2 - lcosh(0.5 * p0) - log(t0);
    }


    fllh = params.early.lpdf(t, kp, tp);
    fllh += params.stop.lsf(t, ks, ts);
   
    if ( t > da )
    {
        fllh += params.late.lsf(t - da, kl, tl);
        fllh += params.anti.lsf(t - da, ka, ta);    
    }
    
    p0 = -0.5 * p0 - M_LN2 - lcosh(0.5 * p0);

    /* Account for the lost mass of the early outliers. */
    fllh = p0 + fllh; 
   

    // Guard against odd values
    if ( fllh == GSL_POSINF )
        fllh = GSL_NAN;

    /* For consistency. There is a tiny probability that early reactions
     * are antisaccades. */

    fllh += (a == PROSACCADE ? LN_P_SERIA_EARLY_PRO : LN_P_SERIA_EARLY_ANTI);

	return fllh;

}

double
seria_late_llh_abstract(double t, int a, SERIA_PARAMETERS params)
{

    double t0 = params.t0;
    double p0 = params.p0;

    double kp = params.kp;
    double tp = params.tp;
    double ka = params.ka;
    double ta = params.ta;
    double kl = params.kl;
    double tl = params.tl;
    double ks = params.ks;
    double ts = params.ts;

    double da = params.da;

    double fllh = GSL_NEGINF;

    double cumint = params.cumint;

    t -= t0;

    // Not a late reaction 
    if ( t < da )
    {
        return GSL_NEGINF;
    }

    if ( t > da )
    {
        /*
        // For optimization
        if ( cumint == CUMINT_NO_INIT )
            cumint = params.inhibition_race(ZERO_DISTS, t, kp, ks, tp, ts);

        fllh = log(cumint + params.early.sf(t, kp, tp));
        */
        switch ( a )
        {
        case PROSACCADE:
                fllh = params.late.lpdf(t - da, kl, tl) +
                    params.anti.lsf(t - da, ka, ta);
            break;
        case ANTISACCADE:
                fllh = params.anti.lpdf(t - da, ka, ta) +
                    params.late.lsf(t - da, kl, tl);
            break;
        }
    
    }

    /*
    p0 = -0.5 * p0 - M_LN2 - lcosh(0.5 * p0);
    fllh += p0; 
    */  
    // Guard against odd values
    if ( fllh == GSL_POSINF )
        fllh = GSL_NAN;

	return fllh;
}

