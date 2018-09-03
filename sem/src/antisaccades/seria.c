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
            cumint = params.inhibition_race(ZERO_DISTS, t, kp, ks, tp, ts);

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

        return fllh; 
    }

    // There is a fixed value for antisaccades
    if ( a == ANTISACCADE )
        return LN_P_SERIA_EARLY_ANTI;

    // The log probability of an early response is 1.0
    if ( t < da )
        return 0.0;

    // The saccade is above the late units delay
    
    fllh = params.early.lpdf(t, kp, tp);
    fllh += params.stop.lsf(t, ks, ts);
    fllh += params.late.lsf(t - da, kl, tl);

    // For optimization
    if ( cumint == CUMINT_NO_INIT )
        cumint = params.inhibition_race(ZERO_DISTS, t, kp, ks, tp, ts);

    fllh = -log1p((cumint + params.early.sf(t, kp, tp)) * exp(
        params.late.lpdf(t - da, kl, tl) - fllh));

    // Guard against odd values
    if ( fllh == GSL_POSINF )
        fllh = GSL_NAN;

	return fllh;
}

