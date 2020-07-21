/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */


#include "antisaccades.h"

double
prosa_llh_abstract(double t, int a, PROSA_PARAMETERS params)
{
    double t0 = params.t0;
    double p0 = params.p0;

    double kp = params.kp;
    double tp = params.tp;
    double ka = params.ka;
    double ta = params.ta;
    double ks = params.ks;
    double ts = params.ts;

    double da = params.da;
    double cumint = params.cumint;

    double llh;

    t -= t0;

    // Take care of the outliers
    if ( t < 0 )
    {
        switch ( a ){
            case PROSACCADE:
                return LN_P_OUTLIER_PRO +
                    (0.5 * p0 - M_LN2 - lcosh(0.5 * p0) - log(t0));
                break;
            case ANTISACCADE:
                return LN_P_OUTLIER_ANTI +
                    (0.5 * p0 - M_LN2 - lcosh(0.5 * p0) - log(t0));
                break;
        }
    }  

    switch ( a )
    {
        case PROSACCADE:
            llh = params.early.lpdf(t, kp, tp);
            llh += params.stop.lsf(t, ks, ts);
                
            // If the saccade happened before the antisaccade delay, the 
            // probability of the antisaccade arriving after one is one and 
            // thus doesn't enter the expresion.
            if ( t > da )
                llh += params.anti.lsf(t - da, ka, ta); 
            break;
        case ANTISACCADE:
            if ( t < da )
            {
                llh = -INFINITY;
            } else
            {
                // Check if the value has not been inititlized yet.
                if ( cumint == CUMINT_NO_INIT )
                    cumint = params.inhibition_race(ZERO_DISTS, t, kp, ks,
                            tp, ts);

                llh = log(cumint + params.early.sf(t, kp, tp));
                llh += params.anti.lpdf(t - da, ka, ta);
            }
            break;
    }
    llh += -0.5 * p0 - M_LN2 - lcosh(-0.5 * p0);
    // Guard against odd values
    if ( llh == INFINITY )
        llh = NAN;

    return llh;
}
