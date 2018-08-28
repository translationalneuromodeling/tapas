/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */


#include "antisaccades.h"

double 
seri_llh_abstract(double t, int a, SERI_PARAMETERS params)
{
    double t0 = params.t0;
    double p0 = params.p0;

    double kp = params.kp;
    double tp = params.tp;
    double ka = params.ka;
    double ta = params.ta;
    double ks = params.ks;
    double ts = params.ts;

    double pp = params.pp;
    double ap = params.ap;

    double da = params.da;

    double cumint = params.cumint;

    double fllh = 0;
    double sllh = 0;

    // Take care of the very early trials.
    t -= t0;

    // Take care of the outliers
    if ( t < 0 )
    {
        switch ( a ){
            case PROSACCADE:
                return LN_P_OUTLIER_PRO + (p0/2  - M_LN2 -log(cosh(p0/2)) -
                        log(t0));
                break;
            case ANTISACCADE:
                return LN_P_OUTLIER_ANTI + (p0/2 - M_LN2 - log(cosh(p0/2)) - 
                        log(t0));
                break;
        }
    }

    switch ( a )
    {  
        case PROSACCADE:
            pp = -pp/2 - M_LN2 - log(cosh(pp/2));
            ap = -ap/2 - M_LN2 - log(cosh(ap/2));
            break;
        case ANTISACCADE:
            pp = pp/2 - M_LN2 - log(cosh(pp/2));
            ap = ap/2 - M_LN2 - log(cosh(ap/2));
            break;
    }
    
    fllh = params.early.lpdf(t, kp, tp);
    fllh += params.stop.lsf(t, ks, ts);

    if ( t > da ) 
        fllh += params.anti.lsf(t - da, ka, ta);

    // a > p > s
    if ( t <= da )
    {
        sllh = -INFINITY;
    } 
    else 
    { 
        // Check if the value has not been inititlized yet.
        if ( cumint == CUMINT_NO_INIT )
            cumint = params.inhibition_race(ZERO_DISTS, t, kp, ks, tp, ts);

        sllh = log(cumint + params.early.sf(t, kp, tp));
        sllh += params.anti.lpdf(t - da, ka, ta);
    }

    p0 = -p0/2 - M_LN2 - log(cosh(p0/2));

    fllh += pp;
    sllh += ap;
    
    // Perform this operation very carefully by taking out the largest 
    // component out of the log. Then perform a log( exp () + 1)
    if ( fllh >= sllh )
        fllh = p0 + fllh + log1p(exp(sllh - fllh));
    else
        fllh = p0 + sllh + log1p(exp(fllh - sllh));
    
    if ( fllh == INFINITY )
        fllh = NAN;

    return fllh;
}


