/* aponteeduardo@gmail.com */
/* copyright (C) 2016 */

#include "antisaccades.h"

double
dora_llh_abstract(double t, int a, DORA_PARAMETERS params)
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

    double fllh = 0;
    double sllh = -INFINITY;

    double cumint = params.cumint;

    t -= t0;

    // Take care of the outliers
    if ( t < 0 )
    {
        switch ( a ){
            case PROSACCADE:
                return LN_P_OUTLIER_PRO +
                    (p0/2 - M_LN2 - log(cosh(p0/2)) - log(t0));
                break;
            case ANTISACCADE:
                return LN_P_OUTLIER_ANTI +
                    (p0/2 - M_LN2 - log(cosh(p0/2)) - log(t0));
                break;
        }
    }

    fllh = params.early.lpdf(t, kp, tp);
    fllh += params.stop.lsf(t, ks, ts);

    if ( t > da )
    {
        fllh += params.late.lsf(t - da, kl, tl);
        fllh += params.anti.lsf(t - da, ka, ta);
    }

    // For optimization
    if ( cumint == CUMINT_NO_INIT )
        cumint = params.inhibition_race(ZERO_DISTS, t, kp, ks, tp, ts);

	switch ( a )
	{
    case PROSACCADE:
        // Account for a late prosaccade
        if ( t > da )
        {
            sllh = params.late.lpdf(t - da, kl, tl) +
                params.anti.lsf(t - da, ka, ta) +
                log(cumint + params.stop.sf(t, ks, ts));
        }
        fllh += log(0.999); 
        break;
    case ANTISACCADE:
        if ( t > da )
        {
            sllh = params.anti.lpdf(t - da, kl, tl) +
                params.late.lsf(t - da, ka, ta) +
                log(cumint + params.stop.sf(t, ks, ts));
        }
        fllh += log(0.001);
        break;
	}

    p0 = -p0/2 - M_LN2 - log(cosh(p0/2));

    // Perform this operation very carefully by taking out the largest 
    // component out of the log. Then perform a log( exp () + 1)
    if ( fllh >= sllh )
        fllh = p0 + fllh + log1p(exp(sllh - fllh));
    else
        fllh = p0 + sllh + log1p(exp(fllh - sllh)); 
    
    // Guard against odd values
    if ( fllh == INFINITY )
        fllh = NAN;

	return fllh;
}

double
dora_llh_gamma(double t, int a, DORA_PARAMETERS params)
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

    double fllh = -INFINITY;
    double sllh = -INFINITY;

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

    fllh = gamma_lpdf(t, kp, tp);
    fllh += gamma_lsf(t, ks, ts);

    if ( t > da )
    {
        fllh += gamma_lsf(t - da, ka, ta);
        fllh += gamma_lsf(t - da, kl, tl);
    }

    // For optimization
    if ( cumint == CUMINT_NO_INIT )
        cumint = ngamma_gslint(ZERO_DISTS, t, kp, ks, tp, ts);

	switch ( a )
	{
    case PROSACCADE:
        // Account for a late prosaccade
        if ( t > da )
        {
            sllh = gamma_lpdf(t - da, kl, tl) 
                + gamma_lsf(t - da, ka, ta) 
                + log(cumint + gamma_sf(t, kp, tp)); 
        }
        fllh += log(0.999); 
        break;
    case ANTISACCADE:
        if ( t > da )
        {
            sllh = gamma_lpdf(t - da, ka, ta) 
                + gamma_lsf(t - da, kl, tl) 
                + log(cumint + gamma_sf(t, kp, tp)); 
        }
        fllh += log(0.001);
        break;
	}

    p0 = -0.5 * p0 - M_LN2 - lcosh(0.5 * p0);

    // Perform this operation very carefully by taking out the largest 
    // component out of the log. Then perform a log( exp () + 1)
    if ( fllh >= sllh )
        fllh = p0 + fllh + log1p(exp(sllh - fllh));
    else
        fllh = p0 + sllh + log1p(exp(fllh - sllh)); 
    
    // Guard against odd values
    if ( fllh == INFINITY )
        fllh = NAN;

	return fllh;
}

double
dora_llh_mixedgamma(double t, int a, DORA_PARAMETERS params)
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

    double fllh = -INFINITY;
    double sllh = -INFINITY;

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

    fllh = invgamma_lpdf(t, kp, tp);
    fllh += invgamma_lsf(t, ks, ts);

    if ( t > da )
    {
        fllh += gamma_lsf(t - da, ka, ta);
        fllh += gamma_lsf(t - da, kl, tl);
    }

    // For optimization
    if ( cumint == CUMINT_NO_INIT )
        cumint = ninvgamma_gslint(ZERO_DISTS, t, kp, ks, tp, ts);

	switch ( a )
	{
    case PROSACCADE:
        // Account for a late prosaccade
        if ( t > da )
        {
            sllh = gamma_lpdf(t - da, kl, tl) 
                + gamma_lsf(t - da, ka, ta) 
                + log(cumint + invgamma_sf(t, kp, tp)); 
        }
        fllh += log(0.999); 
        break;
    case ANTISACCADE:
        if ( t > da )
        {
            sllh = gamma_lpdf(t - da, ka, ta) 
                + gamma_lsf(t - da, kl, tl) 
                + log(cumint + invgamma_sf(t, kp, tp)); 
        }
        fllh += log(0.001);
        break;
	}

    p0 = -0.5 * p0 - M_LN2 - lcosh(0.5 * p0);

    // Perform this operation very carefully by taking out the largest 
    // component out of the log. Then perform a log( exp () + 1)
    if ( fllh >= sllh )
        fllh = p0 + fllh + log1p(exp(sllh - fllh));
    else
        fllh = p0 + sllh + log1p(exp(fllh - sllh)); 
    
    // Guard against odd values
    if ( fllh == INFINITY )
        fllh = NAN;

	return fllh;
}

double
dora_llh_invgamma(double t, int a, DORA_PARAMETERS params)
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

    double fllh = -INFINITY;
    double sllh = -INFINITY;

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

    fllh = invgamma_lpdf(t, kp, tp);
    fllh += invgamma_lsf(t, ks, ts);

    if ( t > da )
    {
        fllh += invgamma_lsf(t - da, ka, ta);
        fllh += invgamma_lsf(t - da, kl, tl);
    }

    // For optimization
    if ( cumint == CUMINT_NO_INIT )
        cumint = ninvgamma_gslint(ZERO_DISTS, t, kp, ks, tp, ts);

	switch ( a )
	{
    case PROSACCADE:
        // Account for a late prosaccade
        if ( t > da )
        {
            sllh = invgamma_lpdf(t - da, kl, tl) 
                + invgamma_lsf(t - da, ka, ta) 
                + log(cumint + invgamma_sf(t, kp, tp)); 
        }
        fllh += log(0.999); 
        break;
    case ANTISACCADE:
        if ( t > da )
        {
            sllh = invgamma_lpdf(t - da, ka, ta) 
                + invgamma_lsf(t - da, kl, tl) 
                + log(cumint + invgamma_sf(t, kp, tp)); 
        }
        fllh += log(0.001);
        break;
	}

    p0 = -0.5 * p0 - M_LN2 - lcosh(0.5 *  p0);

    // Perform this operation very carefully by taking out the largest 
    // component out of the log. Then perform a log( exp () + 1)
    if ( fllh >= sllh )
        fllh = p0 + fllh + log1p(exp(sllh - fllh));
    else
        fllh = p0 + sllh + log1p(exp(fllh - sllh)); 
    
    // Guard against odd values
    if ( fllh == INFINITY )
        fllh = NAN;

	return fllh;
}

double
dora_llh_later(double t, int a, DORA_PARAMETERS params)
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

    double fllh = -INFINITY;
    double sllh = -INFINITY;

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

    fllh = later_lpdf(t, kp, tp);
    fllh += later_lsf(t, ks, ts);

    if ( t > da )
    {
        fllh += later_lsf(t - da, ka, ta);
        fllh += later_lsf(t - da, kl, tl);
    }

    // For optimization
    if ( cumint == CUMINT_NO_INIT )
        cumint = nlater_gslint(ZERO_DISTS, t, kp, ks, tp, ts);

	switch ( a )
	{
    case PROSACCADE:
        // Account for a late prosaccade
        if ( t > da )
        {
            sllh = later_lpdf(t - da, kl, tl) 
                + later_lsf(t - da, ka, ta) 
                + log(cumint + later_sf(t, kp, tp)); 
        }
        fllh += log(0.999); 
        break;
    case ANTISACCADE:
        if ( t > da )
        {
            sllh = later_lpdf(t - da, ka, ta) 
                + later_lsf(t - da, kl, tl) 
                + log(cumint + later_sf(t, kp, tp)); 
        }
        fllh += log(0.001);
        break;
	}

    p0 = -0.5 * p0 - M_LN2 - lcosh(0.5 * p0);

    // Perform this operation very carefully by taking out the largest 
    // component out of the log. Then perform a log( exp () + 1)
    if ( fllh >= sllh )
        fllh = p0 + fllh + log1p(exp(sllh - fllh));
    else
        fllh = p0 + sllh + log1p(exp(fllh - sllh)); 
    
    // Guard against odd values
    if ( fllh == INFINITY )
        fllh = NAN;

	return fllh;
}

double
dora_llh_lognorm(double t, int a, DORA_PARAMETERS params)
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

    double fllh = 0;
    double sllh = -INFINITY;

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

    fllh = lognorm_lpdf(t, kp, tp);
    fllh += lognorm_lsf(t, ks, ts);

    if ( t > da )
    {
        fllh += lognorm_lsf(t - da, ka, ta);
        fllh += lognorm_lsf(t - da, kl, tl);
    }

    // For optimization
    if ( cumint == CUMINT_NO_INIT )
        cumint = nlognorm_gslint(ZERO_DISTS, t, kp, ks, tp, ts);

	switch ( a )
	{
    case PROSACCADE:
        // Account for a late prosaccade
        if ( t > da )
        {
            sllh = lognorm_lpdf(t - da, kl, tl) 
                + lognorm_lsf(t - da, ka, ta) 
                + log(cumint + lognorm_sf(t, kp, tp)); 
        }
        fllh += log(0.999); 
        break;
    case ANTISACCADE:
        if ( t > da )
        {
            sllh = lognorm_lpdf(t - da, ka, ta) 
                + lognorm_lsf(t - da, kl, tl) 
                + log(cumint + lognorm_sf(t, kp, tp)); 
        }
        fllh += log(0.001);
        break;
	}

    p0 = -0.5 * p0 - M_LN2 - lcosh(0.5 * p0);

    // Perform this operation very carefully by taking out the largest 
    // component out of the log. Then perform a log( exp () + 1)
    if ( fllh >= sllh )
        fllh = p0 + fllh + log1p(exp(sllh - fllh));
    else
        fllh = p0 + sllh + log1p(exp(fllh - sllh)); 
    
    // Guard against odd values
    if ( fllh == INFINITY )
        fllh = NAN;

	return fllh;
}


double
dora_llh_wald(double t, int a, DORA_PARAMETERS params)
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

    double fllh = 0;
    double sllh = -INFINITY;

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

    fllh = wald_lpdf(t, kp, tp);
    fllh += wald_lsf(t, ks, ts);

    if ( t > da )
    {
        fllh += wald_lsf(t - da, ka, ta);
        fllh += wald_lsf(t - da, kl, tl);
    }

    // For optimization
    if ( cumint == CUMINT_NO_INIT )
        cumint = nwald_gslint(ZERO_DISTS, t, kp, ks, tp, ts);

	switch ( a )
	{
    case PROSACCADE:
        // Account for a late prosaccade
        if ( t > da )
        {
            sllh = wald_lpdf(t - da, kl, tl) 
                + wald_lsf(t - da, ka, ta) 
                + log(cumint + wald_sf(t, kp, tp)); 
        }
        fllh += log(0.999); 
        break;
    case ANTISACCADE:
        if ( t > da )
        {
            sllh = wald_lpdf(t - da, ka, ta) 
                + wald_lsf(t - da, kl, tl) 
                + log(cumint + wald_sf(t, kp, tp)); 
        }
        fllh += log(0.001);
        break;
	}

    p0 = -0.5 * p0 - M_LN2 - lcosh(0.5 * p0);

    // Perform this operation very carefully by taking out the largest 
    // component out of the log. Then perform a log( exp () + 1)
    if ( fllh >= sllh )
        fllh = p0 + fllh + log1p(exp(sllh - fllh));
    else
        fllh = p0 + sllh + log1p(exp(fllh - sllh)); 
    
    // Guard against odd values
    if ( fllh == INFINITY )
        fllh = NAN;

	return fllh;
}
