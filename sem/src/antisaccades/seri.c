/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */


#include "antisaccades.h"

double
seri_llh_gamma(double t, int a, SERI_PARAMETERS params)
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

    double cumint = params.cumint;
    double da = params.da;

    double fllh = 0;
    double sllh = 0;

    t -= t0;

    // Take care of the outliers
    if ( t < 0 )
    {
        switch ( a )
        {
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

    fllh = gamma_lpdf(t, kp, tp); 
    fllh += log(gsl_sf_gamma_inc_Q(ks, t / ts ));
    
    if ( t > da )
        fllh += log(gsl_sf_gamma_inc_Q(ka, (t - da) / ta )); 
   
    // a > p > s
    
    if ( t < da )
    {
        sllh = -INFINITY;
    } else 
    {
        // Check if the value has not been inititlized yet 
        if ( cumint == CUMINT_NO_INIT )
            cumint = ngamma_gslint(ZERO_DISTS, t, kp, ks, tp, ts);

        sllh = cumint; 
        sllh += gsl_sf_gamma_inc_Q(kp, t / tp);
        sllh = log(sllh);

        sllh += gamma_lpdf(t - da, ka, ta);
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

double 
seri_llh_invgamma(double t, int a, SERI_PARAMETERS params)
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

    fllh = invgamma_lpdf(t, kp, tp);
    fllh += log(gsl_sf_gamma_inc_P(ks, ts/t) );

    if ( t > da ) 
        fllh += log(gsl_sf_gamma_inc_P(ka, ta/ (t - da)) );

    // a > p > s
    if ( t <= da )
    {
        sllh = -INFINITY;
    } 
    else 
    { 
        // Check if the value has not been inititlized yet.
        if ( cumint == CUMINT_NO_INIT )
            cumint = ninvgamma_gslint(ZERO_DISTS, t, kp, ks, tp, ts);

        sllh = cumint; 
        sllh += gsl_sf_gamma_inc_P(kp, tp / t ); 
        sllh = log(sllh);

        sllh += invgamma_lpdf(t - da, ka, ta);
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

double 
seri_llh_mixedgamma(double t, int a, SERI_PARAMETERS params)
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


    fllh = invgamma_lpdf(t, kp, tp);
    fllh += log(gsl_sf_gamma_inc_P(ks, ts / t));

    if ( t > da ) 
        fllh += log(gsl_sf_gamma_inc_Q(ka, (t - da)/ta ) );

    // a > p > s
    if ( t < da )
    {
        sllh = -INFINITY;
    } else {
        if ( cumint == CUMINT_NO_INIT ) 
            cumint = ninvgamma_gslint(ZERO_DISTS, t, kp, ks, tp, ts);

        sllh = cumint;
        sllh += gsl_sf_gamma_inc_P(kp, tp/t ); 
        sllh = log(sllh);

        sllh += gamma_lpdf(t - da, ka, ta);
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

double 
seri_llh_lognorm(double t, int a, SERI_PARAMETERS params)
{
    double t0 = params.t0;
    double p0 = params.p0;

    double mup = params.kp;
    double sigp = params.tp;
    double mua = params.ka;
    double siga = params.ta;
    double mus = params.ks;
    double sigs = params.ts;

    double pp = params.pp;
    double ap = params.ap;

    double cumint = params.cumint;
    double da = params.da;

    double fllh = 0;
    double sllh = 0;

    t -= t0;

    // Take care of the outliers
    if ( t < 0 )
    {
        switch ( a )
        {
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
    fllh = lognorm_lpdf(t, mup, sigp); 
    fllh += log(gsl_cdf_lognormal_Q(t, mus, sigs));
    if ( t > da )
        fllh += log(gsl_cdf_lognormal_Q(t - da, mua, siga));

    // a > p > s

    if ( t < da )
    {
        sllh = -INFINITY;
    } else 
    {
        // Check if the value has not been inititlized yet. 
        if ( cumint == CUMINT_NO_INIT )
            cumint = nlognorm_gslint(ZERO_DISTS, t, mup, mus, sigp, sigs);
        
        sllh = cumint;
        sllh += gsl_cdf_lognormal_Q(t, mup, sigp); 
        sllh = log(sllh);
        sllh += lognorm_lpdf(t - da, mua, siga);
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

// From here one is code included by Dario for the later model.
double 
seri_llh_later(double t, int a, SERI_PARAMETERS params)
{
    double t0 = params.t0;
    double p0 = params.p0;
    
    double mup = params.kp;
    double sigp = params.tp;
    double mua = params.ka;
    double siga = params.ta;
    double mus = params.ks;
    double sigs = params.ts;

    double pp = params.pp;
    double ap = params.ap;

    double cumint = params.cumint;
    double da = params.da;

    double fllh = 0;
    double sllh = 0;

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
    
    fllh += later_lpdf(t, mup, sigp);
    fllh += later_lsf(t, mus, sigs);

    // If the saccade happened before the antisaccade delay, the 
    // probability of the antisaccade arriving after one is one and 
    // thus doesn't enter the expresion.
    if ( t > da )
    {
        fllh += later_lsf(t - da, mua, siga);
    }

    if ( t <= da )
        sllh = -INFINITY;
    else
    {
        // Check if the value has not been inititlized yet.
        if ( cumint == CUMINT_NO_INIT )
            cumint = nlater_gslint(ZERO_DISTS, t, mup, mus, sigp, sigs);
        sllh = cumint + later_sf(t, mup, sigp);
        sllh = log(sllh);
        sllh += later_lpdf(t - da, mua, siga);
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

double 
seri_llh_wald(double t, int a, SERI_PARAMETERS params)
{
    double t0 = params.t0;
    double p0 = params.p0;

    double mup = params.kp;
    double sigp = params.tp;
    double mua = params.ka;
    double siga = params.ta;
    double mus = params.ks;
    double sigs = params.ts;

    double pp = params.pp;
    double ap = params.ap;

    double da = params.da;

    double fllh = 0;
    double sllh = 0;

    double cumint = params.cumint;

    t -= t0;
    // Take care of the outliers
    if ( t < 0 )
    {
        switch ( a )
        {
            case PROSACCADE:
                return LN_P_OUTLIER_PRO + log(p0 / t0 );
                break;
            case ANTISACCADE:
                return LN_P_OUTLIER_ANTI + log(p0/t0);
                break;
        }
    }


    if ( a == PROSACCADE )
    {
        pp = 1 - pp;
        ap = 1 - ap;
    }

    fllh = wald_lpdf(t, mup, sigp); 
 
    fllh += log(1 - wald_cdf(t, mus, sigs));
    if ( t > da )
        fllh += log(1 - wald_cdf(t - da, mua, siga));

    // a > p > s

    if ( t < da )
    {
        sllh = -INFINITY;
    } else 
    {
        // Check if the value has not been inititlized yet.
        if ( cumint == CUMINT_NO_INIT )
            cumint = nwald_gslint(ZERO_DISTS, t, mup, mus, sigp, sigs);

        sllh = cumint;
        sllh += (1 - wald_cdf(t, mup, sigp));
        sllh = log(sllh);

        sllh += wald_lpdf(t - da, mua, siga);
    }

    fllh = log( ( 1 - p0 ) * (pp * exp(fllh) + ap * exp(sllh)));

    if ( fllh == INFINITY )
        fllh = NAN;

    return fllh;
}

