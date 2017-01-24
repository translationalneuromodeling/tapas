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

    double da = params.da;

    double fllh = 0;
    double sllh = 0;

    t -= t0;

    // Take care of the outliers
    if ( t < 0 )
    {
        switch ( a ){
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

    fllh = gamma_lpdf(t, kp, tp); 
    fllh += log(gsl_sf_gamma_inc_Q(ks, t/ts ));
    
    if ( t > da )
        fllh += log(gsl_sf_gamma_inc_Q(ka, (t - da)/ta )); 
   
    // a > p > s
    
    if ( t < da )
    {
        sllh = -INFINITY;
    } else {
        sllh = ngamma_gslint(t, kp, ks, tp, ts);

        // p > a ^ s > a

        sllh += gsl_sf_gamma_inc_Q(kp, t /tp );
        sllh = log(sllh);

        sllh += gamma_lpdf(t - da, ka, ta);
    }

    return log( ( 1 - p0 ) * (pp * exp(fllh) + ap * exp(sllh)));
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

    double fllh = 0;
    double sllh = 0;

    // Take care of the very early trials.
    t -= t0;

    // Take care of the outliers
    if ( t < 0 )
    {
        switch ( a ){
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
        sllh = ninvgamma_gslint(t, kp, ks, tp, ts);
        sllh += gsl_sf_gamma_inc_P(kp, tp / t ); 
        sllh = log(sllh);

        sllh += invgamma_lpdf(t - da, ka, ta);
    }

    return log( ( 1 - p0 ) * (pp * exp(fllh) + ap * exp(sllh))); 
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

    double fllh = 0;
    double sllh = 0;

    t -= t0;

    // Take care of the outliers
    if ( t < 0 )
    {
        switch ( a ){
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

    fllh = invgamma_lpdf(t, kp, tp);
    fllh += log(gsl_sf_gamma_inc_P(ks, ts / t));

    if ( t > da ) 
        fllh += log(gsl_sf_gamma_inc_Q(ka, (t - da)/ta ) );

    // a > p > s
    if ( t < da )
    {
        sllh = -INFINITY;
    } else {
     
        sllh += ninvgamma_gslint(t, kp, ks, tp, ts);
        sllh += gsl_sf_gamma_inc_P(kp, tp/t ); 
        sllh = log(sllh);

        sllh += gamma_lpdf(t - da, ka, ta);
    }

    return log( ( 1 - p0 ) * (pp * exp(fllh) + ap * exp(sllh)));
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

    double da = params.da;

    double fllh = 0;
    double sllh = 0;

    t -= t0;
    // Take care of the outliers
    if ( t < 0 )
    {
        switch ( a ){
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
        sllh = nlognorm_gslint(t, mup, mus, sigp, sigs);
        sllh += gsl_cdf_lognormal_Q(t, mup, sigp); 
        sllh = log(sllh);

        sllh += lognorm_lpdf(t - da, mua, siga);
    }

    return log( ( 1 - p0 ) * (pp * exp(fllh) + ap * exp(sllh)));
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

    double da = params.da;

    double fllh = 0;
    double sllh = 0;

    t -= t0;
    // Take care of the outliers
    if ( t < 0 )
    {
        switch ( a ){
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

    fllh += later_lpdf(t, mup, sigp);
    fllh += log(1 - later_cdf(t, mus, sigs));

    // If the saccade happened before the antisaccade delay, the 
    // probability of the antisaccade arriving after one is one and 
    // thus doesn't enter the expresion.
    if ( t > da )
    {
        fllh += log(1 - later_cdf(t - da, mua, siga));
    }

    if ( t <= da )
        sllh = -INFINITY;
    else
    {
        sllh = nlater_gslint(t, mup, mus, sigp, sigs);
        sllh += 1 - later_cdf(t, mup, sigp);

        sllh = log(sllh);

        sllh += later_lpdf(t - da, mua, siga);
        // Normalization constant
    }

    return log( ( 1 - p0 ) * (pp * exp(fllh) + ap * exp(sllh) ));
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

    t -= t0;
    // Take care of the outliers
    if ( t < 0 )
    {
        switch ( a ){
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
        sllh = wald_gslint(t, mup, mus, sigp, sigs);
        sllh += (1 - wald_cdf(t, mup, sigp));
        sllh = log(sllh);

        sllh += wald_lpdf(t - da, mua, siga);
    }

    return log( ( 1 - p0 ) * (pp * exp(fllh) + ap * exp(sllh)));
}

