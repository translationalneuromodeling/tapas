/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */


#include "antisaccades.h"

double 
prosa_llh_gamma(double t, int a, PROSA_PARAMETERS params)
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

    double llh;

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

    switch ( a ){
        case PROSACCADE:
            llh = gamma_lpdf(t, kp, tp) +  
                log(gsl_sf_gamma_inc_Q(ks, t/ts ));
            // If the saccade happened before the antisaccade delay, the 
            // probability of the antisaccade arriving after one is one and 
            // thus doesn't enter the expresion.
            if ( t > da )
                llh += log(gsl_sf_gamma_inc_Q(ka, (t - da)/ta ));
            break;
        case ANTISACCADE:

            // a > p > s
            
            if ( t < da )
            {
                llh = -INFINITY;
            } else
            {
                llh = ngamma_gslint(t, kp, ks, tp, ts);
                llh += gsl_sf_gamma_inc_Q(kp, t /tp );
                llh = log(llh);

                llh += gamma_lpdf(t - da, ka, ta);
            }
            break;
    }

    return log( 1 - p0 ) + llh;
}

double 
prosa_llh_invgamma(double t, int a, PROSA_PARAMETERS params)
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

    double llh;

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

    switch ( a ){
        case PROSACCADE:

            llh = invgamma_lpdf(t, kp, tp) +  
                log(gsl_sf_gamma_inc_P(ks, ts/t ));
            // If the saccade happened before the antisaccade delay, the 
            // probability of the antisaccade arriving after one is one and 
            // thus doesn't enter the expresion.
            if ( t > da )
                llh += log(gsl_sf_gamma_inc_P(ka, ta/(t - da) ));
            break;
        case ANTISACCADE:
            // a > p > s
            if ( t < da )
            {
                llh = -INFINITY;
            }  else {
                llh = ninvgamma_gslint(t, kp, ks, tp, ts);
                llh += gsl_sf_gamma_inc_P(kp, tp / t );
                llh = log(llh);

                llh += invgamma_lpdf(t - da, ka, ta);
            }
            break;
    }

    return log( 1 - p0 ) + llh;
}

double 
prosa_llh_mixedgamma(double t, int a, PROSA_PARAMETERS params)
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
    double llh;
    
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

    switch ( a ){
        case PROSACCADE:
            llh = invgamma_lpdf(t, kp, tp) +  
                log(gsl_sf_gamma_inc_P(ks, ts / t ));
            // If the saccade happened before the antisaccade delay, the 
            // probability of the antisaccade arriving after one is one and 
            // thus doesn't enter the expresion.
            if ( t > da )
                llh += log(gsl_sf_gamma_inc_Q(ka, (t - da)/ta ) );
            break;
        case ANTISACCADE:
            // a > p > s
            if ( t <= da )
            {
               llh = -INFINITY;
            } 
            else 
            { 
                llh = ninvgamma_gslint(t, kp, ks, tp, ts);
                llh += gsl_sf_gamma_inc_P(kp, tp / t );
                llh = log(llh);

                llh += gamma_lpdf((t - da), ka, ta);
            }
            break;
    }

    return log(1 - p0) + llh;
}

// From here one is code included by Jakob for the lognormal model.
double 
prosa_llh_lognorm(double t, int a, PROSA_PARAMETERS params) 
{
    double t0 = params.t0;
    double p0 = params.p0;

    double mup = params.kp;
    double sigp = params.tp;
    double mua = params.ka;
    double siga = params.ta;
    double mus = params.ks;
    double sigs = params.ts;

    double da = params.da;
    double llh;

    t -= t0;
    // Take care of the outliers
    if ( t < 0 )
    {
        switch ( a ){
            case PROSACCADE:
                return LN_P_OUTLIER_PRO + log(p0 / t0 );
                break;
            case ANTISACCADE:
                return LN_P_OUTLIER_ANTI + log(p0 / t0);
                break;
        }
    }

    switch ( a ){
        case PROSACCADE:

            llh = lognorm_lpdf(t, mup, sigp) +  
                log(gsl_cdf_lognormal_Q(t, mus, sigs));
            // If the saccade happened before the antisaccade delay, the 
            // probability of the antisaccade arriving after one is one and 
            // thus doesn't enter the expresion.
            if ( t > da )
                llh += log(gsl_cdf_lognormal_Q(t - da, mua, siga));
            break;

        case ANTISACCADE:
            // a > p > s
            if  ( t < da )
               llh = -INFINITY;
            else 
            { 
                llh = nlognorm_gslint(t, mup, mus, sigp, sigs);
                llh += gsl_cdf_lognormal_Q(t, mup, sigp );
                llh = log(llh);

                llh += lognorm_lpdf(t - da, mua, siga);
            }
            break;
    }
    
    return log(1 - p0) + llh;
}

double 
prosa_llh_later(double t, int a, PROSA_PARAMETERS params)
{
    double t0 = params.t0;
    double p0 = params.p0;

    double mup = params.kp;
    double sigp = params.tp;
    double mua = params.ka;
    double siga = params.ta;
    double mus = params.ks;
    double sigs = params.ts;

    double da = params.da;
    double llh = 0;
    
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

    switch ( a ){
        case PROSACCADE:
            llh += later_lpdf(t, mup, sigp);
            llh += log(1 - later_cdf(t, mus, sigs));

            // If the saccade happened before the antisaccade delay, the 
            // probability of the antisaccade arriving after one is one and 
            // thus doesn't enter the expresion.
            if ( t > da )
            {
                llh += log(1 - later_cdf(t - da, mua, siga));
            }
            break;
        case ANTISACCADE:

            if ( t <= da )
                llh = -INFINITY;
            else
            {
                llh = nlater_gslint(t, mup, mus, sigp, sigs);
                llh += 1 - later_cdf(t, mup, sigp); 

                llh = log(llh);
     
                llh += later_lpdf(t - da, mua, siga);
                // Normalization constant
            }
            break;
    }
    return log(1 - p0) + llh;
}

double 
prosa_llh_wald(double t, int a, PROSA_PARAMETERS params)
{
    double t0 = params.t0;
    double p0 = params.p0;

    double mup = params.kp;
    double sigp = params.tp;
    double mua = params.ka;
    double siga = params.ta;
    double mus = params.ks;
    double sigs = params.ts;

    double da = params.da;
    double llh = 0;
    
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

    switch ( a ){
        case PROSACCADE:
            llh += wald_lpdf(t, mup, sigp);
            llh += log(1 - wald_cdf(t, mus, sigs));

            // If the saccade happened before the antisaccade delay, the 
            // probability of the antisaccade arriving after one is one and 
            // thus doesn't enter the expresion.
            if ( t > da )
            {
                llh += log(1 - wald_cdf(t - da, mua, siga));
            }
            break;
        case ANTISACCADE:

            if ( t <= da )
                llh = -INFINITY;
            else
            {
                llh = wald_gslint(t, mup, mus, sigp, sigs);
                llh += 1 - wald_cdf(t, mup, sigp); 

                llh = log(llh);
     
                llh += wald_lpdf(t - da, mua, siga);
                // Normalization constant
            }
            break;
    }
    return log(1 - p0) + llh;
}


