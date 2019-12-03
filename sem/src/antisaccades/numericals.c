/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */

#include "antisaccades.h"

#define INTSTEPS 8000

// Log of cosh

double
lcosh(double x)
{
    return gsl_sf_lncosh(x); 
}


double
wald_pdf(double x, double mu, double lambda)
{
    
    return sqrt(lambda / (2.0 * M_PI * x * x * x)) * 
        exp(- 0.5 * lambda * (x - mu) * (x - mu )/(mu * mu * x));
}

double
wald_lpdf(double x, double mu, double lambda)
{
    // Wald distribution log pdf 
    return 0.5 * ( log( lambda ) - (LOG2PI + 3 * log( x )) ) 
        - 0.5 * lambda * ( x - mu ) * (x - mu) / (mu * mu * x);
}

double
wald_cdf(double x, double mu, double lambda)
{
    SOONER_GSLERROR( x <= 0 );
    SOONER_GSLERROR( lambda <= 0 );
    SOONER_GSLERROR( mu <= 0 );
    //SOONER_GSLERROR( );

    double lx = sqrt(lambda / (2.0 * x));
    double xm = x / mu;
    double f;

    //gsl_sf_result gr;

    f = 1.0 + gsl_sf_erf(lx * (xm - 1.0));

    f += exp(2.0 * lambda / mu) * (1 - gsl_sf_erf(lx * (xm + 1.0)));

    return f * 0.5;

}

double
wald_lcdf(double x, double mu, double lambda)
{
    return log(wald_cdf(x, mu, lambda));
}


double
wald_sf(double x, double mu, double lambda)
{

    return 1 - wald_cdf(x, mu, lambda);
}

double
wald_lsf(double x, double mu, double lambda)
{

    return log1p(-wald_cdf(x, mu, lambda));
}

double
dt_wald(double x, void *vpars)
{
    double *pars = (double *) vpars;
    double mu1 = pars[0];
    double mu2 = pars[1];
    double sig1 = pars[2];
    double sig2 = pars[3];
    double r;
    
    r = wald_pdf(x, mu1, sig1) * wald_cdf(x, mu2, sig2);
    
    return r;
}

double
nwald_gslint(double t0, double x, double mu1, double mu2, double sig1, 
    double sig2)
{
    gsl_function af;

    double pars[4];
    double v;
    double aberr;
    size_t neval;

    pars[0] = mu1;
    pars[1] = mu2;
    pars[2] = sig1;
    pars[3] = sig2;

    af.function = &dt_wald;
    af.params = pars;
       
    gsl_integration_qng(&af, t0, x, SEM_TOL, SEM_RTOL, &v, &aberr, &neval);

    return v;
}

double
invgamma_lpdf(double x, double k, double t)
{
    // Gamma log pdf
    if ( k > SEM_GAMMA_MAX_SHAPE )
        return NAN;

    gsl_sf_result gr;

    SOONER_GSLERROR( x < 0 );
    SOONER_GSLERROR( k < 0 );
    SOONER_GSLERROR( t < 0 );

    SOONER_GSLERROR( gsl_sf_lngamma_e( k, &gr ) );

    return -gr.val - ((k + 1) * log(x)) - (t/x ) + (k * log(t));
}

double
invgamma_pdf(double x, double k, double t)
{

    return exp(invgamma_lpdf(x, k, t));
}

double
invgamma_cdf(double x, double k, double t)
{
    // Gamma log pdf
    if ( k > SEM_GAMMA_MAX_SHAPE )
        return NAN;

    return gsl_sf_gamma_inc_Q(k, t / x);

}

double
invgamma_lcdf(double x, double k, double t)
{
    // Gamma log pdf
    if ( k > SEM_GAMMA_MAX_SHAPE )
        return NAN;

    return log(gsl_sf_gamma_inc_Q(k, t / x));

}

double
invgamma_sf(double x, double k, double t)
{
    // Gamma log pdf
    if ( k > SEM_GAMMA_MAX_SHAPE )
        return NAN;

    return gsl_sf_gamma_inc_P(k, t / x); 
}

double
invgamma_lsf(double x, double k, double t)
{
    // Gamma log pdf
    if ( k > SEM_GAMMA_MAX_SHAPE )
        return NAN;

    return log(gsl_sf_gamma_inc_P(k, t / x)); 
}

double
later_lpdf(double x, double mu, double sigma)
{
    
    double tt = ( 1 - x * mu)/ (sigma * x);

    return -2 * log(x) - 0.5 * tt * tt - 0.5 * LOG2PI - 
        log( gsl_cdf_gaussian_Q(-mu, sigma) ) - log(sigma);

}

double
later_pdf(double x, double mu, double sigma)
{

    double tt = ( 1 - x * mu)/ (sigma * x);

    return exp(-0.5 * tt * tt - 2 * log(x) )/(gsl_cdf_gaussian_Q(-mu, sigma) 
        * sigma * SQRT2PI);

}

double
later_cdf(double x, double mu, double sigma)
{

    return gsl_cdf_gaussian_Q((1/x) - mu, sigma) / 
        gsl_cdf_gaussian_Q(-mu, sigma); 

}

double
later_lcdf(double x, double mu, double sigma)
{

    return log(gsl_cdf_gaussian_Q(1/x - mu, sigma)) - 
        log(gsl_cdf_gaussian_Q(-mu, sigma)); 

}

double
later_sf(double x, double mu, double sigma)
{

    return 1 - (gsl_cdf_gaussian_Q(1/x - mu, sigma) / 
        gsl_cdf_gaussian_Q(-mu, sigma)); 

}

double
later_lsf(double x, double mu, double sigma)
{
    return log(1 - (gsl_cdf_gaussian_Q(1/x - mu, sigma) / 
        gsl_cdf_gaussian_Q(-mu, sigma))); 

}
    
double 
gamma_lpdf(double x, double k, double t)
{
    // Gamma log pdf
    // Gamma log pdf
    if ( k > SEM_GAMMA_MAX_SHAPE )
        return NAN;

    gsl_sf_result gr;

    SOONER_GSLERROR( x < 0 );
    SOONER_GSLERROR( k < 0 );
    SOONER_GSLERROR( t < 0 );

    SOONER_GSLERROR( gsl_sf_lngamma_e( k, &gr ) );

    return -gr.val + (k - 1) * log(x) - x/t - k * log(t);
}

double
gamma_pdf(double x, double k, double t)
{

    return gamma_lpdf(x, k, t);
}

double
gamma_cdf(double x, double k, double t)
{
    // Gamma log pdf
    if ( k > SEM_GAMMA_MAX_SHAPE )
        return NAN;

    return gsl_sf_gamma_inc_P(k, x / t);
}

double
gamma_lcdf(double x, double k, double t)
{
    // Gamma log pdf
    if ( k > SEM_GAMMA_MAX_SHAPE )
        return NAN;

    return log(gsl_sf_gamma_inc_P(k, x / t));
}

double
gamma_sf(double x, double k, double t)
{
    // Gamma log pdf
    if ( k > SEM_GAMMA_MAX_SHAPE )
        return NAN;

    return gsl_sf_gamma_inc_Q(k, x / t);
}

double 
gamma_lsf(double x, double k, double t)
{
    // Gamma log pdf
    if ( k > SEM_GAMMA_MAX_SHAPE )
        return NAN;

    return log(gsl_sf_gamma_inc_Q(k, x / t));
}

double 
lognorm_lpdf(double x, double mu, double sigma)
{

    double tv = (log(x) - mu)/sigma;
  
    if ( sigma < 1e-2 )
        return NAN;

    // Provide a bit of numerical stability to the data. 
    return  - log(x * sigma) - 0.5 * LOG2PI - 0.5 * tv * tv;
}

double
lognorm_pdf(double x, double mu, double sigma)
{
    double prob = exp(lognorm_lpdf(x, mu, sigma));

    return prob;
}

double
lognorm_cdf(double x, double k, double t)
{
    if ( t < 1e-2 )
        return NAN;

    return gsl_cdf_lognormal_P(x, k, t);
}

double
lognorm_lcdf(double x, double k, double t)
{
    if ( t < 1e-2 )
        return NAN;

    return log(gsl_cdf_lognormal_P(x, k, t));
}

double
lognorm_sf(double x, double k, double t)
{
    if ( t < 1e-2 )
        return NAN;

    return gsl_cdf_lognormal_Q(x, k, t);
}

double
lognorm_lsf(double x, double k, double t)
{
    if ( t < 1e-2 )
        return NAN;

    return log(gsl_cdf_lognormal_Q(x, k, t));
}

double 
tngamma(double x, double a, double b, double c0, double c1)
{
    int i;
    gsl_sf_result gr;

    double l = 0;
    double fb = b;
    double cc1 = 1;

    c0 = 1/c0;
    c1 = 1/c1;

    for ( i = 0; i < NITER_NGAMMA; i++ )
    {
        SOONER_GSLERROR( gsl_sf_gamma_inc_P_e( a + b + i, (c0 + c1 ) * x, 
            &gr ) );
        l += cc1 *  gr.val / fb;
        fb *= (b + i + 1) * (c0 + c1);
        cc1 *= c1 * (a + b + i);
    } 

    SOONER_GSLERROR( gsl_sf_beta_e(a, b, &gr ) );
    
    return l * pow(c0 + c1, -(a + b)) * pow(c0, a) * pow(c1, b) / gr.val;
}


double 
ngamma(double x, double a, double b, double c0, double c1)
{
    int i;
    gsl_sf_result gr;

    double l;
    double fb = b;
    double cc1;
    double c;
    c0 = 1/c0;
    c1 = 1/c1;
    double lc0 = log(c0);
    double lc1 = log(c1);
    double lc0c1 = log(c0 + c1);


    c = 0;
    l = 0;
    fb = log(b);
    cc1 = 0;

    for ( i = 0; i < NITER_NGAMMA; i++ )
    {
        SOONER_GSLERROR( gsl_sf_gamma_inc_P_e(a + b + i, (c0 + c1) * x, &gr ) );

        l += exp(cc1 - fb) * gr.val;
        fb += log(b + i + 1) + lc0c1;
        cc1 += lc1 + log(a + b + i);
    } 


    SOONER_GSLERROR( gsl_sf_lnbeta_e( a, b, &gr ) );
    c -= gr.val;

    c += - (a + b) * lc0c1 + a * lc0 + b * lc1;

    return l * exp(c);
}



double 
tninvgamma(double x, double a, double b, double c0, double c1)
{
    int i;
    gsl_sf_result gr;
    double l = 0;
    double fb = b;
    double cc1 = 1;
    

    for ( i = 0; i < NITER_NGAMMA ; i++ )
    {
        SOONER_GSLERROR( gsl_sf_gamma_inc_Q_e( a + b + i, (c0 + c1) / x, 
            &gr) );

        l += cc1 *  gr.val / fb;
        fb *= (b + i + 1) * (c0 + c1);
        cc1 *= c1 * (a + b + i);
    } 

    SOONER_GSLERROR( gsl_sf_gammainv_e( a , &gr ) );
    l *= gr.val;
    SOONER_GSLERROR( gsl_sf_gammainv_e(b, &gr ) );
    l *= gr.val;
    SOONER_GSLERROR( gsl_sf_gamma_e(a + b, &gr ) );
    l *= gr.val;

    l *= pow(c0 + c1, -(a + b)) * pow(c0, a) * pow(c1, b);

    SOONER_GSLERROR( gsl_sf_gamma_inc_Q_e( a, c0 / x, &gr ) );

    l = gr.val - l;
    
    return l;
}

double 
ninvgamma(double x, double a, double b, double c0, double c1)
{
    int i;    
    gsl_sf_result gr;

    double l;
    double fb;
    double cc1;

    double lc0c1 = log(c0 + c1);
    double lc0 = log(c0);
    double lc1 = log(c1);

    double cc = - (a + b) * lc0c1 + a * lc0 + b * lc1;

    double unreg = 0;

    SOONER_GSLERROR( gsl_sf_lngamma_e( a + b, &gr ) );
    unreg += gr.val;

    SOONER_GSLERROR( gsl_sf_lngamma_e( a , &gr ) );
    unreg -= gr.val;

    SOONER_GSLERROR( gsl_sf_lngamma_e( b, &gr ) );
    unreg -= gr.val;

    l = 0;
    cc1 = 0;
    fb = log(b);

    for ( i = 0; i < NITER_NGAMMA + 20; i++ )
    {
        SOONER_GSLERROR( gsl_sf_gamma_inc_Q_e( a + b + i, 
            (c0 + c1) / x, &gr ) ); 
        l += exp(cc1 + cc - fb) * gr.val;
        cc1 += lc1 + log(a + b + i);
        fb += log((b + i + 1)) + lc0c1;
    } 

    SOONER_GSLERROR( gsl_sf_gamma_inc_Q_e( a, c0 / x, &gr ) );

    return gr.val - l * exp(unreg);
}


double 
ngamma_lexp(double x, double a, double b, double c0, double c1)
{

    double l = 0;
    double dt = x / INTSTEP;
    double ct = dt;
    gsl_sf_result gr;

    while ( ct < x - dt )
    {
        SOONER_GSLERROR( gsl_sf_gamma_inc_P_e(b, ct / c1, &gr) );
        l += pow(ct, a - 1) * exp(-ct / c0) * gr.val;
        ct += dt;
    }

    SOONER_GSLERROR( gsl_sf_gammainv_e(a, &gr) );

    l *= dt * gr.val * pow(c0, -a);

    return l;
}

double
dngamma(double x, void *vpars)
{
    double *pars = (double *) vpars;
    double a = pars[0];
    double b = pars[1];
    double c0 = pars[2];
    double c1 = pars[3];
    double r;

    gsl_sf_result gr;

    SOONER_GSLERROR( gsl_sf_gamma_inc_P_e(b, x / c1, &gr) );
    r = exp(-x / c0 + (a - 1) * log(x) + log(gr.val));

    return r;
}

double
ngamma_gslint(double t0, double x, double a, double b, double c0, double c1)
{
    gsl_function af;
    gsl_sf_result gr;

    double pars[5];
    double v;
    double aberr;
    size_t neval;

    SOONER_GSLERROR( gsl_sf_gammainv_e(a, &gr) );

    pars[0] = a;
    pars[1] = b;
    pars[2] = c0;
    pars[3] = c1;

    af.function = &dngamma;
    af.params = pars;
    SOONER_GSLERROR(gsl_integration_qng(&af, t0, x, SEM_TOL, SEM_RTOL, &v,
               &aberr, &neval));

    if (aberr > 0.001)
    {
        return GSL_NEGINF;
    }

    return v * gr.val * pow(c0, -a);
}

    
double
dninvgamma(double x, void *vpars)
{
    double *pars = (double *) vpars;
    double a = pars[0];
    double b = pars[1];
    double c0 = pars[2];
    double c1 = pars[3];
    double c2 = pars[4];
    double r;

    gsl_sf_result gr;

    SOONER_GSLERROR( gsl_sf_gamma_inc_Q_e(b, c1 / x, &gr) );
    r = exp(-c0 / x - (a + 1) * log(x) + c2) * gr.val;

    return r;
}

#define INTSTEPS 3000

double
ninvgamma_gslint(double xs, double xe, double a, double b, double c0, 
        double c1)
{
    // Zero mass if the start is bellow zero.
    if ( xe < 0.00001)
        return 0;
    gsl_function af;
    gsl_sf_result gr;

    double pars[5];
    double v;
    double aberr;
    size_t neval; 

    SOONER_GSLERROR( gsl_sf_lngamma_e(a, &gr) );

    pars[0] = a;
    pars[1] = b;
    pars[2] = c0;
    pars[3] = c1;
    pars[4] = - gr.val + log(c0) * a;


    af.function = &dninvgamma;
    af.params = pars;

    /*
    gsl_integration_workspace *wspace = 
        gsl_integration_workspace_alloc( INTSTEPS );

    gsl_integration_qag(
        &af,
        xs,
        xe,
        SEM_TOL,
        SEM_RTOL,
        INTSTEPS,
        GSL_INTEG_GAUSS61,
        wspace,
        &v,
        &aberr);

    gsl_integration_workspace_free(wspace);
    */

     SOONER_GSLERROR(gsl_integration_qng(&af, xs, xe, SEM_TOL, SEM_RTOL, &v,
                &aberr, &neval));

    if (aberr > 0.001)
    {
        return GSL_NAN;
    }

    return v;
}


double
ninvgamma_high_precision_gslint(double xs, double xe, double a, double b, double c0, 
        double c1)
{
    // Zero mass if the start is bellow zero.
    if ( xe < 0.00001)
        return 0;
    gsl_function af;
    gsl_sf_result gr;

    double pars[5];
    double v;
    double aberr;
    size_t neval; 

    SOONER_GSLERROR( gsl_sf_lngamma_e(a, &gr) );

    pars[0] = a;
    pars[1] = b;
    pars[2] = c0;
    pars[3] = c1;
    pars[4] = - gr.val + log(c0) * a;

    af.function = &dninvgamma;
    af.params = pars;
    
    gsl_integration_workspace *wspace =
        gsl_integration_workspace_alloc( INTSTEP );

    gsl_integration_qag(
            &af,
            xs,
            xe,
            SEM_TOL,
            SEM_RTOL,
            INTSTEPS,
            GSL_INTEG_GAUSS61,
            wspace,
            &v,
            &aberr);

    gsl_integration_workspace_free(wspace);

    if (aberr > 0.001)
    {
        return GSL_NEGINF;
    }

    return v;
}


double 
ninvgamma_lexp(double x, double a, double b, double c0, double c1)
{

    double l = 0;
    double dt = x / (INTSTEP + 150);
    double ct = dt;
    double nv;
    gsl_sf_result gr;

    SOONER_GSLERROR( gsl_sf_gammainv_e(a, &gr) );
    nv = gr.val;

    while ( ct < x - dt )
    {
        SOONER_GSLERROR( gsl_sf_gamma_inc_Q_e(b, c1 / ct, &gr) );
        l += nv * pow(ct, -a - 1) * exp(-c0 / ct) * gr.val;
        ct += dt;
    }

    l *= dt * pow(c0, a);
    
    return l;
}

double
dnlognorm(double x, void *vpars)
{
    double *pars = (double *) vpars;
    double mu1 = pars[0];
    double sig1 = pars[1];
    double mu2 = pars[2];
    double sig2 = pars[3];
    double r;
    	
    r = exp(
            lognorm_lpdf(x, mu1, sig1) +
        log(gsl_cdf_lognormal_P(x, mu2, sig2)));
    //r = gsl_cdf_lognormal_P(x, mu2, sig2);

    return r;
}

double
nlognorm_gslint(double t0, double x, double mu1, double mu2, double sig1, 
        double sig2)
{
    gsl_function af;

    double pars[4];
    double v;
    double aberr;
    size_t neval;

    pars[0] = mu1;
    pars[1] = sig1;
    pars[2] = mu2;
    pars[3] = sig2;

    af.function = &dnlognorm;
    af.params = pars;
    SOONER_GSLERROR(gsl_integration_qng(&af, t0, x, SEM_TOL, SEM_RTOL, 
                &v, &aberr, &neval));

    if (aberr > 0.001)
    {
        return GSL_NEGINF;
    }


    return v;
}


double
dnlater(double x, void *vpars)
{
    double *pars = (double *) vpars;
    double mu1 = pars[0];
    double mu2 = pars[1];
    double sig1 = pars[2];
    double sig2 = pars[3];
    double tt = ( 1 - x * mu1)/ (sig1 * x);

    return exp(-0.5 * tt * tt - 2 * log(x)) *
        gsl_cdf_gaussian_Q(1/x - mu2, sig2);
}

double
nlater_gslint(double t0, double x, double mu1, double mu2, double sig1,
       double sig2)
{
    gsl_function af;

    double pars[4];
    double v;
    double aberr;
    size_t neval;

    pars[0] = mu1;
    pars[1] = mu2;
    pars[2] = sig1;
    pars[3] = sig2;

    af.function = &dnlater;
    af.params = pars;
       
    SOONER_GSLERROR(gsl_integration_qng(&af, t0, x, SEM_TOL, SEM_RTOL, &v,
                &aberr, &neval));

    if (aberr > 0.001)
    {
        return GSL_NEGINF;
    }

    return v / ( gsl_cdf_gaussian_Q(-mu1, sig1) * sig1 * SQRT2PI * 
            gsl_cdf_gaussian_Q(-mu2, sig2));
}


