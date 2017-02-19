/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */

#include "antisaccades.h"

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

    double lx = sqrt(lambda / x);
    double xm = x / mu;
    double f;

    //gsl_sf_result gr;

    f = gsl_cdf_ugaussian_P(lx * (xm - 1.0));

    f += exp(2.0 * lambda / mu) * gsl_cdf_ugaussian_P(-lx * (xm + 1.0));

    return f;

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
wald_gslint(double x, double mu1, double mu2, double sig1, double sig2)
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
       
    gsl_integration_qng(&af, 0.00001, x, 0.001, 0.01, &v, &aberr, &neval);

    return v;
}



double
invgamma_lpdf(double x, double k, double t)
{
    // Gamma log pdf

    gsl_sf_result gr;

    SOONER_GSLERROR( x < 0 );
    SOONER_GSLERROR( k < 0 );
    SOONER_GSLERROR( t < 0 );

    SOONER_GSLERROR( gsl_sf_lngamma_e( k, &gr ) );

    return -gr.val - ((k + 1) * log(x)) - (t/x ) + (k * log(t));
}


double
later_lpdf(double x, double mu, double sigma)
{
    
    double tt = ((( 1 / x) - mu)/sigma);

    return -2 * log(x) - 0.5 * tt * tt - 0.5 * LOG2PI - 
        log( gsl_cdf_gaussian_Q(-mu, sigma) ) - log(sigma);

}

double
later_pdf(double x, double mu, double sigma)
{

    double tt = ((( 1 / x) - mu)/sigma);

    return exp(-0.5 * tt * tt )/(gsl_cdf_gaussian_Q(-mu, sigma) 
        * sigma * SQRT2PI * x * x);

}

double
later_cdf(double x, double mu, double sigma)
{

    return gsl_cdf_gaussian_Q(1/x - mu, sigma) / 
        gsl_cdf_gaussian_Q(-mu, sigma); 

}

double 
gamma_lpdf(double x, double k, double t)
{
    // Gamma log pdf

    gsl_sf_result gr;

    SOONER_GSLERROR( x < 0 );
    SOONER_GSLERROR( k < 0 );
    SOONER_GSLERROR( t < 0 );

    SOONER_GSLERROR( gsl_sf_lngamma_e( k, &gr ) );

    return -gr.val + (k - 1) * log(x) - x/t - k * log(t);
}


double 
lognorm_lpdf(double x, double mu, double sigma)
{

    double tv = (log(x) - mu)/sigma;
    
    return - log(x * sigma) - 0.5 * LOG2PI - 0.5 * tv * tv;
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
    double nv = pars[4];
    double r;

    gsl_sf_result gr;

    SOONER_GSLERROR( gsl_sf_gamma_inc_P_e(b, x / c1, &gr) );
    r = pow(x, a - 1) * exp(-x / c0) * gr.val;

    r *= nv;

    return r;
}

double
ngamma_gslint(double x, double a, double b, double c0, double c1)
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
    pars[4] = gr.val * pow(c0, -a);

    af.function = &dngamma;
    af.params = pars;
    gsl_integration_qng(&af, 0.00001, x, 0.001, 0.01, &v, &aberr, &neval);

    return v;
}

    
double
dninvgamma(double x, void *vpars)
{
    double *pars = (double *) vpars;
    double a = pars[0];
    double b = pars[1];
    double c0 = pars[2];
    double c1 = pars[3];
    double nv = pars[4];
    double r;

    gsl_sf_result gr;

    SOONER_GSLERROR( gsl_sf_gamma_inc_Q_e(b, c1 / x, &gr) );
    r = nv * pow(x, -a - 1) * exp(-c0 / x) * gr.val;

    return r;
}

double
ninvgamma_gslint(double x, double a, double b, double c0, double c1)
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
    pars[4] = gr.val * pow(c0, a);

    af.function = &dninvgamma;
    af.params = pars;
    gsl_integration_qng(&af, 0.00001, x, 0.001, 0.01, &v, &aberr, &neval);

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
    	
    r = gsl_ran_lognormal_pdf(x, mu1, sig1) * 
        gsl_cdf_lognormal_P(x, mu2, sig2);

    return r;
}

double
nlognorm_gslint(double x, double mu1, double mu2, double sig1, double sig2)
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
    gsl_integration_qng(&af, 0.00001, x, 0.001, 0.01, &v, &aberr, &neval);

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
    double r;
    
    r = later_pdf(x, mu1, sig1) * later_cdf(x, mu2, sig2);
    
    return r;
}

double
nlater_gslint(double x, double mu1, double mu2, double sig1, double sig2)
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
       
    gsl_integration_qng(&af, 0.00001, x, 0.001, 0.01, &v, &aberr, &neval);

    return v;
}


