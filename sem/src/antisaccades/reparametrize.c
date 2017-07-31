/* aponteeduardo@gmail.com */
/* copyright (C) 2017 */

#include "antisaccades.h"

// This is done to reparametrize the code inside c as opposed to matlab


double
transform_mv_to_gamma_k(double mu, double sigma2)
{
    // Transform mean and variance to parameter k
    return mu * mu / sigma2;
}

double
transform_log_mv_to_gamma_k(double mu, double sigma2)
{
    // Transform mean and variance to parameter k
    return exp(2 * mu - sigma2);
}


double
transform_mv_to_gamma_t(double mu, double sigma2)
{
    // Transform mean and variance to parameter k
    return mu / sigma2;
}

double
transform_log_mv_to_gamma_t(double mu, double sigma2)                                                                                                         
{
    // Transform mean and variance to parameter k
    return exp(mu - sigma2);
}

double
transform_mv_to_invgamma_k(double mu, double sigma2)
{
    // Transform mean and variance to parameter k
    return mu * mu / sigma2 + 2;
}

double
transform_mv_to_invgamma_t(double mu, double sigma2)
{
    // Transform mean and variance to parameter k
    return 1 / (mu * (mu * mu / sigma2 + 1));
}

double
transform_log_mv_to_invgamma_k(double mu, double sigma2)
{
    // Transform mean and variance to parameter k
    return exp(2 * mu  - sigma2) + 2;
}

double
transform_log_mv_to_invgamma_t(double mu, double sigma2)
{
    // Transform mean and variance to parameter k
    return exp(- mu - log(exp(2 * mu - sigma2) + 1 ));

}

double
transform_inv_to_wald_l(double mu, double sigma2)
{
    // From the mean and variance of the reciprocal to the lambda parameter
    
    return 1.0/(sqrt(0.25 * mu * mu + sigma2) - 0.5 * mu);
}

double 
transform_inv_to_wald_mu(double mu, double sigma2)
{
    // From the mean and variance of the reciprocal to the mu parameter
    
   return 1.0/(1.5 * mu - sqrt(0.25 * mu * mu + sigma2));
}


int
reparametrize_seri_invgamma(const double *theta, SERI_PARAMETERS *stheta)
{

    stheta->kp = transform_log_mv_to_gamma_k(theta[0], theta[1]) + 2;
    stheta->tp = transform_log_mv_to_gamma_t(theta[0], theta[1]);

    stheta->ka = transform_log_mv_to_gamma_k(theta[2], theta[3]) + 2;
    stheta->ta = transform_log_mv_to_gamma_t(theta[2], theta[3]);

    stheta->ks = transform_log_mv_to_gamma_k(theta[4], theta[5]) + 2;
    stheta->ts = transform_log_mv_to_gamma_t(theta[4], theta[5]);

    stheta->pp = theta[6]; 
    stheta->ap = theta[7];

    stheta->t0 = exp(theta[8]);
    stheta->da = exp(theta[9]);

    stheta->p0 = theta[10]; // atan(theta[10])/M_PI + 0.5;

    stheta->cumint = CUMINT_NO_INIT; // Initilize value to empty

    return 0;
}

int
reparametrize_seri_wald(const double *theta, SERI_PARAMETERS *stheta)
{
    double mu, sigma2;
    
    mu = exp(theta[0]);
    sigma2 = exp(theta[1]);

    stheta->kp = transform_inv_to_wald_mu(mu, sigma2);
    stheta->tp = transform_inv_to_wald_l(mu, sigma2);

    mu = exp(theta[2]);
    sigma2 = exp(theta[3]);
    
    stheta->ka = transform_inv_to_wald_mu(mu, sigma2);
    stheta->ta = transform_inv_to_wald_l(mu, sigma2);

    mu = exp(theta[4]);
    sigma2 = exp(theta[5]);
    
    stheta->ks = transform_inv_to_wald_mu(mu, sigma2);
    stheta->ts = transform_inv_to_wald_l(mu, sigma2);

    stheta->pp = atan(theta[6])/M_PI + 0.5;
    stheta->ap = atan(theta[7])/M_PI + 0.5;

    stheta->t0 = exp(theta[8]);
    stheta->da = exp(theta[9]);

    stheta->p0 = atan(theta[10])/M_PI + 0.5;

    stheta->cumint = CUMINT_NO_INIT; // Initilize value to empty

    return 0;
}

int
reparametrize_seri_mixedgamma(const double *theta, SERI_PARAMETERS *stheta)
{
    double mu, sigma2;
    
    stheta->kp = transform_log_mv_to_gamma_k(theta[0], theta[1]) + 2;
    stheta->tp = transform_log_mv_to_gamma_t(theta[0], theta[1]);

    stheta->ka = transform_log_mv_to_invgamma_k(theta[2], theta[3]);
    stheta->ta = transform_log_mv_to_invgamma_t(theta[2], theta[3]);
    
    stheta->ks = transform_log_mv_to_gamma_k(theta[4], theta[5]) + 2;
    stheta->ts = transform_log_mv_to_gamma_t(theta[4], theta[5]); 

    stheta->pp = theta[6]; 
    stheta->ap = theta[7]; 

    stheta->t0 = exp(theta[8]);
    stheta->da = exp(theta[9]);

    stheta->p0 = theta[10]; 

    stheta->cumint = CUMINT_NO_INIT; // Initilize value to empty

    return 0;
}

int
reparametrize_seri_gamma(const double *theta, SERI_PARAMETERS *stheta)
{
    
    stheta->kp = transform_log_mv_to_invgamma_k(theta[0], theta[1]);
    stheta->tp = transform_log_mv_to_invgamma_t(theta[0], theta[1]);

    stheta->ka = transform_log_mv_to_invgamma_k(theta[2], theta[3]);
    stheta->ta = transform_log_mv_to_invgamma_t(theta[2], theta[3]);

    stheta->ks = transform_log_mv_to_invgamma_k(theta[4], theta[5]);
    stheta->ts = transform_log_mv_to_invgamma_t(theta[4], theta[5]);

    stheta->pp = theta[6]; 
    stheta->ap = theta[7];

    stheta->t0 = exp(theta[8]);
    stheta->da = exp(theta[9]);

    stheta->p0 = theta[10];

    stheta->cumint = CUMINT_NO_INIT; // Initilize value to empty

    return 0;
}

int
reparametrize_seri_later(const double *theta, SERI_PARAMETERS *stheta)
{
    
    stheta->kp = theta[0];
    stheta->tp = exp(0.5 * theta[1]);

    stheta->ka = theta[2]; 
    stheta->ta = exp(0.5 * theta[3]); 

    stheta->ks = theta[4]; 
    stheta->ts = exp(0.5 * theta[5]);

    stheta->pp = theta[6];
    stheta->ap = theta[7];

    stheta->t0 = exp(theta[8]);
    stheta->da = exp(theta[9]);

    stheta->p0 = theta[10]; 

    stheta->cumint = CUMINT_NO_INIT; // Initilize value to empty

    return 0;
}

int
reparametrize_seri_lognorm(const double *theta, SERI_PARAMETERS *stheta)
{
    double mu, sigma2;

    stheta->tp = log(exp(theta[1] - 2 * theta[0]) + 1);
    stheta->kp = -(theta[0] - 0.5 * stheta->tp);
    stheta->tp = sqrt(stheta->tp);
    
    stheta->ta = log(exp(theta[3] - 2 * theta[2]) +  1);
    stheta->ka = -(theta[2] - 0.5 * stheta->ta);
    stheta->ta = sqrt(stheta->ta);

    stheta->ts = log(exp(theta[5] - 2 * theta[4]) + 1);
    stheta->ks = -(theta[4] - 0.5 * stheta->ts);
    stheta->ts = sqrt(stheta->ts);

    stheta->pp = theta[6]; 
    stheta->ap = theta[7];

    stheta->t0 = exp(theta[8]);
    stheta->da = exp(theta[9]);

    stheta->p0 = theta[10];

    stheta->cumint = CUMINT_NO_INIT; // Initilize value to empty

    return 0;
}

int
reparametrize_dora_invgamma(const double *theta, DORA_PARAMETERS *stheta)
{
    
    stheta->kp = transform_log_mv_to_gamma_k(theta[0], theta[1]) + 2;
    stheta->tp = transform_log_mv_to_gamma_t(theta[0], theta[1]);

    stheta->ka = transform_log_mv_to_gamma_k(theta[2], theta[3]) + 2;
    stheta->ta = transform_log_mv_to_gamma_t(theta[2], theta[3]);

    stheta->ks = transform_log_mv_to_gamma_k(theta[4], theta[5]) + 2;
    stheta->ts = transform_log_mv_to_gamma_t(theta[4], theta[5]);

    stheta->kl = transform_log_mv_to_gamma_k(theta[6], theta[7]) + 2;
    stheta->tl = transform_log_mv_to_gamma_t(theta[6], theta[7]);

    stheta->t0 = exp(theta[8]);
    stheta->da = exp(theta[9]);

    stheta->p0 = theta[10]; //atan(theta[10])/M_PI + 0.5;

    stheta->cumint = CUMINT_NO_INIT; // Initilize value to empty

    return 0;
}

int
reparametrize_dora_wald(const double *theta, DORA_PARAMETERS *stheta)
{
    double mu, sigma2;
    
    mu = exp(theta[0]);
    sigma2 = exp(theta[1]);

    stheta->kp = transform_inv_to_wald_mu(mu, sigma2);
    stheta->tp = transform_inv_to_wald_l(mu, sigma2);

    mu = exp(theta[2]);
    sigma2 = exp(theta[3]);
    
    stheta->ka = transform_inv_to_wald_mu(mu, sigma2);
    stheta->ta = transform_inv_to_wald_l(mu, sigma2);

    mu = exp(theta[4]);
    sigma2 = exp(theta[5]);
    
    stheta->ks = transform_inv_to_wald_mu(mu, sigma2);
    stheta->ts = transform_inv_to_wald_l(mu, sigma2);

    mu = exp(theta[6]);
    sigma2 = exp(theta[7]);
    
    stheta->kl = transform_inv_to_wald_mu(mu, sigma2);
    stheta->tl = transform_inv_to_wald_l(mu, sigma2);

    stheta->t0 = exp(theta[8]);
    stheta->da = exp(theta[9]);

    stheta->p0 = theta[10];

    stheta->cumint = CUMINT_NO_INIT; // Initilize value to empty

    return 0;
}

int
reparametrize_dora_mixedgamma(const double *theta, DORA_PARAMETERS *stheta)
{

    stheta->kp = transform_log_mv_to_gamma_k(theta[0], theta[1]) + 2;
    stheta->tp = transform_log_mv_to_gamma_t(theta[0], theta[1]);

    stheta->ka = transform_log_mv_to_invgamma_k(theta[2], theta[3]);
    stheta->ta = transform_log_mv_to_invgamma_t(theta[2], theta[3]);

    stheta->ks = transform_log_mv_to_gamma_k(theta[4], theta[5]) + 2;
    stheta->ts = transform_log_mv_to_gamma_t(theta[4], theta[5]);

    stheta->kl = transform_log_mv_to_invgamma_k(theta[6], theta[7]);
    stheta->tl = transform_log_mv_to_invgamma_t(theta[6], theta[7]);

    stheta->t0 = exp(theta[8]);
    stheta->da = exp(theta[9]);

    stheta->p0 = theta[10];

    stheta->cumint = CUMINT_NO_INIT; // Initilize value to empty

    return 0;
}

int
reparametrize_dora_gamma(const double *theta, DORA_PARAMETERS *stheta)
{
    double mu, sigma2;
    
    stheta->kp = transform_log_mv_to_invgamma_k(theta[0], theta[1]);
    stheta->tp = transform_log_mv_to_invgamma_t(theta[0], theta[1]);

    stheta->ka = transform_log_mv_to_invgamma_k(theta[2], theta[3]);
    stheta->ta = transform_log_mv_to_invgamma_t(theta[2], theta[3]);

    stheta->ks = transform_log_mv_to_invgamma_k(theta[4], theta[5]);
    stheta->ts = transform_log_mv_to_invgamma_t(theta[4], theta[5]);

    stheta->kl = transform_log_mv_to_invgamma_k(theta[6], theta[7]);
    stheta->tl = transform_log_mv_to_invgamma_t(theta[6], theta[7]);

    stheta->t0 = exp(theta[8]);
    stheta->da = exp(theta[9]);

    stheta->p0 = theta[10]; //atan(theta[10])/M_PI + 0.5;

    stheta->cumint = CUMINT_NO_INIT; // Initilize value to empty

    return 0;
}

int
reparametrize_dora_later(const double *theta, DORA_PARAMETERS *stheta)
{
    
    stheta->kp = theta[0];
    stheta->tp = exp(0.5 * theta[1]);

    stheta->ka = theta[2]; 
    stheta->ta = exp(0.5 * theta[3]); 

    stheta->ks = theta[4]; 
    stheta->ts = exp(0.5 * theta[5]);

    stheta->kl = theta[6]; 
    stheta->tl = exp(0.5 * theta[7]);

    stheta->t0 = exp(theta[8]);
    stheta->da = exp(theta[9]);

    stheta->p0 = theta[10];

    stheta->cumint = CUMINT_NO_INIT; // Initilize value to empty

    return 0;
}

int
reparametrize_dora_lognorm(const double *theta, DORA_PARAMETERS *stheta)
{
    
    stheta->tp = log(exp(theta[1] - 2 * theta[0]) + 1);
    stheta->kp = -(theta[0] - 0.5 * stheta->tp);
    stheta->tp = sqrt(stheta->tp);
    
    stheta->ta = log(exp(theta[3] - 2 * theta[2]) +  1);
    stheta->ka = -(theta[2] - 0.5 * stheta->ta);
    stheta->ta = sqrt(stheta->ta);

    stheta->ts = log(exp(theta[5] - 2 * theta[4]) + 1);
    stheta->ks = -(theta[4] - 0.5 * stheta->ts);
    stheta->ts = sqrt(stheta->ts);

    stheta->tl = log(exp(theta[7] - 2 * theta[6])  + 1);
    stheta->kl = -(theta[6] - 0.5 * stheta->tl);
    stheta->tl = sqrt(stheta->tl);

    stheta->t0 = exp(theta[8]);
    stheta->da = exp(theta[9]);

    stheta->p0 = theta[10];

    stheta->cumint = CUMINT_NO_INIT; // Initilize value to empty

    return 0;
}

int
reparametrize_prosa_invgamma(const double *theta, PROSA_PARAMETERS *stheta)
{
    double mu, sigma2;
    
    mu = exp(theta[0]);
    sigma2 = exp(theta[1]);

    stheta->kp = transform_mv_to_gamma_k(mu, sigma2) + 2;
    stheta->tp = transform_mv_to_gamma_t(mu, sigma2);

    mu = exp(theta[2]);
    sigma2 = exp(theta[3]);
    
    stheta->ka = transform_mv_to_gamma_k(mu, sigma2) + 2;
    stheta->ta = transform_mv_to_gamma_t(mu, sigma2);

    mu = exp(theta[4]);
    sigma2 = exp(theta[5]);
    
    stheta->ks = transform_mv_to_gamma_k(mu, sigma2) + 2;
    stheta->ts = transform_mv_to_gamma_t(mu, sigma2); 

    stheta->t0 = exp(theta[6]);
    stheta->da = exp(theta[7]);

    stheta->p0 = theta[8]; 

    stheta->cumint = CUMINT_NO_INIT; // Initilize value to empty

    return 0;
}

int
reparametrize_prosa_wald(const double *theta, PROSA_PARAMETERS *stheta)
{
    double mu, sigma2;
    
    mu = exp(theta[0]);
    sigma2 = exp(theta[1]);

    stheta->kp = transform_inv_to_wald_mu(mu, sigma2);
    stheta->tp = transform_inv_to_wald_l(mu, sigma2);

    mu = exp(theta[2]);
    sigma2 = exp(theta[3]);
    
    stheta->ka = transform_inv_to_wald_mu(mu, sigma2);
    stheta->ta = transform_inv_to_wald_l(mu, sigma2);

    mu = exp(theta[4]);
    sigma2 = exp(theta[5]);
    
    stheta->ks = transform_inv_to_wald_mu(mu, sigma2);
    stheta->ts = transform_inv_to_wald_l(mu, sigma2);

    stheta->t0 = exp(theta[6]);
    stheta->da = exp(theta[7]);
    stheta->p0 = theta[8]; 

    stheta->cumint = CUMINT_NO_INIT; // Initilize value to empty

    return 0;
}

int
reparametrize_prosa_mixedgamma(const double *theta, PROSA_PARAMETERS *stheta)
{
    double mu, sigma2;
    
    mu = exp(theta[0]);
    sigma2 = exp(theta[1]);

    stheta->kp = transform_mv_to_gamma_k(mu, sigma2) + 2;
    stheta->tp = transform_mv_to_gamma_t(mu, sigma2);

    stheta->ka = transform_log_mv_to_invgamma_k(theta[2], theta[3]);
    stheta->ta = transform_log_mv_to_invgamma_t(theta[2], theta[3]);

    mu = exp(theta[4]);
    sigma2 = exp(theta[5]);
    
    stheta->ks = transform_mv_to_gamma_k(mu, sigma2) + 2;
    stheta->ts = transform_mv_to_gamma_t(mu, sigma2); 

    stheta->t0 = exp(theta[6]);
    stheta->da = exp(theta[7]);
    stheta->p0 = theta[8]; 

    stheta->cumint = CUMINT_NO_INIT; // Initilize value to empty

    return 0;
}

int
reparametrize_prosa_gamma(const double *theta, PROSA_PARAMETERS *stheta)
{
    
    stheta->kp = transform_log_mv_to_invgamma_k(theta[0], theta[1]);
    stheta->tp = transform_log_mv_to_invgamma_t(theta[0], theta[1]);
    
    stheta->ka = transform_log_mv_to_invgamma_k(theta[2], theta[3]);
    stheta->ta = transform_log_mv_to_invgamma_t(theta[2], theta[3]);
    
    stheta->ks = transform_log_mv_to_invgamma_k(theta[4], theta[5]);
    stheta->ts = transform_log_mv_to_invgamma_t(theta[4], theta[5]);

    stheta->t0 = exp(theta[6]);
    stheta->da = exp(theta[7]);
    stheta->p0 = theta[8]; 

    stheta->cumint = CUMINT_NO_INIT; // Initilize value to empty

    return 0;
}

int
reparametrize_prosa_later(const double *theta, PROSA_PARAMETERS *stheta)
{
    
    stheta->kp = theta[0];
    stheta->tp = exp(0.5 * theta[1]);

    stheta->ka = theta[2]; 
    stheta->ta = exp(0.5 * theta[3]); 

    stheta->ks = theta[4]; 
    stheta->ts = exp(0.5 * theta[5]);

    stheta->t0 = exp(theta[6]);
    stheta->da = exp(theta[7]);
    stheta->p0 = theta[8];

    stheta->cumint = CUMINT_NO_INIT; // Initilize value to empty

    return 0;
}

int
reparametrize_prosa_lognorm(const double *theta, PROSA_PARAMETERS *stheta)
{

    stheta->tp = log(exp(theta[1] - 2 * theta[0]) + 1);
    stheta->kp = -(theta[0] - 0.5 * stheta->tp);
    stheta->tp = sqrt(stheta->tp);
    
    stheta->ta = log(exp(theta[3] - 2 * theta[2]) +  1);
    stheta->ka = -(theta[2] - 0.5 * stheta->ta);
    stheta->ta = sqrt(stheta->ta);

    stheta->ts = log(exp(theta[5] - 2 * theta[4]) + 1);
    stheta->ks = -(theta[4] - 0.5 * stheta->ts);
    stheta->ts = sqrt(stheta->ts);

    stheta->t0 = exp(theta[6]);
    stheta->da = exp(theta[7]);

    stheta->p0 = theta[8]; 

    stheta->cumint = CUMINT_NO_INIT; // Initilize value to empty

    return 0;
}


