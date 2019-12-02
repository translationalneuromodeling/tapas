/* aponteeduardo@gmail.com */
/* copyright (C) 2017 */

#include "antisaccades.h"

// This is done to reparametrize the code inside c as opposed to matlab

const ACCUMULATOR ACCUMULATOR_INVGAMMA = {
    .pdf = invgamma_pdf,
    .lpdf = invgamma_lpdf,
    .cdf = invgamma_cdf,
    .lcdf = invgamma_lcdf,
    .sf = invgamma_sf,
    .lsf = invgamma_lsf
};

const ACCUMULATOR ACCUMULATOR_GAMMA = {
    .pdf = gamma_pdf,
    .lpdf = gamma_lpdf,
    .cdf = gamma_cdf,
    .lcdf = gamma_lcdf,
    .sf = gamma_sf,
    .lsf = gamma_lsf
};

const ACCUMULATOR ACCUMULATOR_WALD = {
    .pdf = wald_pdf,
    .lpdf = wald_lpdf,
    .cdf = wald_cdf,
    .lcdf = wald_lcdf,
    .sf = wald_sf,
    .lsf = wald_lsf
};

const ACCUMULATOR ACCUMULATOR_LATER = {
    .pdf = later_pdf,
    .lpdf = later_lpdf,
    .cdf = later_cdf,
    .lcdf = later_lcdf,
    .sf = later_sf,
    .lsf = later_lsf
};

const ACCUMULATOR ACCUMULATOR_LOGNORM = {
    .pdf = lognorm_pdf,
    .lpdf = lognorm_lpdf,
    .cdf = lognorm_cdf,
    .lcdf = lognorm_lcdf,
    .sf = lognorm_sf,
    .lsf = lognorm_lsf
};



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
reparametrize_seria_invgamma(const double *theta, SERIA_PARAMETERS *stheta)
{

    stheta->kp = transform_log_mv_to_gamma_k(theta[0], theta[1]);
    stheta->tp = transform_log_mv_to_gamma_t(theta[0], theta[1]);

    stheta->ka = transform_log_mv_to_gamma_k(theta[2], theta[3]);
    stheta->ta = transform_log_mv_to_gamma_t(theta[2], theta[3]);

    stheta->ks = transform_log_mv_to_gamma_k(theta[4], theta[5]);
    stheta->ts = transform_log_mv_to_gamma_t(theta[4], theta[5]);

    stheta->kl = transform_log_mv_to_gamma_k(theta[6], theta[7]);
    stheta->tl = transform_log_mv_to_gamma_t(theta[6], theta[7]);

    stheta->t0 = exp(theta[8]);
    stheta->da = exp(theta[9]);

    stheta->p0 = theta[10]; //atan(theta[10])/M_PI + 0.5;

    stheta->cumint = CUMINT_NO_INIT; // Initilize value to empty

    stheta->early = ACCUMULATOR_INVGAMMA;
    stheta->stop = ACCUMULATOR_INVGAMMA;
    stheta->anti = ACCUMULATOR_INVGAMMA;
    stheta->late = ACCUMULATOR_INVGAMMA;

    stheta->inhibition_race = ninvgamma_gslint;

    return 0;
}

int
reparametrize_seria_wald(const double *theta, SERIA_PARAMETERS *stheta)
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

    stheta->early = ACCUMULATOR_WALD;
    stheta->stop = ACCUMULATOR_WALD;
    stheta->anti = ACCUMULATOR_WALD;
    stheta->late = ACCUMULATOR_WALD;

    stheta->inhibition_race = nwald_gslint;

    return 0;
}

int
reparametrize_seria_mixedgamma(const double *theta, SERIA_PARAMETERS *stheta)
{

    stheta->kp = transform_log_mv_to_gamma_k(theta[0], theta[1]);
    stheta->tp = transform_log_mv_to_gamma_t(theta[0], theta[1]);

    stheta->ka = transform_log_mv_to_invgamma_k(theta[2], theta[3]);
    stheta->ta = transform_log_mv_to_invgamma_t(theta[2], theta[3]);

    stheta->ks = transform_log_mv_to_gamma_k(theta[4], theta[5]);
    stheta->ts = transform_log_mv_to_gamma_t(theta[4], theta[5]);

    stheta->kl = transform_log_mv_to_invgamma_k(theta[6], theta[7]);
    stheta->tl = transform_log_mv_to_invgamma_t(theta[6], theta[7]);

    stheta->t0 = exp(theta[8]);
    stheta->da = exp(theta[9]);

    stheta->p0 = theta[10];

    stheta->cumint = CUMINT_NO_INIT; // Initilize value to empty

    stheta->early = ACCUMULATOR_INVGAMMA;
    stheta->stop = ACCUMULATOR_INVGAMMA;
    stheta->anti = ACCUMULATOR_GAMMA;
    stheta->late = ACCUMULATOR_GAMMA;

    stheta->inhibition_race = ninvgamma_gslint;

    return 0;
}

int
reparametrize_seria_gamma(const double *theta, SERIA_PARAMETERS *stheta)
{

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

    stheta->early = ACCUMULATOR_GAMMA;
    stheta->stop = ACCUMULATOR_GAMMA;
    stheta->anti = ACCUMULATOR_GAMMA;
    stheta->late = ACCUMULATOR_GAMMA;

    stheta->inhibition_race = ngamma_gslint;

    return 0;
}

int
reparametrize_seria_later(const double *theta, SERIA_PARAMETERS *stheta)
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

    stheta->early = ACCUMULATOR_LATER;
    stheta->stop = ACCUMULATOR_LATER;
    stheta->anti = ACCUMULATOR_LATER;
    stheta->late = ACCUMULATOR_LATER;

    stheta->inhibition_race = nlater_gslint;

    return 0;
}

int
reparametrize_seria_lognorm(const double *theta, SERIA_PARAMETERS *stheta)
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

    stheta->early = ACCUMULATOR_LOGNORM;
    stheta->stop = ACCUMULATOR_LOGNORM;
    stheta->anti = ACCUMULATOR_LOGNORM;
    stheta->late = ACCUMULATOR_LOGNORM;

    stheta->inhibition_race = nlognorm_gslint;

    return 0;
}

int
reparametrize_prosa_invgamma(const double *theta, PROSA_PARAMETERS *stheta)
{

    stheta->kp = transform_log_mv_to_gamma_k(theta[0], theta[1]);
    stheta->tp = transform_log_mv_to_gamma_t(theta[0], theta[1]);

    stheta->ka = transform_log_mv_to_gamma_k(theta[2], theta[3]);
    stheta->ta = transform_log_mv_to_gamma_t(theta[2], theta[3]);

    stheta->ks = transform_log_mv_to_gamma_k(theta[4], theta[5]);
    stheta->ts = transform_log_mv_to_gamma_t(theta[4], theta[5]);

    stheta->t0 = exp(theta[6]);
    stheta->da = exp(theta[7]);

    stheta->p0 = theta[8];

    stheta->cumint = CUMINT_NO_INIT; // Initilize value to empty

    stheta->early = ACCUMULATOR_INVGAMMA;
    stheta->stop = ACCUMULATOR_INVGAMMA;
    stheta->anti = ACCUMULATOR_INVGAMMA;

    stheta->inhibition_race = ninvgamma_gslint;

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

    stheta->early = ACCUMULATOR_WALD;
    stheta->stop = ACCUMULATOR_WALD;
    stheta->anti = ACCUMULATOR_WALD;

    stheta->inhibition_race = nwald_gslint;

    return 0;
}

int
reparametrize_prosa_mixedgamma(const double *theta, PROSA_PARAMETERS *stheta)
{

    stheta->kp = transform_log_mv_to_gamma_k(theta[0], theta[1]);
    stheta->tp = transform_log_mv_to_gamma_t(theta[0], theta[1]);

    stheta->ka = transform_log_mv_to_invgamma_k(theta[2], theta[3]);
    stheta->ta = transform_log_mv_to_invgamma_t(theta[2], theta[3]);

    stheta->ks = transform_log_mv_to_gamma_k(theta[4], theta[5]);
    stheta->ts = transform_log_mv_to_gamma_t(theta[4], theta[5]);

    stheta->t0 = exp(theta[6]);
    stheta->da = exp(theta[7]);
    stheta->p0 = theta[8];

    stheta->cumint = CUMINT_NO_INIT; // Initilize value to empty

    stheta->early = ACCUMULATOR_INVGAMMA;
    stheta->stop = ACCUMULATOR_INVGAMMA;
    stheta->anti = ACCUMULATOR_GAMMA;

    stheta->inhibition_race = ninvgamma_gslint;

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

    stheta->early = ACCUMULATOR_GAMMA;
    stheta->stop = ACCUMULATOR_GAMMA;
    stheta->anti = ACCUMULATOR_GAMMA;

    stheta->inhibition_race = ngamma_gslint;

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

    stheta->early = ACCUMULATOR_LATER;
    stheta->stop = ACCUMULATOR_LATER;
    stheta->anti = ACCUMULATOR_LATER;

    stheta->inhibition_race = nlater_gslint;

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

    stheta->early = ACCUMULATOR_LOGNORM;
    stheta->stop = ACCUMULATOR_LOGNORM;
    stheta->anti = ACCUMULATOR_LOGNORM;

    stheta->inhibition_race = nlognorm_gslint;

    return 0;
}

int
linearize_prosa(const PROSA_PARAMETERS *stheta, double *theta)
{
    double p0 = stheta->p0;

    theta[0] = stheta->kp;
    theta[1] = stheta->tp;

    theta[2] = stheta->ka;
    theta[3] = stheta->ta;

    theta[4] = stheta->ks;
    theta[5] = stheta->ts;

    theta[6] = stheta->t0;
    theta[7] = stheta->da;
    theta[8] = 1.0 - exp(-0.5 * p0 - M_LN2 - lcosh(0.5 * p0));

   return 0;
}

int
linearize_seria(const SERIA_PARAMETERS *stheta, double *theta)
{
    
    double p0 = stheta->p0;
    
    theta[0] = stheta->kp;
    theta[1] = stheta->tp;

    theta[2] = stheta->ka;
    theta[3] = stheta->ta;

    theta[4] = stheta->ks;
    theta[5] = stheta->ts;

    theta[6] = stheta->kl;
    theta[7] = stheta->tl;

    theta[8] = stheta->t0;
    theta[9] = stheta->da;
    //atan(theta[10])/M_PI + 0.5;
    theta[10] = 1.0 - exp(-0.5 * p0 - M_LN2 - lcosh(0.5 * p0));

    return 0;
}

