/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */

#ifndef C_ANTISACCADES
#define C_ANTISACCADES

#ifndef TAPAS_PYTHON

#include <matrix.h>
#include <mex.h>

#endif

#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_erf.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_sys.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>
#include <assert.h>

#define PROSACCADE 0
#define ANTISACCADE 1

#define NITER_NGAMMA 15
#define INTSTEP 80

#define DIM_PROSA_THETA 9 
#define DIM_SERI_THETA 11
#define DIM_DORA_THETA 11

#define MAX_INTEGRATION_SIZE 81

#define LOG2PI 1.8378770664093453
#define SQRT2PI 2.5066282746310002 

// THe probability of an outlier that it's an antisaccade 0.0005, that is
// 5 times every 10000 saccades 

#define LN_P_OUTLIER_ANTI -7.6009
#define LN_P_OUTLIER_PRO -0.0005001250

//#define LN_P_LATE_OUTLIER -7.6009024595420822
//#define LN_P_LATE_NOT_OUTLIER -0.0005001250


# define HANDLE_CERROR( err ) {if ( err ) { \
    printf("%s in %s at line %d\n", "Assertion error", \
                    __FILE__, __LINE__ ); \
    mexErrMsgIdAndTxt("tapas:sem:antisaccades:llh", "Error encountered"); \
    }}

# define SOONER_GSLERROR( err ) {if ( err ) { \
    return GSL_NAN; \
    }}

typedef struct
{
    double *t;
    double *a;
    double *u;
    double *theta; // Theta
    
    int nt; // Number of data point
    int np; // Number of parameters

} ANTIS_INPUT;

// Parameters type

typedef struct
{
    double t0; // No decision time
    double p0; // Probability of a non decision action

    double kp;
    double tp;
    double ka;
    double ta;
    double ks;
    double ts;

    double da;

} PROSA_PARAMETERS;

typedef struct
{
    double t0;
    double p0;

    double kp;
    double tp;
    double ka;
    double ta;
    double ks;
    double ts;
    
    double pp;
    double ap; 
    
    double da;
    
} SERI_PARAMETERS;

// Double race model

typedef struct
{
    double t0;
    double p0;

    double kinv;
    double tinv;
    double kva;
    double tva;
    double kvp;
    double tvp;
    double ks;
    double ts;
    
    double da;
    
} DORA_PARAMETERS;

// Likelihood typedef functions

typedef double (*PROSA_LLH)(double t, int a, PROSA_PARAMETERS params); 

typedef double (*SERI_LLH)(double t, int a, SERI_PARAMETERS params); 

typedef double (*DORA_LLH)(double t, int a, DORA_PARAMETERS params); 


// Populate parameters

int
populate_parameters_prosa(const double *theta, PROSA_PARAMETERS *stheta);
// Populate the DORA parametes using teh array theta, where theta is required
// to have 11 parameters 
// theta        -- Array of parameters
// stheta       -- Structure to be populated

int
populate_parameters_seri(const double *theta, SERI_PARAMETERS *stheta);
// Populate the SERI parametes using teh array theta, where theta is required
// to have 11 parameters 
// theta        -- Array of parameters
// stheta       -- Structure to be populated

int
populate_parameters_dora(const double *theta, DORA_PARAMETERS *stheta);
// Populate the DORA parametes using teh array theta, where theta is required
// to have 11 parameters 
// theta        -- Array of parameters
// stheta       -- Structure to be populated


// Likelihoods: Compute the likelihood of a single trial

// Prosa model

double
prosa_llh_gamma(double t, int a, PROSA_PARAMETERS params);

double
prosa_llh_invgamma(double t, int a, PROSA_PARAMETERS params);

double
prosa_llh_mixedgamma(double t, int a, PROSA_PARAMETERS params);

double
prosa_llh_lognorm(double t, int a, PROSA_PARAMETERS params);

double
prosa_llh_later(double t, int a, PROSA_PARAMETERS params);

// TODO
double
prosa_llh_wald(double t, int a, PROSA_PARAMETERS params);

// Seri model

double
seri_llh_gamma(double t, int a, SERI_PARAMETERS params);

double
seri_llh_invgamma(double t, int a, SERI_PARAMETERS params);

double
seri_llh_mixedgamma(double t, int a, SERI_PARAMETERS params);

double
seri_llh_lognorm(double t, int a, SERI_PARAMETERS params);

double
seri_llh_later(double t, int a, SERI_PARAMETERS params);

double
seri_llh_wald(double t, int a, SERI_PARAMETERS params);

// Dora model

double
dora_llh_gamma(double t, int a, DORA_PARAMETERS params);

double
dora_llh_invgamma(double t, int a, DORA_PARAMETERS params);

double
dora_llh_mixedgamma(double t, int a, DORA_PARAMETERS params);

//TODO
double
dora_llh_lognorm(double t, int a, DORA_PARAMETERS params);

//TODO
double
dora_llh_later(double t, int a, DORA_PARAMETERS params);

//TODO
double
dora_llh_wald(double t, int a, DORA_PARAMETERS params);

// Models

// Computation of the likelihood in parallel assuming a differnt set of 
// parameters for each trial.
// svals    -- Input to the likelihoods
// fllh     -- Function for the trial wise likelihood
// llh      -- Double array of the size svals.np. 


int
prosa_model_trial_by_trial(ANTIS_INPUT svals, PROSA_LLH fllh, double *llh);

int
prosa_model_two_states(ANTIS_INPUT svals, PROSA_LLH fllh, double *llh);

int
dora_model_trial_by_trial(ANTIS_INPUT svals, DORA_LLH fllh, double *llh);

int
dora_model_two_states(ANTIS_INPUT svals, DORA_LLH fllh, double *llh);

int
seri_model_trial_by_trial(ANTIS_INPUT svals, SERI_LLH fllh, double *llh);

int
seri_model_two_states(ANTIS_INPUT svals, SERI_LLH fllh, double *llh);


// Numericals

double gamma_lpdf(double x, double k, double t);
// Gamma log pdf
// x        Time.
// k        Shape parameter.
// t        Scale parameter.

double invgamma_lpdf(double x, double k, double t);
// Invgamma log pdf
// x        Time.
// k        Shape parameter.
// t        Scale parameter.

double lognorm_lpdf(double x, double mu, double sigma);
// Invgamma log pdf
// x        Time.
// k        Shape parameter.
// t        Scale parameter.

double later_lpdf(double x, double mu, double sigma);
// Log pdf of the later likelihood
// x        Time
// mu       Location parameter of the corresponding Gaussian
// sigma    Scale of the corresponding Gaussian

double wald_lpdf(double x, double mu, double sigma);
// Log pdf of the later likelihood
// x        Time
// mu       Location parameter of the corresponding Wald distribution
// sigma    Scale of the corresponding sigma parameter

// Probability density functions

double later_pdf(double x, double mu, double sigma);
// Pdf of the later likelihood
// x        Time
// mu       Location parameter of the corresponding Gaussian
// sigma    Scale of the corresponding Gaussian

double wald_pdf(double x, double mu, double lambda);
// Pdf of the later likelihood
// x        Time
// mu       Location parameter of the corresponding Gaussian
// sigma    Scale of the corresponding Gaussian

// Cumulatie density functions

double later_cdf(double x, double mu, double sigma);
// cdf of the later likelihood
// x        Time
// mu       Location parameter of the corresponding Gaussian
// sigma    Scale of the corresponding Gaussian

double wald_cdf(double x, double mu, double sigma);
// cdf of the later likelihood
// x        Time
// mu       Location parameter of the corresponding Gaussian
// sigma    Scale of the corresponding Gaussian


double ngamma(double x, double a, double b, double c0, double c1);
// Unnormalized nested incomplete gamma function.
// x    -- Integration boundary
// a    -- Parameters of the outer gamma function.
// b    -- Parameters of the inter gamma function.
// c0   -- Coefficient associated with a.
// c1   -- Coefficient associated with b.
//
// See Aponte et al. 2016

double ninvgamma(double x, double a, double b, double c0, double c1);
// Unnormalized nested incomplete gamma function.
// x    -- Integration boundary
// a    -- Parameters of the outer gamma function.
// b    -- Parameters of the inter gamma function.
// c0   -- Coefficient associated with a.
// c1   -- Coefficient associated with b.
//
// See Aponte et al. 2016


double ngamma_lexp(double x, double a, double b, double c0, double c1);
// Unnormalized nested incomplete gamma function.
// x    -- Integration boundary
// a    -- Parameters of the outer gamma function.
// b    -- Parameters of the inter gamma function.
// c0   -- Coefficient associated with a.
// c1   -- Coefficient associated with b.
//
// Slow but safe.

double ngamma_gslint(double x, double a, double b, double c0, double c1);
// Unnormalized nested incomplete gamma function.
// x    -- Integration boundary
// a    -- Parameters of the outer gamma function.
// b    -- Parameters of the inter gamma function.
// c0   -- Coefficient associated with a.
// c1   -- Coefficient associated with b.
//
// Uses gsl


double ninvgamma_gslint(double x, double a, double b, double c0, double c1);
// Unnormalized nested incomplete gamma function.
// x    -- Integration boundary
// a    -- Parameters of the outer gamma function.
// b    -- Parameters of the inter gamma function.
// c0   -- Coefficient associated with a.
// c1   -- Coefficient associated with b.
//
// Uses gls integrator.


double ninvgamma_lexp(double x, double a, double b, double c0, double c1);
// Unnormalized nested incomplete gamma function.
// x    -- Integration boundary
// a    -- Parameters of the outer gamma function.
// b    -- Parameters of the inter gamma function.
// c0   -- Coefficient associated with a.
// c1   -- Coefficient associated with b.
//
// Slow but safe.

double
nlognorm_gslint(double x, double mu1, double mu2, double sig1, double sig2);
// Unnormalized nested incomplete gamma function.
// x    -- Integration boundary
// mu1    -- Location parameter of the outer lognormal function.
// mu2    -- Location parameter of the inner lognormal function.
// sig1    -- Scale parameter of the outer lognormal function.
// sig2    -- Scale parameter of the inner lognormal function.
//
// Uses gsldouble


double nlater_gslint(double x, double mu1, double mu2, double sig1, double sig2);
// Nested Integral for Gaussian Rates
// x    -- Integration boundary
// mu1    -- Location parameter of the outer lognormal function.
// mu2    -- Location parameter of the inner lognormal function.
// sig1    -- Scale parameter of the outer lognormal function.
// sig2    -- Scale parameter of the inner lognormal function.
//
// Uses gsl

double wald_gslint(double x, double mu1, double mu2, double sig1, double sig2);
// Nested Integral for Walt rt
// x    -- Integration boundary
// mu1    -- Location parameter of the outer lognormal function.
// mu2    -- Location parameter of the inner lognormal function.
// sig1    -- Scale parameter of the outer lognormal function.
// sig2    -- Scale parameter of the inner lognormal function.
//
// Uses gsl


#endif
