/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */

#ifndef C_ANTISACCADES
#define C_ANTISACCADES

#ifndef TAPAS_PYTHON

#include <matrix.h>
#include <mex.h>

#endif

#include "includes/sem.h"

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
#include <gsl/gsl_sf_trig.h>
#include <assert.h>

#define PROSACCADE 0
#define ANTISACCADE 1

// This is used because sometimes functions are not defined at zero.
#define ZERO_DISTS 0.000001
// Absolute error tolerance
#define SEM_TOL 0.00001
// Relative error tolerance
#define SEM_RTOL 0.0001
// Maximum shape parameter gamma distribution
#define SEM_GAMMA_MAX_SHAPE 100

#define NITER_NGAMMA 15
#define INTSTEP 80

#define DIM_PROSA_THETA 9 
#define DIM_SERIA_THETA 11

#define MAX_INTEGRATION_SIZE 81

#define LOG2PI 1.8378770664093453
#define SQRT2PI 2.5066282746310002 

// THe probability of an outlier that it's an antisaccade 0.0005, that is
// 5 times every 10000 saccades 

#define LN_P_OUTLIER_ANTI -4.6051701859880909
#define LN_P_OUTLIER_PRO -0.010050335853501451

#define LN_P_SERIA_EARLY_PRO -0.0010005003335835344
#define LN_P_SERIA_EARLY_ANTI -6.9077552789821368

//#define LN_P_LATE_OUTLIER -7.6009024595420822
//#define LN_P_LATE_NOT_OUTLIER -0.0005001250

# define HANDLE_CERROR( err ) {if ( err ) { \
    printf("%s in %s at line %d\n", "Assertion error", \
                    __FILE__, __LINE__ ); \
    mexErrMsgIdAndTxt("tapas:sem:antisaccades:llh", "Error encountered"); \
    }}

# define HANDLE_CERROR_MSG( err, msg ) {if ( err ) { \
    printf("%s in %s at line %d\n", "Assertion error", \
                    __FILE__, __LINE__ ); \
    printf("%s \n", ( msg )); \
    mexErrMsgIdAndTxt("tapas:sem:antisaccades:llh", "Error encountered"); \
    }}


# define SOONER_GSLERROR( err ) {if ( err ) { \
    return GSL_NAN; \
    }}

#define CUMINT_NO_INIT -1

typedef double (*NESTED_INTEGRAL)(double x0, double x1, double a,
    double b, double c0, double c1);

// Likelihood, log likelihood, etc..
typedef double (*ACCUMULATOR_FUNCTION)(double time, double shape,
        double scale);

// This class contains the definition of a unit
typedef struct
{
    ACCUMULATOR_FUNCTION pdf;
    ACCUMULATOR_FUNCTION lpdf;

    ACCUMULATOR_FUNCTION cdf;
    ACCUMULATOR_FUNCTION lcdf;

    ACCUMULATOR_FUNCTION sf; // Survival function
    ACCUMULATOR_FUNCTION lsf;
} ACCUMULATOR;

typedef struct
{
    double *t;
    double *a;
    double *u;
    double *theta; // Theta
    
    int nt; // Number of data point
    int np; // Number of parameters

} ANTIS_INPUT;

typedef struct
{
    double *t;
    double *a;
    double *u;

} MODEL_INPUTS;
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

    double cumint; // Current value of the cumulative integral.

    ACCUMULATOR early;
    ACCUMULATOR stop;
    ACCUMULATOR anti;

    NESTED_INTEGRAL inhibition_race; 

} PROSA_PARAMETERS;

// SERIA model 
typedef struct
{
    double t0;
    double p0;

    double kp; // Early prossacades
    double tp; // Late prosaccades
    double ka; // Voluntary antisaccade
    double ta; 
    double kl; // Involuntary prosaccade
    double tl;
    double ks; // Stop
    double ts;  
    double da;

    double cumint; // Current value of the cumulative integral.
    
    ACCUMULATOR early;
    ACCUMULATOR stop;
    ACCUMULATOR anti;
    ACCUMULATOR late;

    NESTED_INTEGRAL inhibition_race; 

} SERIA_PARAMETERS;

// --------------------------------------------------------------------------
// Summaries
// --------------------------------------------------------------------------

typedef struct
{
    
    double late_pro_rt; // Reaction time of a late prosaccade
    double anti_rt;     // Reaction time of an antisaccade
    double inhib_fail_rt; // Reaction time of an inhib. fail.
    double inhib_fail_prob; // Probability of an inhibition failure
    double late_pro_prob; // Probability of a late error
    double predicted_pro_prob; // Predicted probability of pro
    double predicted_pro_rt; // Predicted probability of pro
    double predicted_anti_prob; // Predicted probability of anti
    double predicted_anti_rt; // Predicted probability of anti

} SERIA_SUMMARY;

typedef struct
{
    
    double anti_rt;     // Reaction time of an antisaccade
    double inhib_fail_rt; // Reaction time of an inhib. fail.
    double inhib_fail_prob; // Probability of an inhibition failure
    double predicted_pro_prob; // Predicted probability of pro
    double predicted_pro_rt; // Predicted probability of pro
    double predicted_anti_prob; // Predicted probability of anti
    double predicted_anti_rt; // Predicted probability of anti

} PROSA_SUMMARY;


// ------------

typedef double (*SERIA_SUMMARY_FUNCTION)(double time, 
        SERIA_PARAMETERS *params);

typedef double (*PROSA_SUMMARY_FUNCTION)(double time, 
        PROSA_PARAMETERS *params);

// -------------

typedef struct
{
    
    SERIA_SUMMARY_FUNCTION func; // The function to compute
    SERIA_PARAMETERS *params; // Parameters of the model

} SERIA_GSL_INT_INPUT;

typedef struct
{
    
    PROSA_SUMMARY_FUNCTION func; // The function to compute
    PROSA_PARAMETERS *params; // Parameters of the model

} PROSA_GSL_INT_INPUT;

// ---------------

double
seria_summary_parameter(
        SERIA_SUMMARY_FUNCTION summary_func,
        SERIA_PARAMETERS *params);
// Compute the summary parameters from the model.

double
prosa_summary_parameter(
        PROSA_SUMMARY_FUNCTION summary_func,
        PROSA_PARAMETERS *params);
// Compute the summary parameters from the model.

//-------------------

double
seria_summary_wrapper(double t, void *gsl_int_pars);
// Wrapper for numerical integration with gsl. 

double
prosa_summary_wrapper(double t, void *gsl_int_pars);
// Wrapper for numerical integration with gsl. 

//--------------------

// Likelihood typedef functions

typedef double (*PROSA_LLH)(double t, int a, const PROSA_PARAMETERS params); 

typedef double (*SERIA_LLH)(double t, int a, const SERIA_PARAMETERS params); 

// Reparametrization of a function

typedef int (*FILL_PARAMETERS_PROSA)(const double *theta, PROSA_PARAMETERS 
        *parameters);
typedef int (*FILL_PARAMETERS_SERIA)(const double *theta, SERIA_PARAMETERS 
        *parameters);


// Place holder for the function arguments

typedef struct
{
    FILL_PARAMETERS_PROSA fill_parameters;
    PROSA_LLH llh;
    int dim_theta;
} PROSA_MODEL;

typedef struct
{
    FILL_PARAMETERS_SERIA fill_parameters;
    SERIA_LLH llh;
    int dim_theta;
} SERIA_MODEL;

// Summaries of subjects actions

//--------------------

int
seria_model_summary(
    const ANTIS_INPUT svals,
    SERIA_MODEL model, 
    SERIA_SUMMARY *summaries);

int
prosa_model_summary(
    const ANTIS_INPUT svals,
    PROSA_MODEL model, 
    PROSA_SUMMARY *summaries);


// Other numerical
double
lcosh(double x);
// Logarithm of cosine hyperbolicus of x. Behaves better for large abs(x)

// Populate parameters
int
populate_parameters_prosa(const double *theta, PROSA_PARAMETERS *stheta);
// Populate the SERIA parametes using teh array theta, where theta is required
// to have 11 parameters 
// theta        -- Array of parameters
// stheta       -- Structure to be populated

int
populate_parameters_seria(const double *theta, SERIA_PARAMETERS *stheta);
// Populate the SERIA parametes using teh array theta, where theta is required
// to have 11 parameters 
// theta        -- Array of parameters
// stheta       -- Structure to be populated


// Likelihoods: Compute the likelihood of a single trial

// Prosa model
double
prosa_llh_abstract(double t, int a, PROSA_PARAMETERS params);

// Seria model
double
seria_llh_abstract(double t, int a, SERIA_PARAMETERS params);

double
seria_early_llh_abstract(double t, int a, SERIA_PARAMETERS params);

double
seria_late_llh_abstract(double t, int a, SERIA_PARAMETERS params);

// Summary

// Generate a summary of the parameters
// A summary of the parameters for reporting

int
seria_summary_abstract(SERIA_PARAMETERS *params, SERIA_SUMMARY *summary);
// Summary of the parameters.
//
// params       -- Parameters of the seria model.
// summary      -- Output structure containing the relevant parameters.

int
prosa_summary_abstract(PROSA_PARAMETERS *params, PROSA_SUMMARY *summary);
// Summary of the parameters.
//
// params       -- Parameters of the prosa model.
// summary      -- Output structure containing the relevant parameters.


// Models

// Computation of the likelihood in parallel assuming a differnt set of 
// parameters for each trial.
// svals    -- Input to the likelihoods
// fllh     -- Function for the trial wise likelihood
// llh      -- Double array of the size svals.np. 


int
prosa_model_n_states_optimized(ANTIS_INPUT svals, PROSA_MODEL model,
       double *llh);

int
prosa_model_n_states(ANTIS_INPUT svals, PROSA_MODEL model,
       double *llh);

int
seria_model_n_states(ANTIS_INPUT svals, SERIA_MODEL model, double *llh);

int
seria_model_n_states_optimized(ANTIS_INPUT svals, SERIA_MODEL model, 
        double *llh);

// Numericals

double gamma_pdf(double x, double k, double t);
// Gamma log pdf
// x        Time.
// k        Shape parameter.
// t        Scale parameter.

double gamma_lpdf(double x, double k, double t);
// Gamma log pdf
// x        Time.
// k        Shape parameter.
// t        Scale parameter.

double gamma_cdf(double x, double k, double t);
// Gamma cdf
// x        Time.
// k        Shape parameter.
// t        Scale parameter.

double gamma_lcdf(double x, double k, double t);
// Gamma log pdf
// x        Time.
// k        Shape parameter.
// t        Scale parameter.

double gamma_sf(double x, double k, double t);
// Gamma survival function
// x        Time.
// k        Shape parameter.
// t        Scale parameter.

double gamma_lsf(double x, double k, double t);
// Gamma log survival fuction
// x        Time.
// k        Shape parameter.
// t        Scale parameter.

double invgamma_pdf(double x, double k, double t);
// Invgamma log pdf
// x        Time.
// k        Shape parameter.
// t        Scale parameter.

double invgamma_lpdf(double x, double k, double t);
// Invgamma log pdf
// x        Time.
// k        Shape parameter.
// t        Scale parameter.

double invgamma_cdf(double x, double k, double t);
// Invgamma log pdf
// x        Time.
// k        Shape parameter.
// t        Scale parameter.

double invgamma_lcdf(double x, double k, double t);
// Invgamma log pdf
// x        Time.
// k        Shape parameter.
// t        Scale parameter.

double invgamma_sf(double x, double k, double t);
// Invgamma log pdf
// x        Time.
// k        Shape parameter.
// t        Scale parameter.

double invgamma_lsf(double x, double k, double t);
// Invgamma log pdf
// x        Time.
// k        Shape parameter.
// t        Scale parameter.


double lognorm_pdf(double x, double mu, double sigma);
// Invgamma pdf
// x        Time.
// k        Shape parameter.
// t        Scale parameter.

double lognorm_lpdf(double x, double mu, double sigma);
// Invgamma log pdf
// x        Time.
// k        Shape parameter.
// t        Scale parameter.

double lognorm_cdf(double x, double mu, double sigma);
// Pdf of the lognorm likelihood
// x        Time
// mu       Location parameter of the corresponding Gaussian
// sigma    Scale of the corresponding Gaussian

double lognorm_lcdf(double x, double mu, double sigma);
// Pdf of the lognorm likelihood
// x        Time
// mu       Location parameter of the corresponding Gaussian
// sigma    Scale of the corresponding Gaussian

double lognorm_sf(double x, double mu, double sigma);
// Pdf of the lognorm likelihood
// x        Time
// mu       Location parameter of the corresponding Gaussian
// sigma    Scale of the corresponding Gaussian

double lognorm_lsf(double x, double mu, double sigma);
// Pdf of the lognorm likelihood
// x        Time
// mu       Location parameter of the corresponding Gaussian
// sigma    Scale of the corresponding Gaussian

double later_lpdf(double x, double mu, double sigma);
// Log pdf of the later likelihood
// x        Time
// mu       Location parameter of the corresponding Gaussian
// sigma    Scale of the corresponding Gaussian

double later_pdf(double x, double mu, double sigma);
// Pdf of the later likelihood
// x        Time
// mu       Location parameter of the corresponding Gaussian
// sigma    Scale of the corresponding Gaussian

double later_cdf(double x, double mu, double sigma);
// Pdf of the later likelihood
// x        Time
// mu       Location parameter of the corresponding Gaussian
// sigma    Scale of the corresponding Gaussian

double later_lcdf(double x, double mu, double sigma);
// Pdf of the later likelihood
// x        Time
// mu       Location parameter of the corresponding Gaussian
// sigma    Scale of the corresponding Gaussian

double later_sf(double x, double mu, double sigma);
// Pdf of the later likelihood
// x        Time
// mu       Location parameter of the corresponding Gaussian
// sigma    Scale of the corresponding Gaussian

double later_lsf(double x, double mu, double sigma);
// Pdf of the later likelihood
// x        Time
// mu       Location parameter of the corresponding Gaussian
// sigma    Scale of the corresponding Gaussian

double wald_lpdf(double x, double mu, double sigma);
// Log pdf of the later likelihood
// x        Time
// mu       Location parameter of the corresponding Wald distribution
// sigma    Scale of the corresponding sigma parameter

// Probability density functions


double wald_pdf(double x, double mu, double lambda);
// Pdf of the later likelihood
// x        Time
// mu       Location parameter of the corresponding Gaussian
// sigma    Scale of the corresponding Gaussian

double wald_cdf(double x, double mu, double sigma);
// cdf of the wald likelihood
// x        Time
// mu       Location parameter of the corresponding Gaussian
// sigma    Scale of the corresponding Gaussian

double wald_lcdf(double x, double mu, double sigma);
// Log cdf of the wald likelihood
// x        Time
// mu       Location parameter of the corresponding Gaussian
// sigma    Scale of the corresponding Gaussian

double wald_sf(double x, double mu, double sigma);
// Survival function of the wald likelihood
// x        Time
// mu       Location parameter of the corresponding Gaussian
// sigma    Scale of the corresponding Gaussian

double wald_lsf(double x, double mu, double sigma);
// Survival function of the wald likelihood
// x        Time
// mu       Location parameter of the corresponding Gaussian
// sigma    Scale of the corresponding Gaussian


// Cumulative density functions

double later_cdf(double x, double mu, double sigma);
// cdf of the later likelihood
// x        Time
// mu       Location parameter of the corresponding Gaussian
// sigma    Scale of the corresponding Gaussian

double later_sf(double x, double mu, double sigma);
// Survival function (1-cdf) of the later likelihood
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

double ngamma_gslint(double t0, double x, double a, double b, double c0,
       double c1);
// Unnormalized nested incomplete gamma function.
// t0       -- Left boundary
// x        -- Integration boundary
// a        -- Parameters of the outer gamma function.
// b        -- Parameters of the inter gamma function.
// c0       -- Coefficient associated with a.
// c1       -- Coefficient associated with b.
//
// Uses gsl


double ninvgamma_gslint(double x0, double x1, double a, double b, double c0,
    double c1);
// Unnormalized nested incomplete gamma function.
// x0       -- Left integration boundary
// x1       -- right integration boundary
// a        -- Parameters of the outer gamma function.
// b        -- Parameters of the inter gamma function.
// c0       -- Coefficient associated with a.
// c1       -- Coefficient associated with b.
//
// Uses gls integrator.

double ninvgamma_high_precision_gslint(double x0, double x1, double a, 
        double b, double c0, double c1);
// Uses a very expensive, yet more accurate method to compute the 
// difficult integral.
// x0       -- Left integration boundary
// x1       -- right integration boundary
// a        -- Parameters of the outer gamma function.
// b        -- Parameters of the inter gamma function.
// c0       -- Coefficient associated with a.
// c1       -- Coefficient associated with b.
//
// Uses gls integrator.


double ninvgamma_lexp(double x, double a, double b, double c0, double c1);
// Unnormalized nested incomplete gamma function.
// x        -- Integration boundary
// a        -- Parameters of the outer gamma function.
// b        -- Parameters of the inter gamma function.
// c0       -- Coefficient associated with a.
// c1       -- Coefficient associated with b.
//
// Slow but safe.

double
nlognorm_gslint(double t0, double x, double mu1, double mu2, double sig1, 
        double sig2);
// Unnormalized nested incomplete gamma function.
// t0       -- Left bounday
// x        -- Integration boundary
// mu1      -- Location parameter of the outer lognormal function.
// mu2      -- Location parameter of the inner lognormal function.
// sig1     -- Scale parameter of the outer lognormal function.
// sig2     -- Scale parameter of the inner lognormal function.
//
// Uses gsldouble

double nlater_gslint(double t0, double x, double mu1, double mu2, double sig1,
       double sig2);
// Nested Integral for Gaussian Rates
// t0       -- Left boundary of integration 
// x        -- Right boundary of integration boundary
// mu1      -- Location parameter of the outer lognormal function.
// mu2      -- Location parameter of the inner lognormal function.
// sig1     -- Scale parameter of the outer lognormal function.
// sig2     -- Scale parameter of the inner lognormal function.
//
// Uses gsl

double nwald_gslint(double t0, double x, double mu1, double mu2, double sig1,
        double sig2);
// Nested Integral for Walt rt
// t0       -- Lower integration boundary
// x        -- Integration boundary
// mu1      -- Location parameter of the outer lognormal function.
// mu2      -- Location parameter of the inner lognormal function.
// sig1     -- Scale parameter of the outer lognormal function.
// sig2     -- Scale parameter of the inner lognormal function.
//
// Uses gsl


// Reparametrize

int
reparametrize_seria_gamma(const double *theta, SERIA_PARAMETERS *stheta);
// Transforms the parameters entered as arrays of double into structures that
// can be used by the more abstract functions.
//
// theta    -- Array of 12 parameters.
// stheta   -- Output, stucture with the parameters of the seria model.

int
reparametrize_seria_invgamma(const double *theta, SERIA_PARAMETERS *stheta);
// Transforms the parameters entered as arrays of double into structures that
// can be used by the more abstract functions.
//
// theta    -- Array of 12 parameters.
// stheta   -- Output, stucture with the parameters of the seria model.

int
reparametrize_seria_wald(const double *theta, SERIA_PARAMETERS *stheta);
// Transforms the parameters entered as arrays of double into structures that
// can be used by the more abstract functions.
//
// theta    -- Array of 12 parameters.
// stheta   -- Output, stucture with the parameters of the seria model.

int
reparametrize_seria_mixedgamma(const double *theta, SERIA_PARAMETERS *stheta);
// Transforms the parameters entered as arrays of double into structures that
// can be used by the more abstract functions.
//
// theta    -- Array of 12 parameters.
// stheta   -- Output, stucture with the parameters of the seria model.

int
reparametrize_seria_lognorm(const double *theta, SERIA_PARAMETERS *stheta);
// Transforms the parameters entered as arrays of double into structures that
// can be used by the more abstract functions.
//
// theta    -- Array of 12 parameters.
// stheta   -- Output, stucture with the parameters of the seria model.

int
reparametrize_seria_later(const double *theta, SERIA_PARAMETERS *stheta);
// Transforms the parameters entered as arrays of double into structures that
// can be used by the more abstract functions.
//
// theta    -- Array of 12 parameters.
// stheta   -- Output, stucture with the parameters of the seria model.

//

int
reparametrize_prosa_gamma(const double *theta, PROSA_PARAMETERS *stheta);
// Transforms the parameters entered as arrays of double into structures that
// can be used by the more abstract functions.
//
// theta    -- Array of 12 parameters.
// stheta   -- Output, stucture with the parameters of the prosa model.

int
reparametrize_prosa_invgamma(const double *theta, PROSA_PARAMETERS *stheta);
// Transforms the parameters entered as arrays of double into structures that
// can be used by the more abstract functions.
//
// theta    -- Array of 12 parameters.
// stheta   -- Output, stucture with the parameters of the prosa model.

int
reparametrize_prosa_wald(const double *theta, PROSA_PARAMETERS *stheta);
// Transforms the parameters entered as arrays of double into structures that
// can be used by the more abstract functions.
//
// theta    -- Array of 12 parameters.
// stheta   -- Output, stucture with the parameters of the prosa model.

int
reparametrize_prosa_mixedgamma(const double *theta, PROSA_PARAMETERS *stheta);
// Transforms the parameters entered as arrays of double into structures that
// can be used by the more abstract functions.
//
// theta    -- Array of 12 parameters.
// stheta   -- Output, stucture with the parameters of the prosa model.

int
reparametrize_prosa_lognorm(const double *theta, PROSA_PARAMETERS *stheta);
// Transforms the parameters entered as arrays of double into structures that
// can be used by the more abstract functions.
//
// theta    -- Array of 12 parameters.
// stheta   -- Output, stucture with the parameters of the prosa model.

int
reparametrize_prosa_later(const double *theta, PROSA_PARAMETERS *stheta);
// Transforms the parameters entered as arrays of double into structures that
// can be used by the more abstract functions.
//
// theta    -- Array of 12 parameters.
// stheta   -- Output, stucture with the parameters of the prosa model.

// Linearize turns parameter structures to doubles
int
linearize_seria(const SERIA_PARAMETERS *stheta, double *theta);

int
linearize_prosa(const PROSA_PARAMETERS *stheta, double *theta);

#endif
