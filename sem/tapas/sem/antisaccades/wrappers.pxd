# Eduardo Aponte
# aponteeduardo@gmail.com (C) 2017 

cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cdef extern from "src/antisaccades/antisaccades.h":

    cdef int DIM_PROSA_THETA
    cdef int DIM_SERIA_THETA

    ctypedef double (*NESTED_INTEGRAL)(double x0, double x1, double a,
        double b, double c0, double c1)

    ctypedef double (*ACCUMULATOR_FUNCTION)(double time, double shape,
        double scale)

    ctypedef struct ACCUMULATOR:

        ACCUMULATOR_FUNCTION pdf
        ACCUMULATOR_FUNCTION lpdf

        ACCUMULATOR_FUNCTION cdf
        ACCUMULATOR_FUNCTION lcdf

        ACCUMULATOR_FUNCTION sf
        ACCUMULATOR_FUNCTION lsf 

    ctypedef struct ANTIS_INPUT:
        double *t
        double *a
        double *u
        double *theta

        int np
        int nt

    ctypedef struct PROSA_PARAMETERS:

        double t0
        double p0

        double kp
        double tp
        double ka
        double ta
        double ks
        double ts

        double da

        ACCUMULATOR early
        ACCUMULATOR stop
        ACCUMULATOR anti

        NESTED_INTEGRAL inhibition_race
        
    ctypedef struct SERIA_PARAMETERS:

        double t0
        double p0

        double kp
        double tp
        double ka
        double ta
        double kl
        double tl

        double ks
        double ts

        double da

        ACCUMULATOR early
        ACCUMULATOR stop
        ACCUMULATOR anti
        ACCUMULATOR late

        NESTED_INTEGRAL inhibition_race

    ctypedef double ( *PROSA_LLH )(double t, int a, PROSA_PARAMETERS params)
    ctypedef double ( *SERIA_LLH )(double t, int a, SERIA_PARAMETERS params)

    ctypedef int (*FILL_PARAMETERS_PROSA)(const double *theta, PROSA_PARAMETERS
        *parameters)
    ctypedef int (*FILL_PARAMETERS_SERIA)(const double *theta, SERIA_PARAMETERS
        *parameters)

    ctypedef struct PROSA_MODEL:
        FILL_PARAMETERS_PROSA fill_parameters
        PROSA_LLH llh

    ctypedef struct SERIA_MODEL:
        FILL_PARAMETERS_SERIA fill_parameters
        SERIA_LLH llh

    cdef:
        double prosa_llh_abstract(double t, int a, PROSA_PARAMETERS params)

        double seria_llh_abstract(double t, int a, SERIA_PARAMETERS params)

        double seria_early_llh_abstract(double t, int a, SERIA_PARAMETERS params)

        int prosa_model_n_states_optimized(ANTIS_INPUT svals, 
                PROSA_MODEL fllh, double *llh)
        int prosa_model_n_states(ANTIS_INPUT svals, 
                PROSA_MODEL fllh, double *llh)

        int seria_model_n_states_optimized(ANTIS_INPUT svals, 
                SERIA_MODEL fllh, double *llh)
        int seria_model_n_states(ANTIS_INPUT svals, 
                SERIA_MODEL fllh, double *llh)


        int populate_parameters_prosa(const double *theta, 
                PROSA_PARAMETERS *stheta)
        
        int populate_parameters_seria(const double *theta, 
                SERIA_PARAMETERS *stheta)

        # Reparametrization
        int reparametrize_seria_invgamma(const double *theta, 
                SERIA_PARAMETERS *stheta)
        int reparametrize_seria_gamma(const double *theta, 
                SERIA_PARAMETERS *stheta)
        int reparametrize_seria_mixedgamma(const double *theta, 
                SERIA_PARAMETERS *stheta)
        int reparametrize_seria_lognorm(const double *theta, 
                SERIA_PARAMETERS *stheta)
        int reparametrize_seria_later(const double *theta, 
                SERIA_PARAMETERS *stheta)
        int reparametrize_seria_wald(const double *theta, 
                SERIA_PARAMETERS *stheta)

        int reparametrize_prosa_invgamma(const double *theta, 
                PROSA_PARAMETERS *stheta)
        int reparametrize_prosa_gamma(const double *theta, 
                PROSA_PARAMETERS *stheta)
        int reparametrize_prosa_mixedgamma(const double *theta, 
                PROSA_PARAMETERS *stheta)
        int reparametrize_prosa_lognorm(const double *theta, 
                PROSA_PARAMETERS *stheta)
        int reparametrize_prosa_later(const double *theta, 
                PROSA_PARAMETERS *stheta)
        int reparametrize_prosa_wald(const double *theta, 
                PROSA_PARAMETERS *stheta)

        double ngamma_gslint(double t0, double x, double a, double b, 
		double c0, double c1)
        double ninvgamma_gslint(double x0, double x1, double a, double b, 
                double c0, double c1)
        double nlater_gslint(double t0, double x, double mu1, double mu2,
		double sig1, double sig2)
        double nwald_gslint(double t0, double x, double mu1, double mu2, 
		double sig1, double sig2)
