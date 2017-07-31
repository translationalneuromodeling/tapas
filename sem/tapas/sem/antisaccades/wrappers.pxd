# Eduardo Aponte
# aponteeduardo@gmail.com (C) 2017 

cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cdef extern from "src/antisaccades/antisaccades.h":

    cdef int DIM_PROSA_THETA
    cdef int DIM_DORA_THETA
    cdef int DIM_SERI_THETA

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
        
    ctypedef struct SERI_PARAMETERS:

        double t0
        double p0

        double kp
        double tp
        double ka
        double ta
        double ks
        double ts

        double pp
        double ap

        double da

    ctypedef struct DORA_PARAMETERS:

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

    ctypedef double ( *PROSA_LLH )(double t, int a, PROSA_PARAMETERS params)
    ctypedef double ( *SERI_LLH )(double t, int a, SERI_PARAMETERS params)
    ctypedef double ( *DORA_LLH )(double t, int a, DORA_PARAMETERS params)

    ctypedef int (*FILL_PARAMETERS_PROSA)(const double *theta, PROSA_PARAMETERS
        *parameters)
    ctypedef int (*FILL_PARAMETERS_SERI)(const double *theta, SERI_PARAMETERS
        *parameters)
    ctypedef int (*FILL_PARAMETERS_DORA)(const double *theta, DORA_PARAMETERS
        *parameters)
    ctypedef double (*NESTED_INTEGRAL_SERI)(double x0, double x1, double a,
        double b, double c0, double c1)

    ctypedef struct PROSA_MODEL:
        FILL_PARAMETERS_PROSA fill_parameters
        PROSA_LLH llh

    ctypedef struct SERI_MODEL:
        FILL_PARAMETERS_SERI fill_parameters
        SERI_LLH llh
        NESTED_INTEGRAL_SERI nested_integral

    ctypedef struct DORA_MODEL:
        FILL_PARAMETERS_DORA fill_parameters
        DORA_LLH llh
        NESTED_INTEGRAL_SERI nested_integral

    cdef:
        double prosa_llh_gamma(double t, int a, PROSA_PARAMETERS params)

        double prosa_llh_invgamma(double t, int a, PROSA_PARAMETERS params)

        double prosa_llh_mixedgamma(double t, int a, PROSA_PARAMETERS params)

        double prosa_llh_lognorm(double t, int a, PROSA_PARAMETERS params)

        double prosa_llh_later(double t, int a, PROSA_PARAMETERS params)

        double prosa_llh_wald(double t, int a, PROSA_PARAMETERS params)

        double seri_llh_gamma(double t, int a, SERI_PARAMETERS params)

        double seri_llh_invgamma(double t, int a, SERI_PARAMETERS params)

        double seri_llh_mixedgamma(double t, int a, SERI_PARAMETERS params)

        double seri_llh_lognorm(double t, int a, SERI_PARAMETERS params)

        double seri_llh_later(double t, int a, SERI_PARAMETERS params)

        double seri_llh_wald(double t, int a, SERI_PARAMETERS params)

        double dora_llh_gamma(double t, int a, DORA_PARAMETERS params)

        double dora_llh_invgamma(double t, int a, DORA_PARAMETERS params)

        double dora_llh_mixedgamma(double t, int a, DORA_PARAMETERS params)

        double dora_llh_lognorm(double t, int a, DORA_PARAMETERS params)

        double dora_llh_later(double t, int a, DORA_PARAMETERS params)

        double dora_llh_wald(double t, int a, DORA_PARAMETERS params)

        int prosa_model_trial_by_trial(ANTIS_INPUT svals, PROSA_MODEL fllh, 
                double *llh)
        
        int prosa_model_two_states(ANTIS_INPUT svals, PROSA_MODEL fllh, 
                double *llh)

        int dora_model_trial_by_trial(ANTIS_INPUT svals, DORA_MODEL fllh, 
            double *llh)

        int dora_model_two_states(ANTIS_INPUT svals, DORA_MODEL fllh, 
            double *llh)

        int seri_model_trial_by_trial(ANTIS_INPUT svals, SERI_MODEL fllh, 
                double *llh)

        int seri_model_two_states(ANTIS_INPUT svals, SERI_MODEL fllh, 
                double *llh)

        int populate_parameters_prosa(const double *theta, 
                PROSA_PARAMETERS *stheta)
        
        int populate_parameters_seri(const double *theta, 
                SERI_PARAMETERS *stheta)

        int populate_parameters_dora(const double *theta, 
                DORA_PARAMETERS *stheta)

        # Reparametrization
        int reparametrize_seri_invgamma(const double *theta, 
                SERI_PARAMETERS *stheta)
        int reparametrize_seri_gamma(const double *theta, 
                SERI_PARAMETERS *stheta)
        int reparametrize_seri_mixedgamma(const double *theta, 
                SERI_PARAMETERS *stheta)
        int reparametrize_seri_lognorm(const double *theta, 
                SERI_PARAMETERS *stheta)
        int reparametrize_seri_later(const double *theta, 
                SERI_PARAMETERS *stheta)
        int reparametrize_seri_wald(const double *theta, 
                SERI_PARAMETERS *stheta)

        int reparametrize_dora_invgamma(const double *theta, 
                DORA_PARAMETERS *stheta)
        int reparametrize_dora_gamma(const double *theta, 
                DORA_PARAMETERS *stheta)
        int reparametrize_dora_mixedgamma(const double *theta, 
                DORA_PARAMETERS *stheta)
        int reparametrize_dora_lognorm(const double *theta, 
                DORA_PARAMETERS *stheta)
        int reparametrize_dora_later(const double *theta, 
                DORA_PARAMETERS *stheta)
        int reparametrize_dora_wald(const double *theta, 
                DORA_PARAMETERS *stheta)

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


