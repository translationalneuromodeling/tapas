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

        double kinv
        double tinv
        double kva
        double tva
        double kvp
        double tvp

        double ks
        double ts

        double pp
        double ap

        double da



    ctypedef double ( *PROSA_LLH )(double t, int a, PROSA_PARAMETERS params)
    ctypedef double ( *SERI_LLH )(double t, int a, SERI_PARAMETERS params)
    ctypedef double ( *DORA_LLH )(double t, int a, DORA_PARAMETERS params)


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

        int prosa_model_trial_by_trial(ANTIS_INPUT svals, PROSA_LLH fllh, 
                double *llh)
        int prosa_model_two_states(ANTIS_INPUT svals, PROSA_LLH fllh, double *llh)

        int dora_model_trial_by_trial(ANTIS_INPUT svals, DORA_LLH fllh, 
            double *llh)

        int dora_model_two_states(ANTIS_INPUT svals, DORA_LLH fllh, double *llh)

        int seri_model_trial_by_trial(ANTIS_INPUT svals, SERI_LLH fllh, 
                double *llh)

        int seri_model_two_states(ANTIS_INPUT svals, SERI_LLH fllh, double *llh)



def p_seri_model_two_states_gamma(np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    cdef int nd = t.size
    cdef int nt = theta.size/DIM_SERI_THETA

    cdef np.ndarray[np.float64_t, ndim=1] llh = np.empty(nd, dtype=np.float64)
    cdef ANTIS_INPUT svals

    assert len(t) == len(a), 'Arrays t, a, and u should have the same size.'
    assert len(a) == len(u), 'Arrays t, a, and u should have the same size.'
    assert len(theta) == nt * DIM_SERI_THETA, 'Please check len(theta)'

    svals.t = <np.float64_t *> t.data
    svals.a = <np.float64_t *> a.data
    svals.u = <np.float64_t *> u.data
    svals.np = nt
    svals.theta = <np.float64_t *> theta.data
    svals.nt = nd

    seri_model_two_states(svals, seri_llh_gamma, <np.float64_t *> llh.data)

    return llh



def p_seri_model_two_states_invgamma(np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    cdef int nd = t.size
    cdef int nt = theta.size/DIM_SERI_THETA

    cdef np.ndarray[np.float64_t, ndim=1] llh = np.empty(nd, dtype=np.float64)
    cdef ANTIS_INPUT svals

    assert len(t) == len(a), 'Arrays t, a, and u should have the same size.'
    assert len(a) == len(u), 'Arrays t, a, and u should have the same size.'
    assert len(theta) == nt * DIM_SERI_THETA, 'Please check len(theta)'

    svals.t = <np.float64_t *> t.data
    svals.a = <np.float64_t *> a.data
    svals.u = <np.float64_t *> u.data
    svals.np = nt
    svals.theta = <np.float64_t *> theta.data
    svals.nt = nd

    seri_model_two_states(svals, seri_llh_invgamma, <np.float64_t *> llh.data)

    return llh



def p_seri_model_two_states_mixedgamma(np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    cdef int nd = t.size
    cdef int nt = theta.size/DIM_SERI_THETA

    cdef np.ndarray[np.float64_t, ndim=1] llh = np.empty(nd, dtype=np.float64)
    cdef ANTIS_INPUT svals

    assert len(t) == len(a), 'Arrays t, a, and u should have the same size.'
    assert len(a) == len(u), 'Arrays t, a, and u should have the same size.'
    assert len(theta) == nt * DIM_SERI_THETA, 'Please check len(theta)'

    svals.t = <np.float64_t *> t.data
    svals.a = <np.float64_t *> a.data
    svals.u = <np.float64_t *> u.data
    svals.np = nt
    svals.theta = <np.float64_t *> theta.data
    svals.nt = nd

    seri_model_two_states(svals, seri_llh_mixedgamma, <np.float64_t *> llh.data)

    return llh



def p_seri_model_two_states_later(np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    cdef int nd = t.size
    cdef int nt = theta.size/DIM_SERI_THETA

    cdef np.ndarray[np.float64_t, ndim=1] llh = np.empty(nd, dtype=np.float64)
    cdef ANTIS_INPUT svals

    assert len(t) == len(a), 'Arrays t, a, and u should have the same size.'
    assert len(a) == len(u), 'Arrays t, a, and u should have the same size.'
    assert len(theta) == nt * DIM_SERI_THETA, 'Please check len(theta)'

    svals.t = <np.float64_t *> t.data
    svals.a = <np.float64_t *> a.data
    svals.u = <np.float64_t *> u.data
    svals.np = nt
    svals.theta = <np.float64_t *> theta.data
    svals.nt = nd

    seri_model_two_states(svals, seri_llh_later, <np.float64_t *> llh.data)

    return llh



def p_seri_model_two_states_wald(np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    cdef int nd = t.size
    cdef int nt = theta.size/DIM_SERI_THETA

    cdef np.ndarray[np.float64_t, ndim=1] llh = np.empty(nd, dtype=np.float64)
    cdef ANTIS_INPUT svals

    assert len(t) == len(a), 'Arrays t, a, and u should have the same size.'
    assert len(a) == len(u), 'Arrays t, a, and u should have the same size.'
    assert len(theta) == nt * DIM_SERI_THETA, 'Please check len(theta)'

    svals.t = <np.float64_t *> t.data
    svals.a = <np.float64_t *> a.data
    svals.u = <np.float64_t *> u.data
    svals.np = nt
    svals.theta = <np.float64_t *> theta.data
    svals.nt = nd

    seri_model_two_states(svals, seri_llh_wald, <np.float64_t *> llh.data)

    return llh



def p_dora_model_two_states_gamma(np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    cdef int nd = t.size
    cdef int nt = theta.size/DIM_DORA_THETA

    cdef np.ndarray[np.float64_t, ndim=1] llh = np.empty(nd, dtype=np.float64)
    cdef ANTIS_INPUT svals

    assert len(t) == len(a), 'Arrays t, a, and u should have the same size.'
    assert len(a) == len(u), 'Arrays t, a, and u should have the same size.'
    assert len(theta) == nt * DIM_DORA_THETA, 'Please check len(theta)'

    svals.t = <np.float64_t *> t.data
    svals.a = <np.float64_t *> a.data
    svals.u = <np.float64_t *> u.data
    svals.np = nt
    svals.theta = <np.float64_t *> theta.data
    svals.nt = nd

    dora_model_two_states(svals, dora_llh_gamma, <np.float64_t *> llh.data)

    return llh



def p_dora_model_two_states_invgamma(np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    cdef int nd = t.size
    cdef int nt = theta.size/DIM_DORA_THETA

    cdef np.ndarray[np.float64_t, ndim=1] llh = np.empty(nd, dtype=np.float64)
    cdef ANTIS_INPUT svals

    assert len(t) == len(a), 'Arrays t, a, and u should have the same size.'
    assert len(a) == len(u), 'Arrays t, a, and u should have the same size.'
    assert len(theta) == nt * DIM_DORA_THETA, 'Please check len(theta)'

    svals.t = <np.float64_t *> t.data
    svals.a = <np.float64_t *> a.data
    svals.u = <np.float64_t *> u.data
    svals.np = nt
    svals.theta = <np.float64_t *> theta.data
    svals.nt = nd

    dora_model_two_states(svals, dora_llh_invgamma, <np.float64_t *> llh.data)

    return llh



def p_dora_model_two_states_mixedgamma(np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    cdef int nd = t.size
    cdef int nt = theta.size/DIM_DORA_THETA

    cdef np.ndarray[np.float64_t, ndim=1] llh = np.empty(nd, dtype=np.float64)
    cdef ANTIS_INPUT svals

    assert len(t) == len(a), 'Arrays t, a, and u should have the same size.'
    assert len(a) == len(u), 'Arrays t, a, and u should have the same size.'
    assert len(theta) == nt * DIM_DORA_THETA, 'Please check len(theta)'

    svals.t = <np.float64_t *> t.data
    svals.a = <np.float64_t *> a.data
    svals.u = <np.float64_t *> u.data
    svals.np = nt
    svals.theta = <np.float64_t *> theta.data
    svals.nt = nd

    dora_model_two_states(svals, dora_llh_mixedgamma, <np.float64_t *> llh.data)

    return llh



def p_dora_model_two_states_later(np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    cdef int nd = t.size
    cdef int nt = theta.size/DIM_DORA_THETA

    cdef np.ndarray[np.float64_t, ndim=1] llh = np.empty(nd, dtype=np.float64)
    cdef ANTIS_INPUT svals

    assert len(t) == len(a), 'Arrays t, a, and u should have the same size.'
    assert len(a) == len(u), 'Arrays t, a, and u should have the same size.'
    assert len(theta) == nt * DIM_DORA_THETA, 'Please check len(theta)'

    svals.t = <np.float64_t *> t.data
    svals.a = <np.float64_t *> a.data
    svals.u = <np.float64_t *> u.data
    svals.np = nt
    svals.theta = <np.float64_t *> theta.data
    svals.nt = nd

    dora_model_two_states(svals, dora_llh_later, <np.float64_t *> llh.data)

    return llh



def p_dora_model_two_states_wald(np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    cdef int nd = t.size
    cdef int nt = theta.size/DIM_DORA_THETA

    cdef np.ndarray[np.float64_t, ndim=1] llh = np.empty(nd, dtype=np.float64)
    cdef ANTIS_INPUT svals

    assert len(t) == len(a), 'Arrays t, a, and u should have the same size.'
    assert len(a) == len(u), 'Arrays t, a, and u should have the same size.'
    assert len(theta) == nt * DIM_DORA_THETA, 'Please check len(theta)'

    svals.t = <np.float64_t *> t.data
    svals.a = <np.float64_t *> a.data
    svals.u = <np.float64_t *> u.data
    svals.np = nt
    svals.theta = <np.float64_t *> theta.data
    svals.nt = nd

    dora_model_two_states(svals, dora_llh_wald, <np.float64_t *> llh.data)

    return llh



def p_prosa_model_two_states_gamma(np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    cdef int nd = t.size
    cdef int nt = theta.size/DIM_PROSA_THETA

    cdef np.ndarray[np.float64_t, ndim=1] llh = np.empty(nd, dtype=np.float64)
    cdef ANTIS_INPUT svals

    assert len(t) == len(a), 'Arrays t, a, and u should have the same size.'
    assert len(a) == len(u), 'Arrays t, a, and u should have the same size.'
    assert len(theta) == nt * DIM_PROSA_THETA, 'Please check len(theta)'

    svals.t = <np.float64_t *> t.data
    svals.a = <np.float64_t *> a.data
    svals.u = <np.float64_t *> u.data
    svals.np = nt
    svals.theta = <np.float64_t *> theta.data
    svals.nt = nd

    prosa_model_two_states(svals, prosa_llh_gamma, <np.float64_t *> llh.data)

    return llh



def p_prosa_model_two_states_invgamma(np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    cdef int nd = t.size
    cdef int nt = theta.size/DIM_PROSA_THETA

    cdef np.ndarray[np.float64_t, ndim=1] llh = np.empty(nd, dtype=np.float64)
    cdef ANTIS_INPUT svals

    assert len(t) == len(a), 'Arrays t, a, and u should have the same size.'
    assert len(a) == len(u), 'Arrays t, a, and u should have the same size.'
    assert len(theta) == nt * DIM_PROSA_THETA, 'Please check len(theta)'

    svals.t = <np.float64_t *> t.data
    svals.a = <np.float64_t *> a.data
    svals.u = <np.float64_t *> u.data
    svals.np = nt
    svals.theta = <np.float64_t *> theta.data
    svals.nt = nd

    prosa_model_two_states(svals, prosa_llh_invgamma, <np.float64_t *> llh.data)

    return llh



def p_prosa_model_two_states_mixedgamma(np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    cdef int nd = t.size
    cdef int nt = theta.size/DIM_PROSA_THETA

    cdef np.ndarray[np.float64_t, ndim=1] llh = np.empty(nd, dtype=np.float64)
    cdef ANTIS_INPUT svals

    assert len(t) == len(a), 'Arrays t, a, and u should have the same size.'
    assert len(a) == len(u), 'Arrays t, a, and u should have the same size.'
    assert len(theta) == nt * DIM_PROSA_THETA, 'Please check len(theta)'

    svals.t = <np.float64_t *> t.data
    svals.a = <np.float64_t *> a.data
    svals.u = <np.float64_t *> u.data
    svals.np = nt
    svals.theta = <np.float64_t *> theta.data
    svals.nt = nd

    prosa_model_two_states(svals, prosa_llh_mixedgamma, <np.float64_t *> llh.data)

    return llh



def p_prosa_model_two_states_later(np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    cdef int nd = t.size
    cdef int nt = theta.size/DIM_PROSA_THETA

    cdef np.ndarray[np.float64_t, ndim=1] llh = np.empty(nd, dtype=np.float64)
    cdef ANTIS_INPUT svals

    assert len(t) == len(a), 'Arrays t, a, and u should have the same size.'
    assert len(a) == len(u), 'Arrays t, a, and u should have the same size.'
    assert len(theta) == nt * DIM_PROSA_THETA, 'Please check len(theta)'

    svals.t = <np.float64_t *> t.data
    svals.a = <np.float64_t *> a.data
    svals.u = <np.float64_t *> u.data
    svals.np = nt
    svals.theta = <np.float64_t *> theta.data
    svals.nt = nd

    prosa_model_two_states(svals, prosa_llh_later, <np.float64_t *> llh.data)

    return llh



def p_prosa_model_two_states_wald(np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    cdef int nd = t.size
    cdef int nt = theta.size/DIM_PROSA_THETA

    cdef np.ndarray[np.float64_t, ndim=1] llh = np.empty(nd, dtype=np.float64)
    cdef ANTIS_INPUT svals

    assert len(t) == len(a), 'Arrays t, a, and u should have the same size.'
    assert len(a) == len(u), 'Arrays t, a, and u should have the same size.'
    assert len(theta) == nt * DIM_PROSA_THETA, 'Please check len(theta)'

    svals.t = <np.float64_t *> t.data
    svals.a = <np.float64_t *> a.data
    svals.u = <np.float64_t *> u.data
    svals.np = nt
    svals.theta = <np.float64_t *> theta.data
    svals.nt = nd

    prosa_model_two_states(svals, prosa_llh_wald, <np.float64_t *> llh.data)

    return llh


