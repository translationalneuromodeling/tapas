# Eduardo Aponte
# aponteeduardo@gmail.com (C) 2017 

cimport cython
import numpy as np
cimport numpy as np
cimport wrappers

cdef wrapper_seri_model_two_states(
        FILL_PARAMETERS_SERI fill,
        SERI_LLH fllh,
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    cdef int nd = t.size
    cdef int nt = theta.size/DIM_SERI_THETA

    cdef np.ndarray[np.float64_t, ndim=1] llh = np.empty(nd, dtype=np.float64)
    cdef ANTIS_INPUT svals
    cdef SERI_MODEL model

    assert len(t) == len(a), 'Arrays t, a, and u should have the same size.'
    assert len(a) == len(u), 'Arrays t, a, and u should have the same size.'
    assert len(theta) == nt * DIM_SERI_THETA, 'Please check len(theta)'

    svals.t = <np.float64_t *> t.data
    svals.a = <np.float64_t *> a.data
    svals.u = <np.float64_t *> u.data
    svals.np = nt
    svals.theta = <np.float64_t *> theta.data
    svals.nt = nd

    model.fill_parameters = fill
    model.llh = fllh

    seri_model_two_states(svals, model, <np.float64_t *> llh.data)

    return llh

cdef wrapper_dora_model_two_states(
        FILL_PARAMETERS_DORA fill,
        DORA_LLH fllh,
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    cdef int nd = t.size
    cdef int nt = theta.size/DIM_DORA_THETA

    cdef np.ndarray[np.float64_t, ndim=1] llh = np.empty(nd, dtype=np.float64)
    cdef ANTIS_INPUT svals
    cdef DORA_MODEL model

    assert len(t) == len(a), 'Arrays t, a, and u should have the same size.'
    assert len(a) == len(u), 'Arrays t, a, and u should have the same size.'
    assert len(theta) == nt * DIM_DORA_THETA, 'Please check len(theta)'

    svals.t = <np.float64_t *> t.data
    svals.a = <np.float64_t *> a.data
    svals.u = <np.float64_t *> u.data
    svals.np = nt
    svals.theta = <np.float64_t *> theta.data
    svals.nt = nd

    model.fill_parameters = fill 
    model.llh = fllh

    dora_model_two_states(svals, model, <np.float64_t *> llh.data)

    return llh

cdef wrapper_prosa_model_two_states(
        FILL_PARAMETERS_PROSA fill,
        PROSA_LLH fllh,        
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    cdef int nd = t.size
    cdef int nt = theta.size/DIM_PROSA_THETA

    cdef np.ndarray[np.float64_t, ndim=1] llh = np.empty(nd, dtype=np.float64)
    cdef ANTIS_INPUT svals
    cdef PROSA_MODEL model

    assert len(t) == len(a), 'Arrays t, a, and u should have the same size.'
    assert len(a) == len(u), 'Arrays t, a, and u should have the same size.'
    assert len(theta) == nt * DIM_PROSA_THETA, 'Please check len(theta)'

    svals.t = <np.float64_t *> t.data
    svals.a = <np.float64_t *> a.data
    svals.u = <np.float64_t *> u.data
    svals.np = nt
    svals.theta = <np.float64_t *> theta.data
    svals.nt = nd

    model.fill_parameters = fill 
    model.llh = fllh

    prosa_model_two_states(svals, model, <np.float64_t *> llh.data)

    return llh

cdef wrapper_reparametrize_dora(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta,
        FILL_PARAMETERS_DORA fparam):

    cdef DORA_PARAMETERS cparams[1]

    assert len(theta) == DIM_DORA_THETA, 'Please check len(theta)'
   
    fparam(<np.float64_t *> theta.data, cparams)
    ntheta = {
        'kp' : cparams[0].kp,
        'tp' : cparams[0].tp,
        'ka' : cparams[0].ka,
        'ta' : cparams[0].ta,
        'ks' : cparams[0].ks,
        'ts' : cparams[0].ts,
        'kl' : cparams[0].kl,
        'tl' : cparams[0].tl,
        't0' : cparams[0].t0,
        'da' : cparams[0].da,
        'p0' : cparams[0].p0}

    return ntheta

cdef wrapper_reparametrize_seri(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta,
        FILL_PARAMETERS_SERI fparam):
        
    cdef SERI_PARAMETERS cparams[1]

    assert len(theta) == DIM_SERI_THETA, 'Please check len(theta)'
    
    fparam(<np.float64_t *> theta.data, cparams)
    ntheta = {
        'kp' : cparams[0].kp,
        'tp' : cparams[0].tp,
        'ka' : cparams[0].ka,
        'ta' : cparams[0].ta,
        'ks' : cparams[0].ks,
        'ts' : cparams[0].ts,
        'pp' : cparams[0].pp,
        'ap' : cparams[0].ap,
        't0' : cparams[0].t0,
        'da' : cparams[0].da,
        'p0' : cparams[0].p0}

    return ntheta

cdef wrapper_reparametrize_prosa(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta,
        FILL_PARAMETERS_PROSA fparam):
        
    cdef PROSA_PARAMETERS cparams[1]

    assert len(theta) == DIM_PROSA_THETA, 'Please check len(theta)'
    
    fparam(<np.float64_t *> theta.data, cparams)
    ntheta = {
        'kp' : cparams[0].kp,
        'tp' : cparams[0].tp,
        'ka' : cparams[0].ka,
        'ta' : cparams[0].ta,
        'ks' : cparams[0].ks,
        'ts' : cparams[0].ts,
        't0' : cparams[0].t0,
        'da' : cparams[0].da,
        'p0' : cparams[0].p0}

    return ntheta


def p_seri_model_two_states_gamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_seri_model_two_states(reparametrize_seri_gamma,
        seri_llh_gamma, t, a, u, theta)


def p_reparametrize_seri_gamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    ntheta = wrapper_reparametrize_seri(theta, reparametrize_seri_gamma)

    return ntheta

def p_seri_model_two_states_invgamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_seri_model_two_states(reparametrize_seri_invgamma,
        seri_llh_invgamma, t, a, u, theta)


def p_reparametrize_seri_invgamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    ntheta = wrapper_reparametrize_seri(theta, reparametrize_seri_invgamma)

    return ntheta

def p_seri_model_two_states_mixedgamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_seri_model_two_states(reparametrize_seri_mixedgamma,
        seri_llh_mixedgamma, t, a, u, theta)


def p_reparametrize_seri_mixedgamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    ntheta = wrapper_reparametrize_seri(theta, reparametrize_seri_mixedgamma)

    return ntheta

def p_seri_model_two_states_lognorm(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_seri_model_two_states(reparametrize_seri_lognorm,
        seri_llh_lognorm, t, a, u, theta)


def p_reparametrize_seri_lognorm(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    ntheta = wrapper_reparametrize_seri(theta, reparametrize_seri_lognorm)

    return ntheta

def p_seri_model_two_states_later(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_seri_model_two_states(reparametrize_seri_later,
        seri_llh_later, t, a, u, theta)


def p_reparametrize_seri_later(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    ntheta = wrapper_reparametrize_seri(theta, reparametrize_seri_later)

    return ntheta

def p_seri_model_two_states_wald(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_seri_model_two_states(reparametrize_seri_wald,
        seri_llh_wald, t, a, u, theta)


def p_reparametrize_seri_wald(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    ntheta = wrapper_reparametrize_seri(theta, reparametrize_seri_wald)

    return ntheta

def p_dora_model_two_states_gamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_dora_model_two_states(reparametrize_dora_gamma,
        dora_llh_gamma, t, a, u, theta)


def p_reparametrize_dora_gamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    ntheta = wrapper_reparametrize_dora(theta, reparametrize_dora_gamma)

    return ntheta

def p_dora_model_two_states_invgamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_dora_model_two_states(reparametrize_dora_invgamma,
        dora_llh_invgamma, t, a, u, theta)


def p_reparametrize_dora_invgamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    ntheta = wrapper_reparametrize_dora(theta, reparametrize_dora_invgamma)

    return ntheta

def p_dora_model_two_states_mixedgamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_dora_model_two_states(reparametrize_dora_mixedgamma,
        dora_llh_mixedgamma, t, a, u, theta)


def p_reparametrize_dora_mixedgamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    ntheta = wrapper_reparametrize_dora(theta, reparametrize_dora_mixedgamma)

    return ntheta

def p_dora_model_two_states_lognorm(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_dora_model_two_states(reparametrize_dora_lognorm,
        dora_llh_lognorm, t, a, u, theta)


def p_reparametrize_dora_lognorm(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    ntheta = wrapper_reparametrize_dora(theta, reparametrize_dora_lognorm)

    return ntheta

def p_dora_model_two_states_later(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_dora_model_two_states(reparametrize_dora_later,
        dora_llh_later, t, a, u, theta)


def p_reparametrize_dora_later(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    ntheta = wrapper_reparametrize_dora(theta, reparametrize_dora_later)

    return ntheta

def p_dora_model_two_states_wald(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_dora_model_two_states(reparametrize_dora_wald,
        dora_llh_wald, t, a, u, theta)


def p_reparametrize_dora_wald(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    ntheta = wrapper_reparametrize_dora(theta, reparametrize_dora_wald)

    return ntheta

def p_prosa_model_two_states_gamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_prosa_model_two_states(reparametrize_prosa_gamma,
        prosa_llh_gamma, t, a, u, theta)


def p_reparametrize_prosa_gamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    ntheta = wrapper_reparametrize_prosa(theta, reparametrize_prosa_gamma)

    return ntheta

def p_prosa_model_two_states_invgamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_prosa_model_two_states(reparametrize_prosa_invgamma,
        prosa_llh_invgamma, t, a, u, theta)


def p_reparametrize_prosa_invgamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    ntheta = wrapper_reparametrize_prosa(theta, reparametrize_prosa_invgamma)

    return ntheta

def p_prosa_model_two_states_mixedgamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_prosa_model_two_states(reparametrize_prosa_mixedgamma,
        prosa_llh_mixedgamma, t, a, u, theta)


def p_reparametrize_prosa_mixedgamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    ntheta = wrapper_reparametrize_prosa(theta, reparametrize_prosa_mixedgamma)

    return ntheta

def p_prosa_model_two_states_lognorm(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_prosa_model_two_states(reparametrize_prosa_lognorm,
        prosa_llh_lognorm, t, a, u, theta)


def p_reparametrize_prosa_lognorm(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    ntheta = wrapper_reparametrize_prosa(theta, reparametrize_prosa_lognorm)

    return ntheta

def p_prosa_model_two_states_later(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_prosa_model_two_states(reparametrize_prosa_later,
        prosa_llh_later, t, a, u, theta)


def p_reparametrize_prosa_later(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    ntheta = wrapper_reparametrize_prosa(theta, reparametrize_prosa_later)

    return ntheta

def p_prosa_model_two_states_wald(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_prosa_model_two_states(reparametrize_prosa_wald,
        prosa_llh_wald, t, a, u, theta)


def p_reparametrize_prosa_wald(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    ntheta = wrapper_reparametrize_prosa(theta, reparametrize_prosa_wald)

    return ntheta

