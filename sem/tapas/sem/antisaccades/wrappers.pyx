# Eduardo Aponte
# aponteeduardo@gmail.com (C) 2017 

cimport cython
import numpy as np
cimport numpy as np
cimport wrappers

cdef wrapper_seria_model_n_states(
        FILL_PARAMETERS_SERIA fill,
        SERIA_LLH fllh,
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    cdef int nd = t.size
    cdef int nt = theta.size/DIM_SERIA_THETA

    cdef np.ndarray[np.float64_t, ndim=1] llh = np.empty(nd, dtype=np.float64)
    cdef ANTIS_INPUT svals
    cdef SERIA_MODEL model

    assert len(t) == len(a), 'Arrays t, a, and u should have the same size.'
    assert len(a) == len(u), 'Arrays t, a, and u should have the same size.'
    assert len(theta) == nt * DIM_SERIA_THETA, 'Please check len(theta)'

    svals.t = <np.float64_t *> t.data
    svals.a = <np.float64_t *> a.data
    svals.u = <np.float64_t *> u.data
    svals.np = nt
    svals.theta = <np.float64_t *> theta.data
    svals.nt = nd

    model.fill_parameters = fill 
    model.llh = fllh

    seria_model_n_states(svals, model, <np.float64_t *> llh.data)

    return llh

cdef wrapper_prosa_model_n_states(
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

    prosa_model_n_states(svals, model, <np.float64_t *> llh.data)

    return llh

cdef wrapper_reparametrize_seria(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta,
        FILL_PARAMETERS_SERIA fparam):

    cdef SERIA_PARAMETERS cparams[1]

    assert len(theta) == DIM_SERIA_THETA, 'Please check len(theta)'
   
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


cdef wrapper_summary_seria(
        FILL_PARAMETERS_SERIA f_init_parameters,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    # C struct with summary
    cdef SERIA_SUMMARY c_summary[1]
    # C struct with parameters
    cdef SERIA_PARAMETERS c_params[1]

    assert len(theta) == DIM_SERIA_THETA, 'Please check len(theta)'
   
    # Initilize the parametes 
    f_init_parameters(<np.float64_t *> theta.data, c_params)

    # Otherwise python goes crazy
    gsl_set_error_handler_off()

    # Fill the summary
    seria_summary_abstract(c_params, c_summary)
    
    # Turn it on again
    gsl_set_error_handler(NULL)

    # Python dictionary with the summary
    p_summary = {
        'late_pro_rt' : c_summary[0].late_pro_rt,
        'anti_rt' : c_summary[0].anti_rt,
        'inhib_fail_rt' : c_summary[0].inhib_fail_rt,
        'inhib_fail_prob' : c_summary[0].inhib_fail_prob,
        'late_pro_prob' : c_summary[0].late_pro_prob,
        'predicted_pro_prob' : c_summary[0].predicted_pro_prob,
        'predicted_pro_rt' : c_summary[0].predicted_pro_rt,
        'predicted_anti_prob' : c_summary[0].predicted_anti_prob,
        'predicted_anti_rt' : c_summary[0].predicted_anti_rt,
        }

    return p_summary


cdef wrapper_summary_prosa(
        FILL_PARAMETERS_PROSA f_init_parameters, 
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    # C struct with summary
    cdef PROSA_SUMMARY c_summary[1]
    # C struct with parameters
    cdef PROSA_PARAMETERS c_params[1]

    assert len(theta) == DIM_PROSA_THETA, 'Please check len(theta)'
   
    # Initilize the parametes 
    f_init_parameters(<np.float64_t *> theta.data, c_params)

    # Otherwise python goes crazy
    gsl_set_error_handler_off()

    # Fill the summary
    prosa_summary_abstract(c_params, c_summary)
    
    # Turn it on again
    gsl_set_error_handler(NULL)


    # Python dictionary with the summary
    p_summary = {
        'anti_rt' : c_summary[0].anti_rt,
        'inhib_fail_rt' : c_summary[0].inhib_fail_rt,
        'inhib_fail_prob' : c_summary[0].inhib_fail_prob,
        'predicted_pro_prob' : c_summary[0].predicted_pro_prob,
        'predicted_pro_rt' : c_summary[0].predicted_pro_rt,
        'predicted_anti_prob' : c_summary[0].predicted_anti_prob,
        'predicted_anti_rt' : c_summary[0].predicted_anti_rt,
        }

    return p_summary

# ===========================================================================

def p_seria_model_n_states_gamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_seria_model_n_states(reparametrize_seria_gamma,
        seria_llh_abstract, t, a, u, theta)


def p_reparametrize_seria_gamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    ntheta = wrapper_reparametrize_seria(theta, reparametrize_seria_gamma)

    return ntheta

def p_seria_early_llh_n_states_gamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_seria_model_n_states(reparametrize_seria_gamma,
        seria_early_llh_abstract, t, a, u, theta)

def p_seria_late_llh_n_states_gamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_seria_model_n_states(reparametrize_seria_gamma,
        seria_late_llh_abstract, t, a, u, theta)

# ===========================================================================

def p_seria_model_n_states_invgamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_seria_model_n_states(reparametrize_seria_invgamma,
        seria_llh_abstract, t, a, u, theta)


def p_reparametrize_seria_invgamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    ntheta = wrapper_reparametrize_seria(theta, reparametrize_seria_invgamma)

    return ntheta

def p_seria_early_llh_n_states_invgamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_seria_model_n_states(reparametrize_seria_invgamma,
        seria_early_llh_abstract, t, a, u, theta)

def p_seria_late_llh_n_states_invgamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_seria_model_n_states(reparametrize_seria_invgamma,
        seria_late_llh_abstract, t, a, u, theta)
    
# ===========================================================================

def p_seria_model_n_states_mixedgamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_seria_model_n_states(reparametrize_seria_mixedgamma,
        seria_llh_abstract, t, a, u, theta)


def p_reparametrize_seria_mixedgamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    ntheta = wrapper_reparametrize_seria(theta, reparametrize_seria_mixedgamma)

    return ntheta

def p_seria_early_llh_n_states_mixedgamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_seria_model_n_states(reparametrize_seria_gamma,
        seria_early_llh_abstract, t, a, u, theta)

def p_seria_late_llh_n_states_mixedgamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_seria_model_n_states(reparametrize_seria_mixedgamma,
        seria_late_llh_abstract, t, a, u, theta)

# ===========================================================================

def p_seria_model_n_states_lognorm(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_seria_model_n_states(reparametrize_seria_lognorm,
        seria_llh_abstract, t, a, u, theta)


def p_reparametrize_seria_lognorm(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    ntheta = wrapper_reparametrize_seria(theta, reparametrize_seria_lognorm)

    return ntheta

def p_seria_early_llh_n_states_lognorm(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_seria_model_n_states(reparametrize_seria_lognorm,
        seria_early_llh_abstract, t, a, u, theta)

def p_seria_late_llh_n_states_lognorm(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_seria_model_n_states(reparametrize_seria_lognorm,
        seria_late_llh_abstract, t, a, u, theta)

# ===========================================================================

def p_seria_model_n_states_later(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_seria_model_n_states(reparametrize_seria_later,
        seria_llh_abstract, t, a, u, theta)


def p_reparametrize_seria_later(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    ntheta = wrapper_reparametrize_seria(theta, reparametrize_seria_later)

    return ntheta

def p_seria_early_llh_n_states_later(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_seria_model_n_states(reparametrize_seria_later,
        seria_early_llh_abstract, t, a, u, theta)

def p_seria_early_llh_n_states_later(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_seria_model_n_states(reparametrize_seria_later,
        seria_early_llh_abstract, t, a, u, theta)

def p_seria_late_llh_n_states_later(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_seria_model_n_states(reparametrize_seria_later,
        seria_late_llh_abstract, t, a, u, theta)

# ===========================================================================

def p_seria_model_n_states_wald(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_seria_model_n_states(reparametrize_seria_wald,
        seria_llh_abstract, t, a, u, theta)


def p_reparametrize_seria_wald(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    ntheta = wrapper_reparametrize_seria(theta, reparametrize_seria_wald)

    return ntheta

def p_seria_early_llh_n_states_wald(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_seria_model_n_states(reparametrize_seria_wald,
        seria_early_llh_abstract, t, a, u, theta)

def p_seria_late_llh_n_states_wald(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_seria_model_n_states(reparametrize_seria_wald,
        seria_late_llh_abstract, t, a, u, theta)

# ===========================================================================

def p_prosa_model_n_states_gamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_prosa_model_n_states(reparametrize_prosa_gamma,
        prosa_llh_abstract, t, a, u, theta)


def p_reparametrize_prosa_gamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    ntheta = wrapper_reparametrize_prosa(theta, reparametrize_prosa_gamma)

    return ntheta

def p_prosa_model_n_states_invgamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_prosa_model_n_states(reparametrize_prosa_invgamma,
        prosa_llh_abstract, t, a, u, theta)


def p_reparametrize_prosa_invgamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    ntheta = wrapper_reparametrize_prosa(theta, reparametrize_prosa_invgamma)

    return ntheta

def p_prosa_model_n_states_mixedgamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_prosa_model_n_states(reparametrize_prosa_mixedgamma,
        prosa_llh_abstract, t, a, u, theta)


def p_reparametrize_prosa_mixedgamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    ntheta = wrapper_reparametrize_prosa(theta, reparametrize_prosa_mixedgamma)

    return ntheta

def p_prosa_model_n_states_lognorm(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_prosa_model_n_states(reparametrize_prosa_lognorm,
        prosa_llh_abstract, t, a, u, theta)


def p_reparametrize_prosa_lognorm(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    ntheta = wrapper_reparametrize_prosa(theta, reparametrize_prosa_lognorm)

    return ntheta

def p_prosa_model_n_states_later(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_prosa_model_n_states(reparametrize_prosa_later,
        prosa_llh_abstract, t, a, u, theta)


def p_reparametrize_prosa_later(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    ntheta = wrapper_reparametrize_prosa(theta, reparametrize_prosa_later)

    return ntheta

def p_prosa_model_n_states_wald(
        np.ndarray[np.float64_t, ndim=1, mode="c"] t,
        np.ndarray[np.float64_t, ndim=1, mode="c"] a,
        np.ndarray[np.float64_t, ndim=1, mode="c"] u,
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_prosa_model_n_states(reparametrize_prosa_wald,
        prosa_llh_abstract, t, a, u, theta)


def p_reparametrize_prosa_wald(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):

    ntheta = wrapper_reparametrize_prosa(theta, reparametrize_prosa_wald)

    return ntheta


# ============================================================================
# Summaries
# ============================================================================

# Seria

def p_summary_seria_gamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_summary_seria(reparametrize_seria_gamma, theta)

def p_summary_seria_invgamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_summary_seria(reparametrize_seria_invgamma, theta)

def p_summary_seria_lognorm(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_summary_seria(reparametrize_seria_lognorm, theta)

def p_summary_seria_mixedgamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_summary_seria(reparametrize_seria_mixedgamma, theta)

def p_summary_seria_later(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_summary_seria(reparametrize_seria_later, theta)

def p_summary_seria_wald(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_summary_seria(reparametrize_seria_wald, theta)

# Prosa

def p_summary_prosa_gamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_summary_prosa(reparametrize_prosa_gamma, theta)

def p_summary_prosa_invgamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_summary_prosa(reparametrize_prosa_invgamma, theta)

def p_summary_prosa_lognorm(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_summary_prosa(reparametrize_prosa_lognorm, theta)

def p_summary_prosa_mixedgamma(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_summary_prosa(reparametrize_prosa_mixedgamma, theta)

def p_summary_prosa_later(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_summary_prosa(reparametrize_prosa_later, theta)

def p_summary_prosa_wald(
        np.ndarray[np.float64_t, ndim=1, mode="c"] theta):
 
    return wrapper_summary_prosa(reparametrize_prosa_wald, theta)

