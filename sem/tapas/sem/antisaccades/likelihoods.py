#! /usr/bin/env python

# aponteeduardo@gmail.com
# copyright (C) 2016


''''

Likelihoods of the PROSA and SERIA models.

'''


import numpy as np

from tapas.sem.antisaccades.wrappers import *


def wrapper(llh, rt, ac, tt, theta):

    rt = np.array(rt, dtype=np.float64)
    rt = rt.reshape(rt.size)
    ac = np.array(ac, dtype=np.float64)
    ac = ac.reshape(ac.size)
    tt = np.array(tt, dtype=np.float64)
    tt = tt.reshape(ac.size)
    theta = np.array(theta, dtype=np.float64)
    theta = theta.reshape(theta.size)
    vals = llh(rt, ac, tt, theta)

    return vals


def seria_n_states_gamma(rt, ac, tt, theta):
    ''' Likelihood of the seria model using the n_states distribution

    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.


    '''

    return wrapper(p_seria_model_n_states_gamma, rt, ac, tt, theta)


def seria_n_states_invgamma(rt, ac, tt, theta):
    ''' Likelihood of the seria model using the n_states distribution

    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.


    '''

    return wrapper(p_seria_model_n_states_invgamma, rt, ac, tt, theta)


def seria_n_states_mixedgamma(rt, ac, tt, theta):
    ''' Likelihood of the seria model using the n_states distribution

    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.


    '''

    return wrapper(p_seria_model_n_states_mixedgamma, rt, ac, tt, theta)


def seria_n_states_lognorm(rt, ac, tt, theta):
    ''' Likelihood of the seria model using the n_states distribution

    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.


    '''

    return wrapper(p_seria_model_n_states_lognorm, rt, ac, tt, theta)


def seria_n_states_later(rt, ac, tt, theta):
    ''' Likelihood of the seria model using the n_states distribution

    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.


    '''

    return wrapper(p_seria_model_n_states_later, rt, ac, tt, theta)


def seria_n_states_wald(rt, ac, tt, theta):
    ''' Likelihood of the seria model using the n_states distribution

    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.


    '''

    return wrapper(p_seria_model_n_states_wald, rt, ac, tt, theta)


def prosa_n_states_gamma(rt, ac, tt, theta):
    ''' Likelihood of the prosa model using the n_states distribution

    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.


    '''

    return wrapper(p_prosa_model_n_states_gamma, rt, ac, tt, theta)


def prosa_n_states_invgamma(rt, ac, tt, theta):
    ''' Likelihood of the prosa model using the n_states distribution

    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.


    '''

    return wrapper(p_prosa_model_n_states_invgamma, rt, ac, tt, theta)


def prosa_n_states_mixedgamma(rt, ac, tt, theta):
    ''' Likelihood of the prosa model using the n_states distribution

    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.


    '''

    return wrapper(p_prosa_model_n_states_mixedgamma, rt, ac, tt, theta)


def prosa_n_states_lognorm(rt, ac, tt, theta):
    ''' Likelihood of the prosa model using the n_states distribution

    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.


    '''

    return wrapper(p_prosa_model_n_states_lognorm, rt, ac, tt, theta)


def prosa_n_states_later(rt, ac, tt, theta):
    ''' Likelihood of the prosa model using the n_states distribution

    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.


    '''

    return wrapper(p_prosa_model_n_states_later, rt, ac, tt, theta)


def prosa_n_states_wald(rt, ac, tt, theta):
    ''' Likelihood of the prosa model using the n_states distribution

    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.


    '''

    return wrapper(p_prosa_model_n_states_wald, rt, ac, tt, theta)


def seria_early_llh_n_states_gamma(rt, ac, tt, theta):
    ''' Likelihood of an early response in the seria model using the n_states
        distribution.

    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.


    '''

    return wrapper(p_seria_early_llh_n_states_gamma, rt, ac, tt, theta)


def seria_early_llh_n_states_invgamma(rt, ac, tt, theta):
    ''' Likelihood of an early response in the seria model using the n_states
        distribution.

    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.


    '''

    return wrapper(p_seria_early_llh_n_states_invgamma, rt, ac, tt, theta)


def seria_early_llh_n_states_mixedgamma(rt, ac, tt, theta):
    ''' Likelihood of an early response in the seria model using the n_states
        distribution.

    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.


    '''

    return wrapper(p_seria_early_llh_n_states_mixedgamma, rt, ac, tt, theta)


def seria_early_llh_n_states_lognorm(rt, ac, tt, theta):
    ''' Likelihood of an early response in the seria model using the n_states
        distribution.

    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.


    '''

    return wrapper(p_seria_early_llh_n_states_lognorm, rt, ac, tt, theta)


def seria_early_llh_n_states_later(rt, ac, tt, theta):
    ''' Likelihood of an early response in the seria model using the n_states
        distribution.

    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.


    '''

    return wrapper(p_seria_early_llh_n_states_later, rt, ac, tt, theta)


def seria_early_llh_n_states_wald(rt, ac, tt, theta):
    ''' Likelihood of an early response in the seria model using the n_states
        distribution.

    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.


    '''

    return wrapper(p_seria_early_llh_n_states_wald, rt, ac, tt, theta)


def seria_late_llh_n_states_gamma(rt, ac, tt, theta):
    ''' Likelihood of an late response in the seria model using the n_states
        distribution.

    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.


    '''

    return wrapper(p_seria_late_llh_n_states_gamma, rt, ac, tt, theta)


def seria_late_llh_n_states_invgamma(rt, ac, tt, theta):
    ''' Likelihood of an late response in the seria model using the n_states
        distribution.

    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.


    '''

    return wrapper(p_seria_late_llh_n_states_invgamma, rt, ac, tt, theta)


def seria_late_llh_n_states_mixedgamma(rt, ac, tt, theta):
    ''' Likelihood of an late response in the seria model using the n_states
        distribution.

    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.


    '''

    return wrapper(p_seria_late_llh_n_states_mixedgamma, rt, ac, tt, theta)


def seria_late_llh_n_states_lognorm(rt, ac, tt, theta):
    ''' Likelihood of an late response in the seria model using the n_states
        distribution.

    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.


    '''

    return wrapper(p_seria_late_llh_n_states_lognorm, rt, ac, tt, theta)


def seria_late_llh_n_states_later(rt, ac, tt, theta):
    ''' Likelihood of an late response in the seria model using the n_states
        distribution.

    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.


    '''

    return wrapper(p_seria_late_llh_n_states_later, rt, ac, tt, theta)


def seria_late_llh_n_states_wald(rt, ac, tt, theta):
    ''' Likelihood of an late response in the seria model using the n_states
        distribution.

    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.


    '''

    return wrapper(p_seria_late_llh_n_states_wald, rt, ac, tt, theta)


if __name__ == '__main__':
    pass
