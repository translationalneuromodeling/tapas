#! /usr/bin/env python

# aponteeduardo@gmail.com
# copyright (C) 2016


''''

Likelihoods of the SOONER and SERI models.

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


def seri_two_states_gamma(rt, ac, tt, theta):
    ''' Likelihood of the seri model using the two_states distribution
    
    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.

        
    '''

    return wrapper(p_seri_model_two_states_gamma, rt, ac, tt, theta)

def seri_two_states_invgamma(rt, ac, tt, theta):
    ''' Likelihood of the seri model using the two_states distribution
    
    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.

        
    '''

    return wrapper(p_seri_model_two_states_invgamma, rt, ac, tt, theta)

def seri_two_states_mixedgamma(rt, ac, tt, theta):
    ''' Likelihood of the seri model using the two_states distribution
    
    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.

        
    '''

    return wrapper(p_seri_model_two_states_mixedgamma, rt, ac, tt, theta)

def seri_two_states_lognorm(rt, ac, tt, theta):
    ''' Likelihood of the seri model using the two_states distribution
    
    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.

        
    '''

    return wrapper(p_seri_model_two_states_lognorm, rt, ac, tt, theta)

def seri_two_states_later(rt, ac, tt, theta):
    ''' Likelihood of the seri model using the two_states distribution
    
    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.

        
    '''

    return wrapper(p_seri_model_two_states_later, rt, ac, tt, theta)

def seri_two_states_wald(rt, ac, tt, theta):
    ''' Likelihood of the seri model using the two_states distribution
    
    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.

        
    '''

    return wrapper(p_seri_model_two_states_wald, rt, ac, tt, theta)

def dora_two_states_gamma(rt, ac, tt, theta):
    ''' Likelihood of the dora model using the two_states distribution
    
    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.

        
    '''

    return wrapper(p_dora_model_two_states_gamma, rt, ac, tt, theta)

def dora_two_states_invgamma(rt, ac, tt, theta):
    ''' Likelihood of the dora model using the two_states distribution
    
    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.

        
    '''

    return wrapper(p_dora_model_two_states_invgamma, rt, ac, tt, theta)

def dora_two_states_mixedgamma(rt, ac, tt, theta):
    ''' Likelihood of the dora model using the two_states distribution
    
    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.

        
    '''

    return wrapper(p_dora_model_two_states_mixedgamma, rt, ac, tt, theta)

def dora_two_states_lognorm(rt, ac, tt, theta):
    ''' Likelihood of the dora model using the two_states distribution
    
    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.

        
    '''

    return wrapper(p_dora_model_two_states_lognorm, rt, ac, tt, theta)

def dora_two_states_later(rt, ac, tt, theta):
    ''' Likelihood of the dora model using the two_states distribution
    
    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.

        
    '''

    return wrapper(p_dora_model_two_states_later, rt, ac, tt, theta)

def dora_two_states_wald(rt, ac, tt, theta):
    ''' Likelihood of the dora model using the two_states distribution
    
    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.

        
    '''

    return wrapper(p_dora_model_two_states_wald, rt, ac, tt, theta)

def prosa_two_states_gamma(rt, ac, tt, theta):
    ''' Likelihood of the prosa model using the two_states distribution
    
    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.

        
    '''

    return wrapper(p_prosa_model_two_states_gamma, rt, ac, tt, theta)

def prosa_two_states_invgamma(rt, ac, tt, theta):
    ''' Likelihood of the prosa model using the two_states distribution
    
    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.

        
    '''

    return wrapper(p_prosa_model_two_states_invgamma, rt, ac, tt, theta)

def prosa_two_states_mixedgamma(rt, ac, tt, theta):
    ''' Likelihood of the prosa model using the two_states distribution
    
    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.

        
    '''

    return wrapper(p_prosa_model_two_states_mixedgamma, rt, ac, tt, theta)

def prosa_two_states_lognorm(rt, ac, tt, theta):
    ''' Likelihood of the prosa model using the two_states distribution
    
    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.

        
    '''

    return wrapper(p_prosa_model_two_states_lognorm, rt, ac, tt, theta)

def prosa_two_states_later(rt, ac, tt, theta):
    ''' Likelihood of the prosa model using the two_states distribution
    
    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.

        
    '''

    return wrapper(p_prosa_model_two_states_later, rt, ac, tt, theta)

def prosa_two_states_wald(rt, ac, tt, theta):
    ''' Likelihood of the prosa model using the two_states distribution
    
    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.

        
    '''

    return wrapper(p_prosa_model_two_states_wald, rt, ac, tt, theta)

