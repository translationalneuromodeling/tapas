#! /usr/bin/env python

# aponteeduardo@gmail.com
# copyright (C) 2016


''''

Likelihoods of the SOONER and SERI models.

'''


import numpy as np

from wrapped import *

def fun_factory(llh):
    ''' A factory for the log likelihoods of the ware models. '''

    def decorator(afunc):
        def afunc(rt, ac, tt, theta):
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

        return afunc

    return decorator

@fun_factory(p_seri_model_two_states_gamma)
def seri_two_states_gamma(rt, ac, tt, theta):
    ''' Likelihood of the seri model using the gamma distribution
    
    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.

        
    '''

    return


@fun_factory(p_seri_model_two_states_invgamma)
def seri_two_states_invgamma(rt, ac, tt, theta):
    ''' Likelihood of the seri model using the invgamma distribution
    
    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.

        
    '''

    return


@fun_factory(p_seri_model_two_states_mixedgamma)
def seri_two_states_mixedgamma(rt, ac, tt, theta):
    ''' Likelihood of the seri model using the mixedgamma distribution
    
    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.

        
    '''

    return


@fun_factory(p_seri_model_two_states_later)
def seri_two_states_later(rt, ac, tt, theta):
    ''' Likelihood of the seri model using the later distribution
    
    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.

        
    '''

    return


@fun_factory(p_seri_model_two_states_wald)
def seri_two_states_wald(rt, ac, tt, theta):
    ''' Likelihood of the seri model using the wald distribution
    
    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.

        
    '''

    return


@fun_factory(p_dora_model_two_states_gamma)
def dora_two_states_gamma(rt, ac, tt, theta):
    ''' Likelihood of the dora model using the gamma distribution
    
    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.

        
    '''

    return


@fun_factory(p_dora_model_two_states_invgamma)
def dora_two_states_invgamma(rt, ac, tt, theta):
    ''' Likelihood of the dora model using the invgamma distribution
    
    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.

        
    '''

    return


@fun_factory(p_dora_model_two_states_mixedgamma)
def dora_two_states_mixedgamma(rt, ac, tt, theta):
    ''' Likelihood of the dora model using the mixedgamma distribution
    
    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.

        
    '''

    return


@fun_factory(p_dora_model_two_states_later)
def dora_two_states_later(rt, ac, tt, theta):
    ''' Likelihood of the dora model using the later distribution
    
    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.

        
    '''

    return


@fun_factory(p_dora_model_two_states_wald)
def dora_two_states_wald(rt, ac, tt, theta):
    ''' Likelihood of the dora model using the wald distribution
    
    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.

        
    '''

    return


@fun_factory(p_prosa_model_two_states_gamma)
def prosa_two_states_gamma(rt, ac, tt, theta):
    ''' Likelihood of the prosa model using the gamma distribution
    
    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.

        
    '''

    return


@fun_factory(p_prosa_model_two_states_invgamma)
def prosa_two_states_invgamma(rt, ac, tt, theta):
    ''' Likelihood of the prosa model using the invgamma distribution
    
    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.

        
    '''

    return


@fun_factory(p_prosa_model_two_states_mixedgamma)
def prosa_two_states_mixedgamma(rt, ac, tt, theta):
    ''' Likelihood of the prosa model using the mixedgamma distribution
    
    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.

        
    '''

    return


@fun_factory(p_prosa_model_two_states_later)
def prosa_two_states_later(rt, ac, tt, theta):
    ''' Likelihood of the prosa model using the later distribution
    
    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.

        
    '''

    return


@fun_factory(p_prosa_model_two_states_wald)
def prosa_two_states_wald(rt, ac, tt, theta):
    ''' Likelihood of the prosa model using the wald distribution
    
    rt      Reaction times
    ac      Action
    tt      Trial typ
    pars    Parameters.

        
    '''

    return

