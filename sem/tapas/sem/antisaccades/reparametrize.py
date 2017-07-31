#! /usr/bin/env python

# aponteeduardo@gmail.com
# copyright (C) 2016


''''

Likelihoods of the SOONER and SERI models.

'''

from pdb import set_trace as _

import numpy as np
import wrappers as c_wrappers


def wrapper(theta, reparam):
    
    theta = np.array(theta, dtype=np.float64)
    theta = theta.reshape(theta.size)
    vals = reparam(theta)

    return vals



def reparametrize_seri_gamma(theta):
    ''' Reparametrize the seri model using the invgamma distribution
    
    theta    Parameters.

        
    '''

    return wrapper(theta, c_wrappers.p_reparametrize_seri_gamma)


def reparametrize_seri_invgamma(theta):
    ''' Reparametrize the seri model using the invgamma distribution
    
    theta    Parameters.

        
    '''

    return wrapper(theta, c_wrappers.p_reparametrize_seri_invgamma)


def reparametrize_seri_mixedgamma(theta):
    ''' Reparametrize the seri model using the invgamma distribution
    
    theta    Parameters.

        
    '''

    return wrapper(theta, c_wrappers.p_reparametrize_seri_mixedgamma)


def reparametrize_seri_lognorm(theta):
    ''' Reparametrize the seri model using the invgamma distribution
    
    theta    Parameters.

        
    '''

    return wrapper(theta, c_wrappers.p_reparametrize_seri_lognorm)


def reparametrize_seri_later(theta):
    ''' Reparametrize the seri model using the invgamma distribution
    
    theta    Parameters.

        
    '''

    return wrapper(theta, c_wrappers.p_reparametrize_seri_later)


def reparametrize_seri_wald(theta):
    ''' Reparametrize the seri model using the invgamma distribution
    
    theta    Parameters.

        
    '''

    return wrapper(theta, c_wrappers.p_reparametrize_seri_wald)


def reparametrize_dora_gamma(theta):
    ''' Reparametrize the seri model using the invgamma distribution
    
    theta    Parameters.

        
    '''

    return wrapper(theta, c_wrappers.p_reparametrize_dora_gamma)


def reparametrize_dora_invgamma(theta):
    ''' Reparametrize the seri model using the invgamma distribution
    
    theta    Parameters.

        
    '''

    return wrapper(theta, c_wrappers.p_reparametrize_dora_invgamma)


def reparametrize_dora_mixedgamma(theta):
    ''' Reparametrize the seri model using the invgamma distribution
    
    theta    Parameters.

        
    '''

    return wrapper(theta, c_wrappers.p_reparametrize_dora_mixedgamma)


def reparametrize_dora_lognorm(theta):
    ''' Reparametrize the seri model using the invgamma distribution
    
    theta    Parameters.

        
    '''

    return wrapper(theta, c_wrappers.p_reparametrize_dora_lognorm)


def reparametrize_dora_later(theta):
    ''' Reparametrize the seri model using the invgamma distribution
    
    theta    Parameters.

        
    '''

    return wrapper(theta, c_wrappers.p_reparametrize_dora_later)


def reparametrize_dora_wald(theta):
    ''' Reparametrize the seri model using the invgamma distribution
    
    theta    Parameters.

        
    '''

    return wrapper(theta, c_wrappers.p_reparametrize_dora_wald)


def reparametrize_prosa_gamma(theta):
    ''' Reparametrize the seri model using the invgamma distribution
    
    theta    Parameters.

        
    '''

    return wrapper(theta, c_wrappers.p_reparametrize_prosa_gamma)


def reparametrize_prosa_invgamma(theta):
    ''' Reparametrize the seri model using the invgamma distribution
    
    theta    Parameters.

        
    '''

    return wrapper(theta, c_wrappers.p_reparametrize_prosa_invgamma)


def reparametrize_prosa_mixedgamma(theta):
    ''' Reparametrize the seri model using the invgamma distribution
    
    theta    Parameters.

        
    '''

    return wrapper(theta, c_wrappers.p_reparametrize_prosa_mixedgamma)


def reparametrize_prosa_lognorm(theta):
    ''' Reparametrize the seri model using the invgamma distribution
    
    theta    Parameters.

        
    '''

    return wrapper(theta, c_wrappers.p_reparametrize_prosa_lognorm)


def reparametrize_prosa_later(theta):
    ''' Reparametrize the seri model using the invgamma distribution
    
    theta    Parameters.

        
    '''

    return wrapper(theta, c_wrappers.p_reparametrize_prosa_later)


def reparametrize_prosa_wald(theta):
    ''' Reparametrize the seri model using the invgamma distribution
    
    theta    Parameters.

        
    '''

    return wrapper(theta, c_wrappers.p_reparametrize_prosa_wald)

