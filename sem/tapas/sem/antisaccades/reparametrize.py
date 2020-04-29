#! /usr/bin/env python

# aponteeduardo@gmail.com
# copyright (C) 2016


''''

Likelihoods of the SOONER and SERI models.

'''

import numpy as np
from . import wrappers as c_wrappers


def wrapper(theta, reparam):

    theta = np.array(theta, dtype=np.float64)
    theta = theta.reshape(theta.size)
    vals = reparam(theta)

    return vals


def reparametrize_seria_gamma(theta):
    ''' Reparametrize the seri model using the invgamma distribution

    theta    Parameters.


    '''

    return wrapper(theta, c_wrappers.p_reparametrize_seria_gamma)


def reparametrize_seria_invgamma(theta):
    ''' Reparametrize the seri model using the invgamma distribution

    theta    Parameters.


    '''

    return wrapper(theta, c_wrappers.p_reparametrize_seria_invgamma)


def reparametrize_seria_mixedgamma(theta):
    ''' Reparametrize the seri model using the invgamma distribution

    theta    Parameters.


    '''

    return wrapper(theta, c_wrappers.p_reparametrize_seria_mixedgamma)


def reparametrize_seria_lognorm(theta):
    ''' Reparametrize the seri model using the invgamma distribution

    theta    Parameters.


    '''

    return wrapper(theta, c_wrappers.p_reparametrize_seria_lognorm)


def reparametrize_seria_later(theta):
    ''' Reparametrize the seri model using the invgamma distribution

    theta    Parameters.


    '''

    return wrapper(theta, c_wrappers.p_reparametrize_seria_later)


def reparametrize_seria_wald(theta):
    ''' Reparametrize the seri model using the invgamma distribution

    theta    Parameters.


    '''

    return wrapper(theta, c_wrappers.p_reparametrize_seria_wald)


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


if __name__ == '__main__':
    pass
