#! /usr/bin/env python

# aponteeduardo@gmail.com
# copyright (C) 2017


''''

Different utilities for the model

'''

import numpy as np

def transform_mv_to_gamma_k(mu, sigma2):
    '''Transform mean and variange to the k parameter of the gamma dist. '''

    return mu * mu /sigma2

def transform_mv_to_gamma_t(mu, sigma2):
    '''Transform mean and variance to the t parameter of the gamma dist. '''

    return mu /sigma2

def reparametrize_seri_invgamma(theta):
    ''' Reparametrize the parameters to the rt space. '''

    stheta = {}

    stheta['kp'] = transform_mv_to_gamma_k(np.exp(theta[0, :]), 
            np.exp(theta[1, :])) + 2
    stheta['tp'] = transform_mv_to_gamma_t(np.exp(theta[0, :]), 
            np.exp(theta[1, :])) 

    stheta['ka'] = transform_mv_to_gamma_k(np.exp(theta[2, :]), 
            np.exp(theta[3, :])) + 2
    stheta['ta'] = transform_mv_to_gamma_t(np.exp(theta[2, :]), 
            np.exp(theta[3, :])) 

    stheta['ks'] = transform_mv_to_gamma_k(np.exp(theta[4, :]), 
            np.exp(theta[5, :])) + 2
    stheta['ts'] = transform_mv_to_gamma_t(np.exp(theta[4, :]), 
            np.exp(theta[5, :]))

    stheta['pp'] = np.arctan(theta[6, :])/np.pi + 0.5
    stheta['ap'] = np.arctan(theta[7, :])/np.pi + 0.5

    stheta['t0'] = np.exp(theta[8, :])
    stheta['da'] = np.exp(theta[9, :])

    stheta['p0'] = np.arctan(theta[10, :])/np.pi + 0.5

    return stheta

def reparametrize_dora_invgamma(theta):
    ''' Reparametrize the parameters to the rt space. '''

    stheta = {}

    stheta['kp'] = transform_mv_to_gamma_k(np.exp(theta[0, :]), 
            np.exp(theta[1, :])) + 2
    stheta['tp'] = transform_mv_to_gamma_t(np.exp(theta[0, :]), 
            np.exp(theta[1, :])) 

    stheta['ka'] = transform_mv_to_gamma_k(np.exp(theta[2, :]), 
            np.exp(theta[3, :])) + 2
    stheta['ta'] = transform_mv_to_gamma_t(np.exp(theta[2, :]), 
            np.exp(theta[3, :])) 

    stheta['ks'] = transform_mv_to_gamma_k(np.exp(theta[4, :]), 
            np.exp(theta[5, :])) + 2
    stheta['ts'] = transform_mv_to_gamma_t(np.exp(theta[4, :]), 
            np.exp(theta[5, :]))

    stheta['kl'] = transform_mv_to_gamma_k(np.exp(theta[6, :]), 
            np.exp(theta[7, :])) + 2
    stheta['tl'] = transform_mv_to_gamma_t(np.exp(theta[6, :]), 
            np.exp(theta[7, :]))

    stheta['t0'] = np.exp(theta[8, :])
    stheta['da'] = np.exp(theta[9, :])

    stheta['p0'] = np.arctan(theta[10, :])/np.pi + 0.5

    return stheta





if __name__ == '__main__':

    pass    

