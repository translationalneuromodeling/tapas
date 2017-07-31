#! /usr/bin/env python

# aponteeduardo@gmail.com
# copyright (C) 2017


''''

Contains functions related to the parameters.

'''


from pdb import set_trace as _
import numpy as np

import containers
import reparametrize as reparam
import likelihoods 

class Parameters(containers.AlgebraicObject):
    ''' Abstract class. '''
    pass

class ParametersProsa(Parameters):
    '''Parameters of the dora model. '''

    fields = ['kp', 'tp', 'ka', 'ta', 'ks', 'ts', 't0', 'da', 'p0']

    pass

class ParametersDora(Parameters):
    '''Parameters of the dora model. '''

    fields = ['kp', 'tp', 'ka', 'ta', 'ks', 'ts', 'kl', 'tl', 't0', 
        'da', 'p0']

    pass

class ParametersSeri(Parameters):
    '''Parameters of the dora model. '''

    fields = ['kp', 'tp', 'ka', 'ta', 'ks', 'ts', 'pp', 'ap', 't0', 
        'da', 'p0']

    pass

class ParametersSeriGamma(ParametersSeri):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_seri_gamma(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.seri_two_states_gamma(t, a, tt, theta)

    

class ParametersSeriInvgamma(ParametersSeri):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_seri_invgamma(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.seri_two_states_invgamma(t, a, tt, theta)

    
class ParametersSeriMixedgamma(ParametersSeri):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_seri_mixedgamma(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.seri_two_states_mixedgamma(t, a, tt, theta)

    

class ParametersSeriLognorm(ParametersSeri):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_seri_lognorm(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.seri_two_states_lognorm(t, a, tt, theta)

    

class ParametersSeriLater(ParametersSeri):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_seri_later(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.seri_two_states_later(t, a, tt, theta)

    

class ParametersSeriWald(ParametersSeri):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_seri_wald(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.seri_two_states_wald(t, a, tt, theta)

    

class ParametersDoraGamma(ParametersDora):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_dora_gamma(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.dora_two_states_gamma(t, a, tt, theta)

    

class ParametersDoraInvgamma(ParametersDora):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_dora_invgamma(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.dora_two_states_invgamma(t, a, tt, theta)

    

class ParametersDoraMixedgamma(ParametersDora):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_dora_mixedgamma(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.dora_two_states_mixedgamma(t, a, tt, theta)

    

class ParametersDoraLognorm(ParametersDora):

    @staticmethod
    def reparametrize(samples):
   
        return reparam.reparametrize_dora_lognorm(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.dora_two_states_lognorm(t, a, tt, theta)

    
class ParametersDoraLater(ParametersDora):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_dora_later(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.dora_two_states_later(t, a, tt, theta)

    

class ParametersDoraWald(ParametersDora):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_dora_wald(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.dora_two_states_wald(t, a, tt, theta)

    

class ParametersProsaGamma(ParametersProsa):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_prosa_gamma(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.prosa_two_states_gamma(t, a, tt, theta)

    

class ParametersProsaInvgamma(ParametersProsa):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_prosa_invgamma(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.prosa_two_states_invgamma(t, a, tt, theta)

    

class ParametersProsaMixedgamma(ParametersProsa):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_prosa_mixedgamma(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.prosa_two_states_mixedgamma(t, a, tt, theta)

    

class ParametersProsaLognorm(ParametersProsa):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_prosa_lognorm(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.prosa_two_states_lognorm(t, a, tt, theta)

    

class ParametersProsaLater(ParametersProsa):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_prosa_later(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.prosa_two_states_later(t, a, tt, theta)

    

class ParametersProsaWald(ParametersProsa):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_prosa_wald(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.prosa_two_states_wald(t, a, tt, theta)

if __name__ == '__main__':
    pass    

