#! /usr/bin/env python

# aponteeduardo@gmail.com
# copyright (C) 2017


''''

Contains functions related to the parameters.

'''


from . import containers
from . import reparametrize as reparam
from . import likelihoods


class Parameters(containers.AlgebraicObject):
    ''' Abstract class. '''
    
    @classmethod
    def make_dict(clc, samples):
        '''Return a dictionary from a vector of samples.'''

        sdict = dict(list(zip(clc.fields, samples)))

        return sdict


class ParametersProsa(Parameters):
    '''Parameters of the seria model. '''

    fields = ['kp', 'tp', 'ka', 'ta', 'ks', 'ts', 't0', 'da', 'p0']

    pass


class ParametersSeria(Parameters):
    '''Parameters of the seria model. '''

    fields = [
            'kp', 'tp', 'ka', 'ta', 'ks', 'ts', 'kl', 'tl', 't0',
            'da', 'p0']

    pass


class ParametersSeriaGamma(ParametersSeria):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_seria_gamma(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.seria_n_states_gamma(t, a, tt, theta)

    @staticmethod
    def l_early(t, a, tt, theta):

        return likelihoods.seria_early_llh_n_states_gamma(t, a, tt, theta)

    @staticmethod
    def l_late(t, a, tt, theta):

        return likelihoods.seria_late_llh_n_states_gamma(t, a, tt, theta)


class ParametersSeriaInvgamma(ParametersSeria):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_seria_invgamma(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.seria_n_states_invgamma(t, a, tt, theta)

    @staticmethod
    def l_early(t, a, tt, theta):

        return likelihoods.seria_early_llh_n_states_invgamma(t, a, tt, theta)

    @staticmethod
    def l_late(t, a, tt, theta):

        return likelihoods.seria_late_llh_n_states_invgamma(t, a, tt, theta)


class ParametersSeriaMixedgamma(ParametersSeria):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_seria_mixedgamma(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.seria_n_states_mixedgamma(t, a, tt, theta)

    @staticmethod
    def l_early(t, a, tt, theta):

        return likelihoods.seria_early_llh_n_states_mixedgamma(t, a, tt, theta)

    @staticmethod
    def l_late(t, a, tt, theta):

        return likelihoods.seria_late_llh_n_states_mixedgamma(t, a, tt, theta)


class ParametersSeriaLognorm(ParametersSeria):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_seria_lognorm(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.seria_n_states_lognorm(t, a, tt, theta)

    @staticmethod
    def l_early(t, a, tt, theta):

        return likelihoods.seria_early_llh_n_states_lognorm(t, a, tt, theta)

    @staticmethod
    def l_late(t, a, tt, theta):

        return likelihoods.seria_late_llh_n_states_lognorm(t, a, tt, theta)


class ParametersSeriaLater(ParametersSeria):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_seria_later(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.seria_n_states_later(t, a, tt, theta)

    @staticmethod
    def l_early(t, a, tt, theta):

        return likelihoods.seria_early_llh_n_states_later(t, a, tt, theta)

    @staticmethod
    def l_late(t, a, tt, theta):

        return likelihoods.seria_late_llh_n_states_later(t, a, tt, theta)


class ParametersSeriaWald(ParametersSeria):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_seria_wald(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.seria_n_states_wald(t, a, tt, theta)

    @staticmethod
    def l_early(t, a, tt, theta):

        return likelihoods.seria_early_llh_n_states_wald(t, a, tt, theta)

    @staticmethod
    def l_late(t, a, tt, theta):

        return likelihoods.seria_late_llh_n_states_wald(t, a, tt, theta)


class ParametersProsaGamma(ParametersProsa):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_prosa_gamma(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.prosa_n_states_gamma(t, a, tt, theta)


class ParametersProsaInvgamma(ParametersProsa):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_prosa_invgamma(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.prosa_n_states_invgamma(t, a, tt, theta)


class ParametersProsaMixedgamma(ParametersProsa):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_prosa_mixedgamma(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.prosa_n_states_mixedgamma(t, a, tt, theta)


class ParametersProsaLognorm(ParametersProsa):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_prosa_lognorm(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.prosa_n_states_lognorm(t, a, tt, theta)


class ParametersProsaLater(ParametersProsa):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_prosa_later(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.prosa_n_states_later(t, a, tt, theta)


class ParametersProsaWald(ParametersProsa):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_prosa_wald(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.prosa_n_states_wald(t, a, tt, theta)


if __name__ == '__main__':
    pass
