#! /usr/bin/env python

# aponteeduardo@gmail.com
# copyright (C) 2017


''''

Contains functions related to the parameters.

'''


import containers
import reparametrize as reparam
import likelihoods


class Parameters(containers.AlgebraicObject):
    ''' Abstract class. '''
    pass


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


class ParametersSeriaGamma(ParametersDora):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_seria_gamma(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.seria_n_states_gamma(t, a, tt, theta)

    @staticmethod
    def ler(t, a, tt, theta):

        return likelihoods.seria_early_llh_n_states_gamma(t, a, tt, theta)


class ParametersSeriaInvgamma(ParametersDora):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_seria_invgamma(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.seria_n_states_invgamma(t, a, tt, theta)

    @staticmethod
    def ler(t, a, tt, theta):

        return likelihoods.seria_early_llh_n_states_invgamma(t, a, tt, theta)


class ParametersSeriaMixedgamma(ParametersDora):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_seria_mixedgamma(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.seria_n_states_mixedgamma(t, a, tt, theta)

    @staticmethod
    def ler(t, a, tt, theta):

        return likelihoods.seria_early_llh_n_states_mixedgamma(t, a, tt, theta)

class ParametersSeriaLognorm(ParametersDora):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_seria_lognorm(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.seria_n_states_lognorm(t, a, tt, theta)

    @staticmethod
    def ler(t, aa, tt, theta):

        return likelihoods.seria_early_llh_n_states_lognorm(t, aa, tt, theta)


class ParametersSeriaLater(ParametersDora):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_seria_later(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.seria_n_states_later(t, a, tt, theta)

    @staticmethod
    def ler(t, aa, tt, theta):

        return likelihoods.seria_early_llh_n_states_later(t, aa, tt, theta)


class ParametersSeriaWald(ParametersDora):

    @staticmethod
    def reparametrize(samples):

        return reparam.reparametrize_seria_wald(samples)

    @staticmethod
    def lpdf(t, a, tt, theta):

        return likelihoods.seria_n_states_wald(t, a, tt, theta)

    @staticmethod
    def ler(t, aa, tt, theta):

        return likelihoods.seria_early_llh_n_states_wald(t, aa, tt, theta)


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
