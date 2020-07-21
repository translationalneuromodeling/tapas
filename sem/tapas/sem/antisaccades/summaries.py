#! /usr/bin/env python

# aponteeduardo@gmail.com
# copyright (C) 2016


''''

Graphical tools

'''


import numpy as np
from scipy.integrate import cumtrapz
from scipy import stats

from . import containers
from . import wrappers


def lcosh(x):
    ''' Log of cosine hyperbolic '''

#    if np.abs(x) > 10.0:
#        y = np.abs(x) + np.log1p(np.exp(-2.0 * np.abs(x)))
#    else:
#        y = np.log(np.exp(x) + np.exp(-x))

    return np.log(np.exp(x) + np.exp(-x)) - np.log(2.0)


class SummaryStats(containers.AlgebraicObject):
    ''' Creates the summary of a bmodel. '''

    def __init__(self, samples=None):

        if samples is None:
            super(SummaryStats, self).__init__()
        else:
            try:
                samples = samples.get_raw_values()
                samples = np.array(samples)
            except AttributeError:
                # Try to get the raw values. If it does not work, let it
                # break afterwards
                pass

            results = self.summary_statistics(samples)
            super(SummaryStats, self).__init__(results)

        return

    def set_values_from_iter(self, values):
        '''Set the values from a interable. '''

        for i, val in enumerate(values):

            self.__dict__[self.__class__.fields[i]] = val

        return

    def summary_statistics(self, samples):
        ''' Compute the summary statistics. '''

        raise NotImplementedError

        return None


class SummarySeria(SummaryStats):
    ''' Creates the summary of  the dora model. '''

    fields = [
        'late_pro_rt',
        'anti_rt',
        'inhib_fail_rt',
        'inhib_fail_prob',
        'late_pro_prob',
        'predicted_pro_prob',
        'predicted_pro_rt',
        'predicted_anti_prob',
        'predicted_anti_rt',
            ]

    def __init__(self, *args, **kargs):

        super(SummarySeria, self).__init__(*args, **kargs)

        return None


class SummarySeriaGamma(SummarySeria):
    '''Class for the summaries '''

    def summary_statistics(self, samples):

        results = wrappers.p_summary_seria_gamma(samples)

        return results


class SummarySeriaInvgamma(SummarySeria):
    '''Class for the summaries '''

    def summary_statistics(self, samples):

        results = wrappers.p_summary_seria_invgamma(samples)

        return results


class SummarySeriaMixedgamma(SummarySeria):
    '''Class for the summaries '''

    def summary_statistics(self, samples):

        results = wrappers.p_summary_seria_mixedgamma(samples)

        return results


class SummarySeriaLognorm(SummarySeria):
    '''Class for the summaries '''

    def summary_statistics(self, samples):

        results = wrappers.p_summary_seria_lognorm(samples)

        return results


class SummarySeriaWald(SummarySeria):
    '''Class for the summaries '''

    def summary_statistics(self, samples):

        results = wrappers.p_summary_seria_gamma(samples)

        return results


class SummarySeriaLater(SummarySeria):
    '''Class for the summaries '''

    def summary_statistics(self, samples):

        results = wrappers.p_summary_seria_wald(samples)

        return results


class SummaryProsa(SummaryStats):
    ''' Creates the summary of  the dora model. '''

    fields = [
        'anti_rt',
        'inhib_fail_rt',
        'inhib_fail_prob',
        'predicted_pro_prob',
        'predicted_pro_rt',
        'predicted_anti_prob',
        'predicted_anti_rt',
            ]

    def __init__(self, *args, **kargs):

        super(SummaryProsa, self).__init__(*args, **kargs)

        return None


class SummaryProsaGamma(SummaryProsa):
    '''Class for the summaries '''

    def summary_statistics(self, samples):

        results = wrappers.p_summary_prosa_gamma(samples)

        return results


class SummaryProsaInvgamma(SummaryProsa):
    '''Class for the summaries '''

    def summary_statistics(self, samples):

        results = wrappers.p_summary_prosa_gamma(samples)

        return results


class SummaryProsaMixedgamma(SummaryProsa):
    '''Class for the summaries '''

    def summary_statistics(self, samples):

        results = wrappers.p_summary_prosa_gamma(samples)

        return results


class SummaryProsaLognorm(SummaryProsa):
    '''Class for the summaries '''

    def summary_statistics(self, samples):

        results = wrappers.p_summary_prosa_gamma(samples)

        return results


class SummaryProsaWald(SummaryProsa):
    '''Class for the summaries '''

    def summary_statistics(self, samples):

        results = wrappers.p_summary_prosa_gamma(samples)

        return results


class SummaryProsaLater(SummaryProsa):
    '''Class for the summaries '''

    def summary_statistics(self, samples):

        results = wrappers.p_summary_prosa_gamma(samples)

        return results


if __name__ == '__main__':
    pass
