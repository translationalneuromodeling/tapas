#! /usr/bin/env python

# aponteeduardo@gmail.com
# copyright (C) 2016


''''

Graphical tools

'''


import numpy as np
from scipy.integrate import cumtrapz
from scipy import stats

import containers


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

    # Fields are early unit, late unit, stop unit, late prosaccade,
    # probability of a late antisaccade and late errors
    fields = [
            'te',  # Early response 0
            'ts',  # Stop signal 1
            'ta',  # antisaccades 2
            'tlp',  # Late prosaccades 3
            'ter',  # Early reaction 4
            'tif',  # Non stop trials hit time 5
            'per',  # Probability of early prosaccade 6
            'pif',  # Probability of an inhibition failure 7
            'pp',  # Probability of a prosaccade 8
            'pa',  # Late antisaccade 9
            ]

    def __init__(self, *args, **kargs):

        super(SummarySeria, self).__init__(*args, **kargs)

        return None


class SummarySeriaInvgamma(SummaryDora):

    def __int__(self, *args, **kargs):

        super(SummarySeriaInvgamma, self).__init__(*args, **kargs)

        return


class SummarySeriaMixedgamma(SummaryDora):
    """Summary of the DORA model using the mixed gamma combination."""

    def __int__(self, *args, **kargs):

        super(SummarySeriaMixedgamma, self).__init__(*args, **kargs)

        return

    def summary_statistics(self, stheta):

        results = {}

        time = np.linspace(0, 15, 1000)

        results['te'] = stheta.tp / (stheta.kp - 1) + stheta.t0
        results['ts'] = stheta.ts / (stheta.ks - 1) + stheta.t0

        p_lpdf = stats.invgamma.logpdf(
            time, stheta.kp, scale=stheta.tp, loc=stheta.t0)
        p_lsf = stats.invgamma.logsf(
            time, stheta.kp, scale=stheta.tp, loc=stheta.t0)

        s_lsf = stats.invgamma.logsf(
            time, stheta.ks, scale=stheta.ts, loc=stheta.t0)
        s_lpdf = stats.invgamma.logpdf(
            time, stheta.ks, scale=stheta.ts, loc=stheta.t0)

        l_lpdf = stats.gamma.logpdf(
            time, stheta.kl, scale=stheta.tl, loc=stheta.t0 + stheta.da)
        a_lpdf = stats.gamma.logpdf(
            time, stheta.ka, scale=stheta.ta, loc=stheta.t0 + stheta.da)

        a_lsf = stats.gamma.logsf(
            time, stheta.ka, scale=stheta.ta, loc=stheta.da + stheta.t0)
        l_lsf = stats.gamma.logsf(
            time, stheta.kl, scale=stheta.tl, loc=stheta.da + stheta.t0)

        pr = a_lpdf + l_lsf

        results['pa'] = np.trapz(np.exp(pr), time)
        if results['pa'] == 0.0:
            results['pa'] = 0.000000000001
        if results['pa'] >= 1.0:
            results['pa'] = 0.999999999999

        results['ta'] = np.trapz(time * np.exp(pr), time) / results['pa']

        pr = l_lpdf + a_lsf
        results['tlp'] = \
            np.trapz(time * np.exp(pr), time)/(1.0 - results['pa'])

        results['pp'] = np.trapz(
            np.exp(p_lpdf + s_lsf + l_lsf + a_lsf) +
            np.exp(l_lpdf + a_lsf + p_lsf + s_lsf) +
            np.exp(
                l_lpdf + a_lsf +
                np.log(cumtrapz(np.exp(s_lpdf + p_lsf), time, initial=0.0))),
            time)

        pr = p_lpdf + s_lsf
        results['per'] = np.trapz(np.exp(pr), time)
        results['ter'] = np.trapz(time * np.exp(pr), time)/results['per']

        pr = p_lpdf + s_lsf + l_lsf + a_lsf
        results['pif'] = np.trapz(np.exp(pr), time)
        results['tif'] = np.trapz(time * np.exp(pr), time)/results['pif']

        return results


class SummarySeriaLognorm(SummaryDora):
    """Summary of the DORA model using the mixed gamma combination."""

    def __int__(self, *args, **kargs):

        super(SummarySeriaLognorm, self).__init__(*args, **kargs)

        return

    def summary_statistics(self, stheta):

        results = {}

        time = np.linspace(0, 15, 400)

        results['e'] = np.exp(stheta.ka + stheta.ta**2.0 / 2.0)

        pr = stats.lognorm.logpdf(time, stheta.ka,
            scale=stheta.ta, loc=stheta.t0 + stheta.da) + \
            stats.lognorm.logsf(time, stheta.kl,
                scale=stheta.tl, loc=stheta.t0 + stheta.da)

        results['pia'] = np.trapz(np.exp(pr), time)
        results['a'] = np.trapz(time * np.exp(pr), time) / results['pia']

        pr = stats.lognorm.logpdf(time, stheta.kl,
                scale=stheta.tl, loc=stheta.t0 + stheta.da) + \
            stats.lognorm.logsf(time, stheta.ka,
            scale=stheta.ta, loc=stheta.t0 + stheta.da)

        results['p'] = np.trapz(time * np.exp(pr), time) / \
            (1.0 - results['pia'])
        results['s'] = np.exp(stheta.ka + stheta.ta**2.0 / 2)

        pr = stats.lognorm.logpdf(time, stheta.kp,
                scale=stheta.tp, loc=stheta.t0) + \
            stats.lognorm.logsf(time, stheta.ks,
                    scale=stheta.ts, loc=stheta.t0) + \
            stats.lognorm.logsf(time, stheta.ka, scale=stheta.ta,
                    loc=stheta.t0 + stheta.da) + \
            stats.lognorm.logsf(time, stheta.kl, scale=stheta.tl,
                    loc=stheta.t0 + stheta.da) + \
                -np.log(1.0 + np.exp(-stheta.p0))

        results['pip'] = np.trapz(np.exp(pr), time)

        pr = stats.lognorm.pdf(time, stheta.kp, scale=stheta.tp) * \
             stats.lognorm.sf(time, stheta.ks, scale=stheta.ts)
        results['nst'] = np.trapz(pr, time)

        return results


def gen_llh(theta, model, maxt=8.0, ns=100):
    ''' Generate the likelihood of a model.

    theta       -- Array of double
    model       -- likelihood function
    maxt        -- Limit of integration
    ns          -- Grid size


    '''

    # Time offset

    a = np.zeros((ns, 1))
    tt = np.zeros((ns, 1))
    t = np.linspace(0.0, maxt, ns)

    results = {'time': t}

    # Prosaccades in prosaccade trial
    results['pp'] = model(t, a, tt, theta)

    # Prosaccade in antisccade trial
    tt[:] = 1
    results['ap'] = model(t, a, tt, theta)

    # Antisaccade in prosaccade trial

    a[:] = 1
    tt[:] = 0

    results['pa'] = model(t, a, tt, theta)

    # Antisaccad in antisaccade trial

    a[:] = 1
    tt[:] = 1

    results['aa'] = model(t, a, tt, theta)

    return results


if __name__ == '__main__':
    pass
