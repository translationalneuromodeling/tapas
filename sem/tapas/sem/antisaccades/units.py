#! /usr/bin/env python

# aponteeduardo@gmail.com
# copyright (C) 2016


''''

Graphical tools

'''


from pdb import set_trace as _
import numpy as np
from scipy import stats

from . import containers


class Units(containers.TimeSeries):
    ''' Abstract class for units, where units refer to the distribution
    of the units. '''

    pass


class UnitsSeria(containers.TimeSeries):

    fields = ['e', 'a', 'i', 'p', 'ns']

    def __init__(self, time=None, samples=None):

        if samples is None:
            super(UnitsSeria, self).__init__()
        else:
            results = self.llh(time, samples)
            super(UnitsSeria, self).__init__(results)

        return


class UnitsSeriaMixedgamma(UnitsSeria):
    ''' Seria untis with mixed gamma. '''

    def __init__(self, time, samples):
        '''Seria untis with mixedgamma. '''
        super(UnitsSeriaMixedgamma, self).__init__(time, samples)

        return None

    def llh(self, time, stheta):

        results = {}
        results['time'] = time

        e_lpdf = stats.invgamma.logpdf(
            time, stheta.kp, scale=stheta.tp, loc=stheta.t0)

        a_lpdf = stats.gamma.logpdf(
            time, stheta.ka, scale=stheta.ta, loc=stheta.t0 + stheta.da)
        p_lsf = stats.gamma.logsf(
            time, stheta.kl, scale=stheta.tl, loc=stheta.t0 + stheta.da)

        results['e'] = e_lpdf
        results['e'] += - np.log(np.trapz(np.exp(results['e']), time))

        results['a'] = a_lpdf + p_lsf
        nc = np.trapz(np.exp(results['a']), time)
        results['a'] -= np.log(nc)
        results['p'] = a_lpdf

        results['i'] = stats.invgamma.logpdf(time, stheta.ks,
            scale=stheta.ts, loc=stheta.t0)

        results['ns'] = results['e'] + stats.invgamma.logsf(time, stheta.ks,
            scale=stheta.ts, loc=stheta.t0)
        results['ns'] = results['ns'] - \
            np.log(np.trapz(np.exp(results['ns']), time))


        return results


class UnitsSeriaInvgamma(UnitsSeria):
    ''' Seria units with invgamma. '''

    def __init__(self, time, samples):

        super(UnitsSeriaInvgamma, self).__init__(time, samples)

        return None

    def llh(self, time, stheta):

        results = {}
        results['time'] = time

        e_lsf = stats.invgamma.logsf(time, stheta.kp, scale=stheta.tp,
                loc=stheta.t0)
        e_lpdf = stats.invgamma.logpdf(time, stheta.kp, scale=stheta.tp,
                loc=stheta.t0)
        e_lcdf = stats.invgamma.logcdf(time, stheta.kp, scale=stheta.tp,
                loc=stheta.t0)

        i_lsf = stats.invgamma.logsf(time, stheta.ks, scale=stheta.ts,
                loc=stheta.t0)
        i_lcdf = stats.invgamma.logcdf(time, stheta.ks, scale=stheta.ts,
                loc=stheta.t0)


        a_lpdf = stats.invgamma.logpdf(time, stheta.ka,
                scale=stheta.ta, loc=stheta.t0 + stheta.da)
        p_lsf = stats.invgamma.logsf(time, stheta.kl,
                scale=stheta.tl, loc=stheta.t0 + stheta.da)

        p_lpdf = stats.invgamma.logpdf(time, stheta.kl,
                scale=stheta.tl, loc=stheta.t0 + stheta.da)
        a_lsf = stats.invgamma.logsf(time, stheta.ka,
                scale=stheta.ta, loc=stheta.t0 + stheta.da)

        results['e'] = e_lpdf #: + i_lsf + p_lsf + a_lsf
        #results['e'] += np.exp(p_lpdf + i_lsf + a_lsf + e_lsf)
        #results['e'] = np.log(results['e'])
        #results['e'] += p_lpdf + i_lsf + a_lsf + e_lsf
        results['e'] += - np.log(np.trapz(np.exp(results['e']), time))



        results['a'] = a_lpdf + p_lsf #i_lcdf + e_lcdf + p_lsf
        nc =  np.trapz(np.exp(results['a']), time)
        results['a'] -= np.log(nc)
        results['p'] = a_lpdf
        #-np.log(1 - nc) + p_lpdf + a_lsf

        results['i'] = stats.invgamma.logpdf(time, stheta.ks,
            scale=stheta.ts, loc=stheta.t0)

        results['ns'] = results['e'] + stats.invgamma.logsf(time, stheta.ks,
            scale=stheta.ts, loc=stheta.t0)
        results['ns'] = results['ns'] - \
            np.log(np.trapz(np.exp(results['ns']), time))

        return results


class UnitsSeriaLognorm(UnitsSeria):
    ''' Seria units with invgamma. '''

    def __init__(self, time, samples):

        super(UnitsSeriaInvgamma, self).__init__(time, samples)

        return None

    def llh(self, time, stheta):

        results = {}
        results['time'] = time

        results['e'] = stats.lognorm.logpdf(time, stheta.kp,
            scale=stheta.tp, loc=stheta.t0)

        results['a'] = stats.lognorm.logpdf(time, stheta.ka,
            scale=stheta.ta, loc=stheta.t0 + stheta.da) + \
            stats.lognorm.logsf(time, stheta.kl,
                scale=stheta.tl, loc=stheta.t0 + stheta.da)

        results['p'] = stats.lognorm.logpdf(time, stheta.kl,
                scale=stheta.tl, loc=stheta.t0 + stheta.da) + \
            stats.lognorm.logsf(time, stheta.ka,
                scale=stheta.ta, loc=stheta.t0 + stheta.da)

        results['i'] = stats.lognorm.logpdf(time, stheta.ks,
            scale=stheta.ts, loc=stheta.t0)

        return results


if __name__ == '__main__':
    pass
