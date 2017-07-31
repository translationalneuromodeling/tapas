#! /usr/bin/env python

# aponteeduardo@gmail.com
# copyright (C) 2016


''''

Graphical tools

'''


from pdb import set_trace as _
from copy import deepcopy
import numpy as np
from scipy.integrate import cumtrapz
from scipy import stats

import containers
import reparametrize as reparam
import likelihoods 

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

class SummaryDora(SummaryStats):
    ''' Creates the summary of  the dora model. '''
    
    # Fields are early unit, late unit, stop unit, late prosaccade, 
    # probability of a late antisaccade and late errors
    fields = ['e', # Early response
            'a', # antisaccades
            's', # Stop signal
            'p', # Late prosaccades
            'pia', # Late antisaccade 
            'pip', # Probability of early prosaccade
            'nst' # Non stop trials
            ]

    def __init__(self, *args, **kargs):

        super(SummaryDora, self).__init__(*args, **kargs)

        return None

class SummaryDoraInvgamma(SummaryDora):

    def __int__(self, *args, **kargs):

        super(SummaryDoraInvgamma, self).__init__(*args, **kargs)

        return

    def summary_statistics(self, stheta):

        results = {}

        time = np.linspace(0, 15, 1000)
        results['e'] = stheta.tp / (stheta.kp - 1) + stheta.t0
       
        p_lpdf = stats.invgamma.logpdf(time, stheta.kp, scale=stheta.tp)
        s_lsf = stats.invgamma.logsf(time, stheta.ks, scale=stheta.ts) 
      
        l_lpdf = stats.invgamma.logpdf(time, stheta.kl, 
                scale=stheta.tl, loc=stheta.t0 + stheta.da) 
        a_lpdf = stats.invgamma.logpdf(time, stheta.ka, 
		scale=stheta.ta, loc=stheta.t0 + stheta.da) 

        a_lsf = stats.invgamma.logsf(time, stheta.ka, scale=stheta.ta, 
                loc=stheta.da + stheta.t0) 
        l_lsf = stats.invgamma.logsf(time, stheta.kl, scale=stheta.tl,
                loc=stheta.da + stheta.t0)


        pr = a_lpdf + l_lsf
        
        results['pia'] = np.trapz(np.exp(pr), time)
        results['a'] = np.trapz(time * np.exp(pr), time) / results['pia']

        pr = l_lpdf + a_lsf
        
        results['p'] = np.trapz(time * np.exp(pr), time) / (1.0 - results['pia'])
        results['s'] = stheta.ts / (stheta.ks - 1) + stheta.t0

        pr = p_lpdf + s_lsf + l_lsf + a_lsf - stheta.p0/2.0 - np.log(2.0) - \
                np.log(np.cosh(stheta.p0/2.0))
        results['pip'] = np.trapz(np.exp(pr), time)

        pr = p_lpdf + s_lsf 
        results['nst'] = np.trapz(time * np.exp(pr), time) / np.trapz(np.exp(pr), time)

        return results



class SummaryDoraMixedgamma(SummaryDora):
    """Summary of the DORA model using the mixed gamma combination."""

    def __int__(self, *args, **kargs):

        super(SummaryDoraMixedgamma, self).__init__(*args, **kargs)

        return

    def summary_statistics(self, stheta):

        results = {}

        time = np.linspace(0, 15, 1000)
        results['e'] = stheta.tp / (stheta.kp - 1) + stheta.t0
       
        p_lpdf = stats.invgamma.logpdf(time, stheta.kp, scale=stheta.tp)
        s_lsf = stats.invgamma.logsf(time, stheta.ks, scale=stheta.ts) 
      
        l_lpdf = stats.gamma.logpdf(time, stheta.kl, 
                scale=stheta.tl, loc=stheta.t0 + stheta.da) 
        a_lpdf = stats.gamma.logpdf(time, stheta.ka, 
		scale=stheta.ta, loc=stheta.t0 + stheta.da) 

        a_lsf = stats.gamma.logsf(time, stheta.ka, scale=stheta.ta, 
                loc=stheta.da + stheta.t0) 
        l_lsf = stats.gamma.logsf(time, stheta.kl, scale=stheta.tl,
                loc=stheta.da + stheta.t0)


        pr = a_lpdf + l_lsf
        
        results['pia'] = np.trapz(np.exp(pr), time)
        if results['pia'] == 0.0:
            results['pia'] = 0.000000000001
        if results['pia'] >= 1.0:
            results['pia'] = 0.999999999999

        results['a'] = np.trapz(time * np.exp(pr), time) / results['pia']

        pr = l_lpdf + a_lsf
        
        results['p'] = np.trapz(time * np.exp(pr), time) / (1.0 - results['pia'])
        results['s'] = stheta.ts / (stheta.ks - 1) + stheta.t0

        pr = p_lpdf + s_lsf + l_lsf + a_lsf - stheta.p0/2.0 - np.log(2.0) - \
                lcosh(stheta.p0/2.0) 
        results['pip'] = np.trapz(np.exp(pr), time)

        pr = p_lpdf + s_lsf 
        results['nst'] = np.trapz(time * np.exp(pr), time) / np.trapz(np.exp(pr), time)

        #pr = p_lpdf + s_lsf 
        #results['nst'] = np.trapz(np.exp(pr), time)

        return results

class SummaryDoraLognorm(SummaryDora):
    """Summary of the DORA model using the mixed gamma combination."""


    def __int__(self, *args, **kargs):

        super(SummaryDoraLognorm, self).__init__(*args, **kargs)

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
        
        results['p'] = np.trapz(time * np.exp(pr), time) / (1.0 - results['pia'])
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
        #_()

        pr = stats.lognorm.pdf(time, stheta.kp, scale=stheta.tp) * \
             stats.lognorm.sf(time, stheta.ks, scale=stheta.ts)
        results['nst'] = np.trapz(pr, time)

        return results

        return results



class SummarySeri(SummaryStats):
    ''' Creates the summary of  the dora model. '''

    fields = ['p', 'a', 's', 'ap']

    def __init__(self, *args, **kargs):

        super(SummarySeri, self).__init__(*args, **kargs)

        return None

class SummarySeriInvgamma(SummarySeri):

    def __int__(self, *args, **kargs):

        super(SummarySeriInvgamma, self).__init__(*args, **kargs)

        return

    def summary_statistics(self, stheta):

        results = {}

        results['p'] = stheta.tp / (stheta.kp - 1) + stheta.t0
        results['a'] = stheta.ta / (stheta.ka - 1) + stheta.t0 + \
                stheta.da
        results['s'] = stheta.ts / (stheta.ks - 1) + stheta.t0
        
        results['ap'] = stheta.ap 

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
    
def generate_fits(theta, model, maxt=8.0, ns=100):

    results = gen_llh(theta, model, maxt, ns)

    return Fits(results)


if __name__ == '__main__':
    pass    

