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

class Fits(containers.TimeSeries):
    ''' An object containing the fits. '''

    fields = ('pp', 'pa', 'ap', 'aa')

    def __init__(self, *args, **kargs):

        super(Fits, self).__init__(*args, **kargs)

        return


    def scale(self, nnp, nna):
        '''Scale only by trial type. '''

        nobj = deepcopy(self)

        nobj.pp = self.pp * nnp
        nobj.ap = self.ap * nna
        nobj.pa = self.pa * nnp
        nobj.aa = self.aa * nna

        return nobj

def generalize_fits_factory(trials):

    class GeneralizedFits(containers.TimeSeries):

        fields = trials

        def __init__(self, *args, **kargs):

            super(GeneralizedFits, self).__init__(*args, **kargs)

            return

        def scale(self, fields):

            nobj = deepcopy(self)

            for key in fields:
                # Make sure that this key is acceptable
                if not key in self.__class__.fields:
                    raise(KeyError)

                nobj.__dict__[key] = self.__dict__[key] * fields[key]

            return nobj

    return GeneralizedFits

def generate_llh_no_outliers(theta, model, time, t0):
    ''' Generate the likelihood of a model but zero everything below t0.

    theta       -- Array of double
    model       -- likelihood function
    time        -- A vector of time points.

    '''

    results = generate_llh(theta, model, time)
    v = time < t0

    results['aa'][v] = -np.inf
    results['ap'][v] = -np.inf
    results['pp'][v] = -np.inf
    results['pa'][v] = -np.inf


    return Fits(results)

def generalized_llh(theta, model, time, trials):

    ns = len(time)

    a = np.zeros((ns, 1))
    tt = np.zeros((ns, 1))

    results = {'time': time}
    for trial in trials:
        a[:] = trial[0]
        tt[:] = trial[1]
        results[trial] = model(time, a, tt, theta)

    return results


def generalized_fits(theta, model, time, trials):

    llh = generalized_llh(theta, model, time, trials)
    LocalFits = generalize_fits_factory(trials)

    return LocalFits(llh)

def generate_llh(theta, model, time):
    ''' Generate the likelihood of a model.

    theta       -- Array of double
    model       -- likelihood function
    time        -- A vector of time points.


    '''

    ns = len(time)

    a = np.zeros((ns, 1))
    tt = np.zeros((ns, 1))

    results = {'time': time}

    # Prosaccades in prosaccade trial
    results['pp'] = model(time, a, tt, theta)

    # Prosaccade in antisccade trial
    tt[:] = 1
    results['ap'] = model(time, a, tt, theta)

    # Antisaccade in prosaccade trial

    a[:] = 1
    tt[:] = 0

    results['pa'] = model(time, a, tt, theta)

    # Antisaccad in antisaccade trial

    a[:] = 1
    tt[:] = 1

    results['aa'] = model(time, a, tt, theta)

    return results



def gen_llh(theta, model, maxt=8.0, ns=100):
    ''' Generate the likelihood of a model.

    theta       -- Array of double
    model       -- likelihood function
    maxt        -- Limit of integration
    ns          -- Grid size


    '''

    # Time offset

    t = np.linspace(0.0, maxt, ns)

    return generate_llh(theta, model, t)

def generate_fits(theta, model, maxt=8.0, ns=100):

    results = gen_llh(theta, model, maxt, ns)

    return Fits(results)


if __name__ == '__main__':
    pass

