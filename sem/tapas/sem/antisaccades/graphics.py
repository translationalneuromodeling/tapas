#! /usr/bin/env python

# aponteeduardo@gmail.com
# copyright (C) 2016


''''

Graphical tools

'''


from copy import deepcopy
import numpy as np
from scipy.integrate import cumtrapz
from scipy import stats

from . import containers


def generalized_fits_factory(trials):

    class GeneralizedFits(containers.TimeSeries):

        fields = trials

        def __init__(self, *args, **kargs):

            try:
                offset = kargs.pop('offset')
            except KeyError:
                offset = 0

            super(GeneralizedFits, self).__init__(*args, **kargs)

            self.offset = offset

            return

        def set_offset(self, offset):

            self.offset = offset

            return

        def get_time(self):

            return self.time + self.offset

        def scale(self, fields):

            nobj = deepcopy(self)

            for key in fields:
                # Make sure that this key is acceptable
                if not key in self.__class__.fields:
                    raise(KeyError)

                nobj.__dict__[key] = self.__dict__[key] * fields[key]

            return nobj

    return GeneralizedFits


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
    LocalFits = generalized_fits_factory(trials)

    return LocalFits(llh)


if __name__ == '__main__':
    pass
