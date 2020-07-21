#! /usr/bin/env python

# aponteeduardo@gmail.com
# copyright (C) 2017


''''

Basic container classes.

'''


from collections import Sequence
import numpy as np
from copy import deepcopy
from scipy.integrate import cumtrapz


class AlgebraicObject(object):
    ''' A class for the units statistics. '''

    fields = []
    fix_fields = []

    def __init__(self, results=None):

        if results is None:
            for f in self.__class__.fix_fields:
                self.__dict__[f] = np.zeros((0, ))
            for f in self.__class__.fields:
                self.__dict__[f] = np.zeros((0, ))
        else:
            for f in self.__class__.fix_fields:
                self.__dict__[f] = results[f]
            for f in self.__class__.fields:
                self.__dict__[f] = results[f]

        return None

    def set_values(self, values):

        for f in self.__class__.fix_fields:
            self.__dict__[f] = values[f]
        for f in self.__class__.fields:
            self.__dict__[f] = values[f]

        return

    def __add__(self, other):

        nobj = deepcopy(self)

        for f in self.__class__.fields:
            nobj.__dict__[f] += other.__dict__[f]

        return nobj

    def __sub__(self, other):

        nobj = deepcopy(self)

        for f in self.__class__.fields:
            nobj.__dict__[f] -= other.__dict__[f]

        return nobj

    def sqrt(self):

        nobj = deepcopy(self)

        for f in self.__class__.fields:
            nobj.__dict__[f] = np.sqrt(nobj.__dict__[f])

        return nobj

    def __mul__(self, scalar):
        ''' Multiply object to scaler. '''

        nobj = deepcopy(self)

        for f in self.__class__.fields:
            nobj.__dict__[f] *= scalar

        return nobj

    # Make scalar multiplication commutative
    __rmul__ = __mul__

    def __truediv__(self, arg):

        return self.__mul__(1./arg)

    def exp(self):

        nobj = deepcopy(self)

        for f in self.__class__.fields:
            nobj.__dict__[f] = np.exp(self.__dict__[f])

        return nobj

    def __mod__(self, other):

        nobj = deepcopy(self)

        for f in self.__class__.fields:
            nobj.__dict__[f] *= other.__dict__[f]

        return nobj

    def __div__(self, scalar):

        nobj = deepcopy(self)

        for f in self.__class__.fields:
            nobj.__dict__[f] /= scalar

        return nobj

    def __getitem__(self, key):

        nobj = self.__class__()

        for f in self.__class__.fix_fields:
            nobj.__dict__[f] = self.__dict__[f][key]

        for f in self.__class__.fields:
            nobj.__dict__[f] = self.__dict__[f][key]

        return nobj

    def __str__(self):

        strt = ''

        for f in self.__class__.fields:
            strt += '{0:s} '.format(f)

        return strt

    def __len__(self):

        return len(self.__dict__[self.__class__.fields[0]])

    def isnan(self):

        val = True
        for f in self.__class__.fields:
            val = np.logical_and(val,
                not np.any(np.isnan(self.__dict__[f])))

        return not val

    def isinf(self):

        val = True
        for f in self.__class__.fields:
            val = np.logical_and(val,
                not np.any(np.isinf(self.__dict__[f])))

        return not val

    def get_raw_values(self):
        ''' Get the raw values. '''

        return [getattr(self, field) for field in self.__class__.fields]


class TimeSeries(AlgebraicObject):

    fix_fields = ['time']

    def __init__(self, *args, **kargs):

        super(TimeSeries, self).__init__(*args, **kargs)

        return None

    def cumsum(self):

        nobj = self.__class__()
        nobj.time = self.time

        for f in self.__class__.fields:
            nobj.__dict__[f] = cumtrapz(self.__dict__[f], self.time)

        return nobj


class FitsContainer(Sequence):

    def __init__(self, *items):

        self.data = list(items)

        return None

    def __len__(self):

        return len(self.data)

    def __getitem__(self, key):

        if isinstance(key, slice):
            return FitsContainer(*(self.data[key]))
        else:
            return self.data[key]

        #return None

    def __str__(self):

        mystr = []
        for i in self:
            mystr += [i]

        return mystr.__str__()

    def __add__(self, other):
        '''Addition of lists. '''

        obj = FitsContainer()
        obj.data = self.data + other.data

        return obj

    def append(self, val):

        self.data.append(val)
        return None

    def sum(self):

        if len(self) == 0:
            raise IndexError

        nobj = deepcopy(self[0])

        for i in self[1:]:
            nobj += i

        return nobj

    def mean(self):

        if len(self) == 0:
            raise IndexError

        nobj = deepcopy(self[0])

        for i in self[1:]:
            nobj += i

        return nobj/float(len(self))

    def var(self):
        '''Variance of the fits. '''

        if len(self) < 2:
            raise IndexError

        ev = self.mean()

        nobj = deepcopy(self[0]) * 0

        for obj in self:
            nobj += (obj - ev) % (obj - ev)

        return nobj/float(len(self) - 1)


if __name__ == '__main__':
    pass

