#! /usr/bin/env python

# aponteeduardo@gmail.com
# copyright (C) 2016


''''

Graphical tools

'''

from pdb import set_trace as _
import numpy as np

def gen_llh(theta, model, maxt=8.0, ns=100):
    ''' Generate the likelihood of a model. '''

    if len(theta) == 16:
        t0 = theta[12]
    if len(theta) == 20:
        t0 = theta[16]
    if len(theta) == 22:
        t0 = theta[8]

    # Time offset
    
    a = np.zeros((ns, 1))
    tt = np.zeros((ns, 1)) 
    t = np.linspace(0.0, maxt, ns)

    results = {'t': t}

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

    ip = np.zeros((ns))
    ip[:] = -np.inf

    results['ip'] = ip
    results['ip'][t < t0] = results['pp'][t < t0]
    results['pp'][t < t0] = -np.inf
    results['ap'][t < t0] = -np.inf
    results['pa'][t < t0] = -np.inf
    results['aa'][t < t0] = -np.inf
    
    return results
    


if __name__ == '__main__':
    pass    

