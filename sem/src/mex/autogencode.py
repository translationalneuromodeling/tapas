#! /usr/bin/env python

# aponteeduardo@gmail.com
# copyright (C) 2017


''''

Automatically generate the necessary functions.

'''

model = ['trial_by_trial', 'two_states']
parametric = ['gamma', 'invgamma', 'mixedgamma', 'lognorm', 'later',
        'wald']
family = ['prosa', 'seri', 'dora']

if __name__ == '__main__':
    with open('template.c.tem', 'r') as fp:
        fstring = fp.read()

    fname = 'c_{0:s}_{1:s}_{2:s}.c'
    for f in family:
        for m in model:
            for p in parametric:
                fcontent = fstring.format(f, m, p)
                with open(fname.format(f, m, p), 'w') as fp:
                    fp.write(fcontent)
                 
    pass    

