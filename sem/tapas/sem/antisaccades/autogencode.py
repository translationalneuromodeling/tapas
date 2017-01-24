#! /usr/bin/env python

# aponteeduardo@gmail.com
# copyright (C) 2017


''''



'''

if __name__ == '__main__':
    model = ['two_states']
    family = ['seri', 'dora', 'prosa']
    parametric = ['gamma', 'invgamma', 'mixedgamma', 'later', 'wald']

    with open('./wrapped.tem', 'r') as fp:
        fstring = fp.read()

    with open('./likelihoods.tem', 'r') as fp:
        sllh = fp.read()

    for f in family:
        for m in model:
            for p in parametric:
                #print fstring.format(f, p, m, f.upper())
                print sllh.format(f, m, p)


    pass    

