#! /usr/bin/env python

# aponteeduardo@gmail.com
# copyright (C) 2017


''''

Automatically generate the necessary functions.

'''


def gen_code_n_states(f, p):

    fname = 'c_{0:s}_n_states_{1:s}.c'.format(f, p)

    code = '''/* aponteeduardo@gmail.com */
/* copyright (C) 2018 */

#include "mexutils.h"

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{{

    wrapper_{0:s}_n_states(nlhs, plhs, nrhs, prhs, reparametrize_{0:s}_{1:s});

}}'''.format(f, p)

    return fname, code


def gen_n_states():

    parametric = [
            'gamma', 'invgamma', 'mixedgamma', 'lognorm', 'later',
            'wald']
    family = ['prosa', 'seria']

    for f in family:
        for p in parametric:
            fname, code = gen_code_n_states(f, p)
            with open(fname, 'w') as fp:
                fp.write(code)
                pass
            print(fname)
            print(code)

    return


def gen_code_multi(f, p):

    fname = 'c_{0:s}_multi_{1:s}.c'.format(f, p)

    code = '''/* aponteeduardo@gmail.com */
/* copyright (C) 2018 */

#include "mexutils.h"

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{{

    wrapper_{0:s}_multi(nlhs, plhs, nrhs, prhs, reparametrize_{0:s}_{1:s});

}}'''.format(f, p)

    return fname, code


def gen_multi():

    parametric = [
            'gamma', 'invgamma', 'mixedgamma', 'lognorm', 'later',
            'wald']
    family = ['prosa', 'seria']

    for f in family:
        for p in parametric:
            fname, code = gen_code_multi(f, p)
            with open(fname, 'w') as fp:
                fp.write(code)
                pass
            print(fname)
            print(code)

    return


def gen_reparametrize():

    parametric = [
            'gamma', 'invgamma', 'mixedgamma', 'lognorm', 'later',
            'wald']
    family = ['prosa', 'seria']

    for f in family:
        for p in parametric:
            fname, code = gen_code_reparametrize(f, p)
            with open(fname, 'w') as fp:
                fp.write(code)
                pass
            print(fname)
            print(code)

    return


def gen_code_reparametrize(f, p):

    fname = 'c_{0:s}_reparametrize_{1:s}.c'.format(f, p)

    code = '''/* aponteeduardo@gmail.com */
/* copyright (C) 2018 */

#include "antisaccades.h"
#include "mexutils.h"

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{{

    reparametrize_{0:s}(nlhs, plhs, nrhs, prhs, reparametrize_{0:s}_{1:s});

}}'''.format(f, p)

    return fname, code


if __name__ == '__main__':
    #gen_reparametrize()
    gen_multi()
    gen_n_states()
    pass
