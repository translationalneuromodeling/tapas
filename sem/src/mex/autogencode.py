#! /usr/bin/env python

# aponteeduardo@gmail.com
# copyright (C) 2017


''''

Automatically generate the necessary functions.

'''


def gen_code(family, model, param, reparam):

    preamble =  '''/* aponteeduardo@gmail.com */
/* copyright (C) 2017 */

#include "antisaccades.h"

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{{
    ANTIS_INPUT svals;
    double *llh;
    {0:s}_MODEL model;

    svals.t = mxGetPr(prhs[0]);
    svals.a = mxGetPr(prhs[1]);
    svals.u = mxGetPr(prhs[2]);
    svals.theta = mxGetPr(prhs[3]);

    svals.nt = *mxGetDimensions(prhs[0]);
    svals.np = mxGetDimensions(prhs[3])[1];

    plhs[0] = mxCreateDoubleMatrix(svals.nt, 1, mxREAL);
    llh = mxGetPr(plhs[0]);
    '''.format(family.upper())

    # likelihood
    llh = '''
    model.llh = {0:s}_llh_{1:s};'''.format(family, 'abstract')

    # Reparamterization

    rpar = '''
    model.fill_parameters = reparametrize_{0:s}_{1:s};'''.format(family, param)

    coda = '''
    {0:s}_model_{1:s}(svals, model, llh);
}}'''.format(family, model)

    fname = 'c_{0:s}_{1:s}_{2:s}{3:s}.c'.format(family, model, param,
            reparam)

    return fname, preamble + llh + rpar + coda


def gen_optimized_code(model, gslint, parametric):

    preamble = '''

/* aponteeduardo@gmail.com */
/* copyright (C) 2017 */

#include "antisaccades.h"
#ifdef HAVE_OMP_H
#include <omp.h>
#endif

void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{{
    double *llh;
    {0:s}_MODEL model;
    int i, j;
    int ns = mxGetDimensions(prhs[1])[0];
    int nc = mxGetDimensions(prhs[1])[1];

    if ( nlhs != 1 )
        HANDLE_CERROR_MSG( 1 , "Wrong left size assignments.");

    if ( nrhs != 2 )
        HANDLE_CERROR_MSG( 1 , "Wrong right size assignments." );

    plhs[0] = mxCreateDoubleMatrix(ns, nc, mxREAL);
    llh = mxGetPr(plhs[0]);

    model.llh = {1:s}_llh_{2:s};
    model.nested_integral = n{3:s}_gslint;
    model.fill_parameters = reparametrize_{1:s}_{2:s};
    gsl_set_error_handler_off();

    #pragma omp parallel for private(i) private(j) collapse(2) schedule(static)
    for (j = 0; j < nc; j++)
    {{
        for (i = 0; i < ns; i++)
        {{

            ANTIS_INPUT svals;
            mxArray *y = mxGetField(prhs[0], i, "y");
            mxArray *u = mxGetField(prhs[0], i, "u");
            mxArray *theta = mxGetCell(prhs[1], i + ns * j);
            double *tllh;
            int k;


            svals.t = mxGetPr(mxGetField(y, 0, "t"));
            svals.a = mxGetPr(mxGetField(y, 0, "a"));
            svals.u = mxGetPr(mxGetField(u, 0, "tt"));

            svals.theta = mxGetPr(theta);

            svals.nt = *mxGetDimensions(mxGetField(y, 0, "t"));
            svals.np = mxGetDimensions(theta)[1];

            tllh = (double *) malloc(svals.nt * sizeof(double));

            {1:s}_model_two_states_optimized(svals, model, tllh);

            llh[i + ns * j] = 0;
            for (k = 0; k < svals.nt; k++)
            {{
                llh[i + ns * j] += tllh[k];
            }}
            if ( abs(llh[i + ns * j]) == INFINITY || isnan(llh[i + ns * j]))
                llh[i + ns * j] = -INFINITY;
            free(tllh);

        }}
    }}
   gsl_set_error_handler(NULL);
}}
'''

    function = preamble.format(model.upper(), model, parametric, gslint)

    return function


def gen_optimized_fname(model, parametric):

    fname = 'c_{0:s}_multi_{1:s}.c'.format(model, parametric)

    return fname


def optimized_code():

    parametric = ['gamma', 'invgamma', 'mixedgamma', 'lognorm', 'later', 'wald']
    gslint = {
        'gamma': 'gamma',
        'invgamma' : 'invgamma',
        'mixedgamma': 'invgamma',
        'lognorm' : 'lognorm',
        'later' : 'later',
        'wald' : 'wald' }
    family = ['prosa', 'seri', 'dora']

    for f in family:
        for p in parametric:
            code = gen_optimized_code(f, gslint[p], p)
            fname = gen_optimized_fname(f, p)
            print fname
            #with open(fname, 'w') as fp:
            #    fp.write(code)
    return


def gen_other():

    model = ['n_states']
    parametric = ['gamma', 'invgamma', 'mixedgamma', 'lognorm', 'later',
            'wald']
    family = ['dora']

    repar = ['']

    for f in family:
        for m in model:
            for p in parametric:
                fname, code = gen_code(f, m, p, repar[0])
                print code
                with open(fname, 'w') as fp:
                    fp.write(code)
                print fname
    '''
    for f in ['seri']:
        for m in model:
            for p in parametric:
                fname, code = gen_code(f, m, p, repar[1])
                with open(fname, 'w') as fp:
                    fp.write(code)
                print fname
    '''


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
            print fname
            print code

    return


if __name__ == '__main__':
    gen_reparametrize()
    #gen_other()
    #optimized_code()
    pass
