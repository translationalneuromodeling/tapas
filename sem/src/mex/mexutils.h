/* aponteeduardo@gmail.com */
/* copyright (C) 2018 */

#ifndef MEXUTILS_H
#define MEXUTILS_H

#include <matrix.h>
#include <mex.h>
#include "antisaccades.h"

#ifdef HAVE_OMP_H
#include <omp.h>
#endif

void
wrapper_seria_n_states(
        int nlhs,
        mxArray *plhs[],
        int nrhs,
        const mxArray *prhs[],
        FILL_PARAMETERS_SERIA reparametrize);

void
wrapper_prosa_n_states(
        int nlhs, 
        mxArray *plhs[], 
        int nrhs, 
        const mxArray *prhs[],
        FILL_PARAMETERS_PROSA reparametrize);

void
wrapper_seria_multi(
        int nlhs,
        mxArray *plhs[],
        int nrhs,
        const mxArray *prhs[],
        FILL_PARAMETERS_SERIA reparametrize);

void
wrapper_prosa_multi(
        int nlhs, 
        mxArray *plhs[], 
        int nrhs, 
        const mxArray *prhs[],
        FILL_PARAMETERS_PROSA reparametrize);

void
reparametrize_prosa(
        int nlhs,
        mxArray *plhs[], 
        int nrhs,
        const mxArray *prhs[],
        FILL_PARAMETERS_PROSA reparametrize);


void
reparametrize_seria(
        int nlhs,
        mxArray *plhs[], 
        int nrhs,
        const mxArray *prhs[],
        FILL_PARAMETERS_SERIA reparametrize);

/* Generate the summary of the parameters */
void
wrapper_seria_summaries(
        int nlhs,
        mxArray *plhs[], 
        int nrhs,
        const mxArray *prhs[],
        FILL_PARAMETERS_SERIA reparametrize); 

void
wrapper_prosa_summaries(
        int nlhs,
        mxArray *plhs[], 
        int nrhs,
        const mxArray *prhs[],
        FILL_PARAMETERS_PROSA reparametrize); 

#endif // MEXUTILS_H
