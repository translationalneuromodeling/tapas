/* aponteeduardo@gmail.com */
/* copyright (C) 2014 */


#ifndef C_MPDCM
#define C_MPDCM

#include <matrix.h>
#include <mex.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

void mpdcm_fmri_int(mxArray **y, const mxArray **u,
    const mxArray **theta, const mxArray **ptheta,
    int *ny0, int *ny1, int *nu0, int *nu1, int *nt0, int *nt1);
    /* Integrates an array of dcm with input u, parameters theta, and priors
       ptheta.
     
       y -- Cell array output
       u -- Cell array of inputs
       theta -- Cell array of parameters
       ptheta -- Structure of priors
       ny0 -- Input: First dimension of the output
       ny1 -- Input: Second dimension of the output
       nu0 -- Input: First dimension of the input
       ny1 -- Input: Second dimension of the output
       nt0 -- Input: First dimension of theta
       nt1 -- Input: Second dimension of theta
    */

#endif
