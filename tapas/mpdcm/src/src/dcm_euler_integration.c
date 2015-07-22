/*%% --------------------------------------------------------------------------------------------------
% dcm_euler_integration - Integrates the DCM-fMRI dynamical system using Euler's method
% [x,s,f,v,q]  = dcm_euler_integration(A,C,U,B,D,...
                   rho,alphainv,tau,gamma,kappa,paramList);

%---------------------------------------------------------------------------------------------
% INPUT:
 * Pls note: All matrices within A, B, D are transposes of the original matrix
%       A           - Region connection matrix
%       C           - Input contribution, represents U*C' (for optimization purposes)
%       U           - Input matrix
%       B           - Bi linear connection matrix
%       D           - Non linear connection matrix
%       rho         - hemodynamic const - one for each region (array of nStates)
%       alphainv    - hemodynamic const
%       tau         - one for each region (array of nStates)
%       gamma       - hemodynamic const
%       kappa       - one for each region (array of nStates)
%       paramList   - [timeStep nTime nStates nInputs subjectNo] 
 *                      timeStep - euler step size
 *                      nTime - total steps
 *                      nStates - number of regions
 *                      nInputs - number of inputs
 *                      subjectNo- not used by function
%
% Optional:
%
%--------------------------------------------------------------------------------------------
% OUTPUT:
 *      x - neural activity
 *      s - vasodilatory signal
 *      f - blood flow
 *      v - blood volume
 *      q - deoxyhemoglobin content
%           
% -------------------------------------------------------------------------------------------
% REFERENCE:
%
% Author:   Sudhir Shankar Raman, TNU, UZH & ETHZ - 2013
%
% This file is part of TAPAS, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. For further details, see <http://www.gnu.org/licenses/>. 
% Modified by Eduardo Aponte 22.07.2015
%%*/

#include<stdio.h>
#include<math.h>
#include "mex.h"
#include "c_mpdcm.h"

void mexFunction(int nlhs, mxArray *plhs[],/*Outputvariables*/
                    int nrhs, const mxArray *prhs[])
{
    /* Extract all input parameters*/
    
    MPFLOAT *A  = (MPFLOAT *) mxGetPr(prhs[0]);
    MPFLOAT *C  = (MPFLOAT *) mxGetPr(prhs[1]);
    MPFLOAT *U  = (MPFLOAT *) mxGetPr(prhs[2]);
    MPFLOAT *B  = (MPFLOAT *) mxGetPr(prhs[3]);
    MPFLOAT *D = (MPFLOAT *) mxGetPr(prhs[4]);
    MPFLOAT *rho = (MPFLOAT *) mxGetPr(prhs[5]);
    MPFLOAT alphainv = (MPFLOAT ) mxGetScalar(prhs[6]);
    MPFLOAT *tau  = (MPFLOAT *) mxGetPr(prhs[7]);
    MPFLOAT gamma  = (MPFLOAT ) mxGetScalar(prhs[8]);
    MPFLOAT *kappa  = (MPFLOAT *) mxGetPr(prhs[9]);
    MPFLOAT *param = (MPFLOAT *) mxGetPr(prhs[10]);
    MPFLOAT temp1; 
    MPFLOAT temp2;

    double *mx_out;
    double *ms_out;
    double *mf1_out;
    double *mv1_out;
    double *mq1_out;

    int jState, mState, kIter, i;
    long iStep;
    MPFLOAT timeStep = param[0];
    long nTime = param[1];
    const int nStates = param[2];
    int nInputs = param[3];
    int dcmTypeB = param[5];
    int dcmTypeD = param[6];

    MPFLOAT *oldf1 = (MPFLOAT *) malloc(nStates * sizeof( MPFLOAT ));
    MPFLOAT *oldv1 = (MPFLOAT *) malloc(nStates * sizeof( MPFLOAT ));
    MPFLOAT *oldq1 = (MPFLOAT *) malloc(nStates * sizeof( MPFLOAT ));
   
    MPFLOAT *x_out = (MPFLOAT *) malloc(nStates * nTime * sizeof( MPFLOAT ));
    MPFLOAT *s_out = (MPFLOAT *) malloc(nStates * nTime * sizeof( MPFLOAT ));
    MPFLOAT *f1_out = (MPFLOAT *) malloc(nStates * nTime * sizeof( MPFLOAT ));
    MPFLOAT *v1_out = (MPFLOAT *) malloc(nStates * nTime * sizeof( MPFLOAT ));
    MPFLOAT *q1_out = (MPFLOAT *) malloc(nStates * nTime * sizeof( MPFLOAT ));
   
   
    /* Initialize the dynamical system to resting state values */
    for(jState=0;jState<nStates;jState++)
    {
        x_out[nTime*jState] = 0.0;
        s_out[nTime*jState] = 0.0;
        f1_out[nTime*jState] = 1.0;
        v1_out[nTime*jState] = 1.0;
        q1_out[nTime*jState] = 1.0; 
        oldf1[jState] = 0.0;
        oldv1[jState] = 0.0;
        oldq1[jState] = 0.0;
    }
    
    /* Euler's integration steps*/
    for(iStep=0;iStep<(nTime-1);iStep++)
    {        
        /* For each region*/
        for(jState=0;jState<nStates;jState++)
        {
            /* update x */
            x_out[iStep+1 + nTime*jState] = 0.0;
            temp1 = 0.0;            
            for(kIter=0;kIter<nStates;kIter++)
            {
                temp1=temp1+x_out[iStep + nTime*kIter]*A[kIter + nStates*jState];
            }            
            x_out[iStep+1 + nTime*jState]= x_out[iStep + nTime*jState] + timeStep* (temp1+ C[iStep + nTime*jState]); 
            
            if(dcmTypeB != 0)
            {
                /*B matrix update*/
                for(kIter=0;kIter<nInputs;kIter++)
                {
                    temp1 = 0.0;
                    for(mState=0;mState<nStates;mState++)
                    {
                        temp1=temp1+x_out[iStep + nTime*mState]*B[mState + nStates*(jState + nStates*kIter)];
                    }
                    x_out[iStep+1 + nTime*jState] = x_out[iStep+1 + nTime*jState] + timeStep*(U[iStep +nTime*kIter]*temp1);
                }
            }
            
            if(dcmTypeD != 0)
            {
                /*D matrix update*/
                for(kIter=0;kIter<nStates;kIter++)         
                {
                     temp1 = 0.0;
                     for(mState=0;mState<nStates;mState++)
                     {
                         temp1=temp1+x_out[iStep + nTime*mState]*D[mState + nStates*(jState + nStates*kIter)];
                     }
                     x_out[iStep+1 + nTime*jState] = x_out[iStep+1 + nTime*jState] + timeStep*(x_out[iStep +nTime*kIter]*temp1);
                }
            }
            /* update s */
            s_out[iStep+1 + nTime*jState] = s_out[iStep + nTime*jState] +
                timeStep*(x_out[iStep + nTime*jState] -
                kappa[jState]*s_out[iStep + nTime*jState] - 
                gamma*(f1_out[iStep + nTime*jState] - 1.0));
            
            /* update f */
            oldf1[jState]= oldf1[jState] + timeStep*(s_out[iStep + nTime*jState]/f1_out[iStep + nTime*jState]);
            
            /* update v */
           temp1 = pow(v1_out[iStep + nTime*jState],alphainv);
           temp2 = (1.0- pow((1.0-rho[jState]),(1.0/f1_out[iStep + nTime*jState])))/rho[jState];
            oldv1[jState]= oldv1[jState] + timeStep*((f1_out[iStep + nTime*jState] - temp1)/
                           (tau[jState]*v1_out[iStep + nTime*jState]));
            
            /* update q */
            oldq1[jState]= oldq1[jState] + 
                timeStep * ((f1_out[iStep + nTime*jState]*temp2 - 
                    temp1*q1_out[iStep + nTime*jState] / 
                    v1_out[iStep + nTime*jState]) / 
                    (tau[jState] * q1_out[iStep + nTime*jState]));

            /* tracking the exponentiated values */
            f1_out[iStep+1 + nTime*jState] = exp(oldf1[jState]);
            v1_out[iStep+1 + nTime*jState] = exp(oldv1[jState]);
            q1_out[iStep+1 + nTime*jState] = exp(oldq1[jState]);
        }       
    }

    /* Double */

    plhs[0] = mxCreateDoubleMatrix(nTime, nStates, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(nTime, nStates, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(nTime, nStates, mxREAL);
    plhs[3] = mxCreateDoubleMatrix(nTime, nStates, mxREAL);
    plhs[4] = mxCreateDoubleMatrix(nTime, nStates, mxREAL);
    
    mx_out = mxGetPr(plhs[0]);
    ms_out = mxGetPr(plhs[1]);
    mf1_out = mxGetPr(plhs[2]);
    mv1_out = mxGetPr(plhs[3]);
    mq1_out = mxGetPr(plhs[4]);    


    /* Transfer memory */

    for ( i = 0; i < nTime * nStates; i++)
        mx_out[i] = (MPFLOAT ) x_out[i];
    for ( i = 0; i < nTime * nStates; i++)
        ms_out[i] = (MPFLOAT ) s_out[i]; 
    for ( i = 0; i < nTime * nStates; i++)
        mf1_out[i] = (MPFLOAT ) f1_out[i]; 
    for ( i = 0; i < nTime * nStates; i++)
        mv1_out[i] = (MPFLOAT ) v1_out[i]; 
    for ( i = 0; i < nTime * nStates; i++)
        mq1_out[i] = (MPFLOAT ) q1_out[i]; 


    /* free all the created memory*/
    free(x_out);
    free(s_out);
    free(f1_out);
    free(v1_out);
    free(q1_out);

    free(oldf1);
    free(oldv1);
    free(oldq1);
    return;    
}

