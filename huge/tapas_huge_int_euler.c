/*%% --------------------------------------------------------------------------------------------------
% dcm_euler_integration - Integrates the DCM-fMRI dynamical system using 
 * Euler's method
% [x,s,f,v,q]  = dcm_euler_integration(A,C,U,B,D,...
                   rho,alphainv,tau,gamma,kappa,paramList);

%---------------------------------------------------------------------------------------------
% INPUT:
 * Pls note: All matrices within A, B, D are transposes of the original
 * matrix
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
%--------------------------------------------------------------------------------------------
% OUTPUT:
 *      x - neural activity
 *      s - vasodilatory signal
 *      f - blood flow
 *      v - blood volume
 *      q - deoxyhemoglobin content
%           

%
% Author: Sudhir Shankar Raman
% Copyright (C) 2018 Translational Neuromodeling Unit
%                    Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
% 
% This file is part of TAPAS, which is released under the terms of the GNU
% General Public Licence (GPL), version 3. For further details, see
% <https://www.gnu.org/licenses/>.
% 
% This software is provided "as is", without warranty of any kind, express
% or implied, including, but not limited to the warranties of
% merchantability, fitness for a particular purpose and non-infringement.
% 
% This software is intended for research only. Do not use for clinical
% purpose. Please note that this toolbox is under active development.
% Considerable changes may occur in future releases. For support please
% refer to:
% https://github.com/translationalneuromodeling/tapas/issues
% 


%%*/
#include<stdio.h>
#include<math.h>
#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[],/*Outputvariables*/
                    int nrhs, const mxArray *prhs[])
{
    /* Extract all input parameters*/
    
    double *A  = mxGetPr(prhs[0]);
    double *C  = mxGetPr(prhs[1]);
    double *U  = mxGetPr(prhs[2]);
    double *B  = mxGetPr(prhs[3]);
    double *D = mxGetPr(prhs[4]);
    double *rho  = mxGetPr(prhs[5]);
    double alphainv  = mxGetScalar(prhs[6]);
    double *tau  = mxGetPr(prhs[7]);
    double gamma  = mxGetScalar(prhs[8]);
    double *kappa  = mxGetPr(prhs[9]);
    double *param = mxGetPr(prhs[10]);
    double *x_out;
    double *s_out;
    double *f1_out;
    double *v1_out;
    double *q1_out;

    int jState,mState,kIter;
    long iStep;
    double temp1; 
    double temp2;
    double timeStep = param[0];
    long nTime = param[1];
    const int nStates = param[2];
    int nInputs = param[3];
    int dcmTypeB = param[5];
    int dcmTypeD = param[6];
    double *oldf1 = (double *) malloc(nStates*sizeof(double));
    double *oldv1 = (double *) malloc(nStates*sizeof(double));
    double *oldq1 = (double *) malloc(nStates*sizeof(double));
    
    plhs[0] = mxCreateDoubleMatrix(nTime, nStates, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(nTime, nStates, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(nTime, nStates, mxREAL);
    plhs[3] = mxCreateDoubleMatrix(nTime, nStates, mxREAL);
    plhs[4] = mxCreateDoubleMatrix(nTime, nStates, mxREAL);
    
    x_out = mxGetPr(plhs[0]);
    s_out = mxGetPr(plhs[1]);
    f1_out = mxGetPr(plhs[2]);
    v1_out = mxGetPr(plhs[3]);
    q1_out = mxGetPr(plhs[4]);    
    
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
            s_out[iStep+1 + nTime*jState] = s_out[iStep + nTime*jState] + timeStep*(x_out[iStep + nTime*jState] -
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
            oldq1[jState]= oldq1[jState] + timeStep*((f1_out[iStep + nTime*jState]*temp2 - 
                    temp1*q1_out[iStep + nTime*jState]/v1_out[iStep + nTime*jState])/(tau[jState]*q1_out[iStep + nTime*jState]));

            /* tracking the exponentiated values */
            f1_out[iStep+1 + nTime*jState] = exp(oldf1[jState]);
            v1_out[iStep+1 + nTime*jState] = exp(oldv1[jState]);
            q1_out[iStep+1 + nTime*jState] = exp(oldq1[jState]);
        }       
    }
    
    /* free all the created memory*/
    free(oldf1);
    free(oldv1);
    free(oldq1);
    return;    
}

