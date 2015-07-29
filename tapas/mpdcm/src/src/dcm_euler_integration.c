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
#include <omp.h>

typedef{
    MPFLOAT *A;
    MPFLOAT *C;
    MPFLOAT *U;
    MPFLOAT *B;
    MPFLOAT *D;
    MPFLOAT *rho;
    MPFLOAT alphainv;
    MPFLOAT *tau;
    MPFLOAT gamma;
    MPFLOAT *kappa;
    MPFLOAT *param;
} inputargs;

typedef{
    MPFLOAT *x_out;
    MPFLOAT *s_out;
    MPFLOAT *f1_out;
    MPFLOAT *v1_out;
    MPFLOAT *q1_out;
} outargs;

void mexFunction(int nlhs, mxArray *plhs[],/*Outputvariables*/
                    int nrhs, const mxArray *prhs[])
{
    /* Extract all input parameters*/
    
    int l;
    int na;

    mxArray *id = prhs[0];

    na = mxGetDimensions(id)[0] * mxGetDimensions(id)[1];
    plhs[0] = mxCreateCellMatrix(na, 1); 
    integrate_system(ds, plhs[0]);

}

void
integrate_system(mxArray *ds, *mxArray rc)
{
    // Input
    // ds       -- Data set
    // rc       -- Results 

    int l;
    int na = mxGetDimensions(ds)[0] * mxGetDimensions(ds)[1]; 
    const char *fieldnames[] = {"x_out", "s_out", "f1_out", "v1_out",
        "q1_out"};

    inputargs *ia = malloc(na * sizeof( inputargs ));
    outputargs *oa = malloc(na * sizeof( outputargs ));

    
    for (l = 0; l < na; l++)
    {
        long nTime 0;
        const int nStates 0; 

        // Inputs

        (ia + l)->A  = (MPFLOAT *) mxGetPr(mxGetField(ds, 0, "A"));
        (ia + l)->C  = (MPFLOAT *) mxGetPr(mxGetField(ds, 0, "C"));
        (ia + l)->U  = (MPFLOAT *) mxGetPr(mxGetField(ds, 0, "U"));
        (ia + l)->B  = (MPFLOAT *) mxGetPr(mxGetField(ds, 0, "B"));
        (ia + l)->D = (MPFLOAT *) mxGetPr(mxGetField(ds, 0, "rhp"));
        (ia + l)->rho = (MPFLOAT *) mxGetPr(mxGetField(ds, 0, "rho"));
        (ia + l)->alphainv = (MPFLOAT ) mxGetScalar(mxGetField(ds, 0, 
            "alphainv"));
        (ia + l)->tau  = (MPFLOAT *) mxGetPr(mxGetField(ds, 0, "tau"));
        (ia + l)->gamma  = (MPFLOAT ) mxGetScalar(mxGetField(ds, 0, "gamma"));
        (ia + l)->kappa  = (MPFLOAT *) mxGetPr(mxGetField(ds, 0, "kappa"));
        (ia + l)->param = (MPFLOAT *) mxGetPr(mxGetField(ds, 0, "param"));
     
        nTime = (ia + l)->param[1];
        nStates = (ia + l)->param[2];

        mxArray *ty = mxCreateStructMatrix(1, 1, 5, fieldnames); 
        
        // Crete the arrays

        mxArray ax_out = mxCreateDoubleMatrix(nTime, nStates, mxREAL);
        mxArray as_out = mxCreateDoubleMatrix(nTime, nStates, mxREAL);
        mxArray af1_out = mxCreateDoubleMatrix(nTime, nStates, mxREAL);
        mxArray av1_out = mxCreateDoubleMatrix(nTime, nStates, mxREAL);
        mxArray aq1_out = mxCreateDoubleMatrix(nTime, nStates, mxREAL);

        // Outputs
        
        (oa + l)->x_out[l] = mxGetPr(ax_out);
        (oa + l)->s_out[l] = mxGetPr(ax_out);
        (oa + l)->f1_out[l] = mxGetPr(ax_out);
        (oa + l)->v1_out[l] = mxGetPr(ax_out);
        (oa + l)->q1_out[l] = mxGetPr(ax_out);

        /* Assign field */

        mxSetField(ty, 0, "x", ax_out);
        mxSetField(ty, 0, "s", as_out);
        mxSetField(ty, 0, "f1", af1_out);
        mxSetField(ty, 0, "v1", av1_out);
        mxSetField(ty, 0, "q1", aq1_out);

        mxSetCell(rc, l, ty);
    
    }


    for ( l = 0; l < na; l++ )
    {
        integrator(ia + l, oa + l);
    }

    free(inputargs);
    free(outputargs);


    return; 
}

void
integrator(inputargs *ia, outputargs *oa)
{

    MPFLOAT *A  = ia->A; 
    MPFLOAT *B  = ia->B;
    MPFLOAT *C  = ia->C;
    MPFLOAT *D = ia->D;
    MPFLOAT *U  = ia->U;
    MPFLOAT *rho = ia->rhoM
    MPFLOAT alphainv = ia->alphainv;
    MPFLOAT *tau  = ia->tau;
    MPFLOAT gamma  = ia->gamma;
    MPFLOAT *kappa  = ia->kappa;
    MPFLOAT *param = ia->param;

    

    int jState;
    long iStep;
    MPFLOAT timeStep = param[0];
    long nTime = param[1];
    const int nStates = param[2];
    int nInputs = param[3];
    int dcmTypeB = param[5];
    int dcmTypeD = param[6];

    MPFLOAT *x_out = oa->x_out;
    MPFLOAT *s_out = oa->s_out;
    MPFLOAT *f1_out = oa->f1_out;
    MPFLOAT *v1_out = oa->v1_out;
    MPFLOAT *q1_out = oa->q1_out;

    MPFLOAT *oldf1 = (MPFLOAT *) malloc(nStates * sizeof( MPFLOAT ));
    MPFLOAT *oldv1 = (MPFLOAT *) malloc(nStates * sizeof( MPFLOAT ));
    MPFLOAT *oldq1 = (MPFLOAT *) malloc(nStates * sizeof( MPFLOAT ));

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
    
    omp_set_num_threads((nStates < omp_get_max_threads()) ? nStates : 
        omp_get_max_threads());
    #pragma omp parallel private(iStep) private(jState)
    for(iStep=0; iStep < (nTime - 1); iStep++)
    {       
        MPFLOAT temp1, temp2;
        int mState, kIter;
        int jState = omp_get_thread_num();

        /* For each region*/
        while ( jState < nStates )
        {
            /* update x */
            x_out[iStep+1 + nTime*jState] = 0.0;
            temp1 = 0.0;            
            for(kIter=0;kIter<nStates;kIter++)
            {
                temp1 += x_out[iStep + nTime*kIter] * A[kIter + nStates*jState];
            }            
            x_out[iStep + 1 + nTime*jState] = x_out[iStep + nTime*jState] + 
                timeStep * (temp1 + C[iStep + nTime * jState]); 
            
            if( dcmTypeB != 0 )
            {
                /*B matrix update*/
                for(kIter=0;kIter<nInputs;kIter++)
                {
                    temp1 = 0.0;
                    for(mState=0;mState<nStates;mState++)
                    {
                        temp1 += x_out[iStep + nTime*mState] * 
                            B[mState + nStates*(jState + nStates*kIter)];
                    }
                    x_out[iStep+1 + nTime*jState] += timeStep * 
                        U[iStep +nTime*kIter] * temp1;
                }
            }
            
            if(dcmTypeD != 0)
            {
                /*D matrix update*/
                for( kIter=0; kIter < nStates; kIter++ )         
                {
                     temp1 = 0.0;
                     for(mState=0; mState<nStates; mState++)
                     {
                         temp1 += x_out[iStep + nTime*mState] * 
                            D[mState + nStates*(jState + nStates*kIter)];
                     }
                     x_out[iStep + 1 + nTime*jState] += timeStep * 
                        x_out[iStep +nTime*kIter] * temp1;
                }
            }
            
            /// update s
            s_out[iStep+1 + nTime*jState] = s_out[iStep + nTime*jState] +
                timeStep * (x_out[iStep + nTime*jState] -
                kappa[jState]*s_out[iStep + nTime*jState] - 
                gamma*(f1_out[iStep + nTime*jState] - 1.0));
            
            // update f 
            oldf1[jState] += timeStep * 
                (s_out[iStep + nTime*jState]/f1_out[iStep + nTime*jState]);
            
            // update v 
            temp1 = pow(v1_out[iStep + nTime*jState],alphainv);
            temp2 = (1.0 - pow((1.0-rho[jState]), 
                (1.0/f1_out[iStep + nTime*jState])))/rho[jState];
            oldv1[jState] += timeStep*( (f1_out[iStep + nTime*jState] - temp1) /
                (tau[jState]*v1_out[iStep + nTime*jState]));
            
            // update q 
            oldq1[jState]= oldq1[jState] + 
                timeStep * ((f1_out[iStep + nTime*jState]*temp2 - 
                    temp1*q1_out[iStep + nTime*jState] / 
                    v1_out[iStep + nTime*jState]) / 
                    (tau[jState] * q1_out[iStep + nTime*jState]));
            
            // tracking the exponentiated values
            
            f1_out[iStep+1 + nTime*jState] = exp(oldf1[jState]);
            v1_out[iStep+1 + nTime*jState] = exp(oldv1[jState]);
            q1_out[iStep+1 + nTime*jState] = exp(oldq1[jState]);

            // Go to the next state
            jState += omp_get_num_threads();
        }
        #pragma omp barrier 
    }
    
    free(oldf1);
    free(oldv1);
    free(oldq1);

}
