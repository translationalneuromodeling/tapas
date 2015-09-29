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

# ifdef MPDOUBLEFLAG
# define MPMXFLOAT mxDOUBLE_CLASS
# else
# define MPMXFLOAT mxSINGLE_CLASS
# endif

typedef struct {
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
    MPFLOAT *epsilon;
    MPFLOAT *param;
} inputargs;

typedef struct {
    MPFLOAT *x_out;
    MPFLOAT *s_out;
    MPFLOAT *f1_out;
    MPFLOAT *v1_out;
    MPFLOAT *q1_out;
    MPFLOAT *y_out;
} outputargs;


void
integrate_system(const mxArray *ds, mxArray *rc);

void
c_integrator(inputargs *ia, outputargs *oa);

MPFLOAT
forward_model(MPFLOAT q, MPFLOAT v, MPFLOAT rho, MPFLOAT epsilon);

void 
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* Extract all input parameters*/
    
    int l;
    int na;

    const mxArray *ds = prhs[0];

    na = mxGetDimensions(ds)[0] * mxGetDimensions(ds)[1];
    plhs[0] = mxCreateCellMatrix(na, 1); 
    integrate_system(ds, plhs[0]);

}

void
integrate_system(const mxArray *ds, mxArray *rc)
{
    // Input
    // ds       -- Data set
    // rc       -- Results 

    int l;
    int na = mxGetDimensions(ds)[0] * mxGetDimensions(ds)[1]; 
    const char *fieldnames[] = {"x", "s", "f1", "v1", "q1", "y"};

    inputargs *ia = malloc(na * sizeof( inputargs ));
    outputargs *oa = malloc(na * sizeof( outputargs ));

    
    for (l = 0; l < na; l++)
    {
        long nTime = 0;
        int nStates = 0;
        mxArray *cds = mxGetCell(ds, l);
        int jState = 0, iStep = 0; 

        // Inputs

        (ia + l)->A  = (MPFLOAT *) mxGetPr(mxGetField(cds, 0, "A"));
        (ia + l)->B  = (MPFLOAT *) mxGetPr(mxGetField(cds, 0, "B"));
        (ia + l)->C  = (MPFLOAT *) mxGetPr(mxGetField(cds, 0, "C"));
        (ia + l)->D = (MPFLOAT *) mxGetPr(mxGetField(cds, 0, "D"));
        (ia + l)->U  = (MPFLOAT *) mxGetPr(mxGetField(cds, 0, "U"));
        (ia + l)->rho = (MPFLOAT *) mxGetPr(mxGetField(cds, 0, "rho"));
        (ia + l)->alphainv = (MPFLOAT ) mxGetScalar(mxGetField(cds, 0, 
            "alphainv"));
        (ia + l)->tau  = (MPFLOAT *) mxGetPr(mxGetField(cds, 0, "tau"));
        (ia + l)->gamma  = (MPFLOAT ) mxGetScalar(mxGetField(cds, 0, "gamma"));
        (ia + l)->kappa  = (MPFLOAT *) mxGetPr(mxGetField(cds, 0, "kappa"));
        (ia + l)->epsilon = (MPFLOAT *) mxGetPr(mxGetField(cds, 0, "epsilon"));
        (ia + l)->param = (MPFLOAT *) mxGetPr(mxGetField(cds, 0, "param"));
     
        nTime = (ia + l)->param[1];
        nStates = (ia + l)->param[2];
        
        mxArray *ty = mxCreateStructMatrix(1, 1, 6, fieldnames); 
        
        // Crete the arrays

        // Outputs

        mxArray *ax_out = mxCreateNumericMatrix(nStates, nTime, MPMXFLOAT,
            mxREAL);
        mxArray *as_out = mxCreateNumericMatrix(nStates, nTime, MPMXFLOAT,
            mxREAL);
        mxArray *af1_out = mxCreateNumericMatrix(nStates, nTime, MPMXFLOAT,
            mxREAL);
        mxArray *av1_out = mxCreateNumericMatrix(nStates, nTime, MPMXFLOAT,
            mxREAL);
        mxArray *aq1_out = mxCreateNumericMatrix(nStates, nTime, MPMXFLOAT,
            mxREAL);
        mxArray *ay_out = mxCreateNumericMatrix(nStates, nTime, MPMXFLOAT,
            mxREAL);
        
        (oa + l)->x_out = (MPFLOAT *) mxGetPr(ax_out);
        (oa + l)->s_out = (MPFLOAT *) mxGetPr(as_out);
        (oa + l)->f1_out = (MPFLOAT *) mxGetPr(af1_out);
        (oa + l)->v1_out = (MPFLOAT *) mxGetPr(av1_out);
        (oa + l)->q1_out = (MPFLOAT *) mxGetPr(aq1_out);
        (oa + l)->y_out = (MPFLOAT *) mxGetPr(ay_out);
        // Assign field 

        mxSetField(ty, 0, "x", ax_out);
        mxSetField(ty, 0, "s", as_out);
        mxSetField(ty, 0, "f1", af1_out);
        mxSetField(ty, 0, "v1", av1_out);
        mxSetField(ty, 0, "q1", aq1_out);
        mxSetField(ty, 0, "y", ay_out);
        
        mxSetCell(rc, l, ty);
     


    }
   

    omp_set_num_threads(omp_get_max_threads());  
    
    # pragma omp parallel for
    for ( l = 0; l < na; l++ )
    {
        c_integrator(ia + l, oa + l);
    }
    
    free(ia);
    free(oa);

   
    return; 
}

void
c_integrator(inputargs *ia, outputargs *oa)
{

    MPFLOAT *A  = ia->A; 
    MPFLOAT *B  = ia->B;
    MPFLOAT *C  = ia->C;
    MPFLOAT *D = ia->D;
    MPFLOAT *U  = ia->U;
    MPFLOAT *rho = ia->rho;
    MPFLOAT alphainv = ia->alphainv;
    MPFLOAT *tau  = ia->tau;
    MPFLOAT gamma  = ia->gamma;
    MPFLOAT *kappa  = ia->kappa;
    MPFLOAT *epsilon = ia->epsilon;
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
    MPFLOAT *y_out = oa->y_out;

    MPFLOAT *oldf1 = (MPFLOAT *) malloc(nStates * sizeof( MPFLOAT ));
    MPFLOAT *oldv1 = (MPFLOAT *) malloc(nStates * sizeof( MPFLOAT ));
    MPFLOAT *oldq1 = (MPFLOAT *) malloc(nStates * sizeof( MPFLOAT ));

    /* Initialize the dynamical system to resting state values */
    for(jState=0; jState < nStates; jState++)
    {
        x_out[jState] = 0.0;
        s_out[jState] = 0.0;
        f1_out[jState] = 1.0;
        v1_out[jState] = 1.0;
        q1_out[jState] = 1.0; 
        oldf1[jState] = 0.0;
        oldv1[jState] = 0.0;
        oldq1[jState] = 0.0;

        y_out[jState] = forward_model(q1_out[jState], v1_out[jState], 
            rho[jState], epsilon[jState]);
    }

    y_out += nStates;

    // Euler's integration steps
    
    //omp_set_num_threads((nStates < omp_get_max_threads()) ? nStates : 
    //    omp_get_max_threads());  
   
    //#pragma omp parallel private(iStep) private(jState) num_threads(3)
    for(iStep=0; iStep < (nTime - 1); iStep++)
    {       
        MPFLOAT temp1, temp2;
        int mState, kIter;

        //int jState = omp_get_thread_num();
        int jState = 0;
        // For each region
        while ( jState < nStates )
        {
            // update x
            //switch ( n%5 )
            //case 0;
            x_out[jState + nStates] = 0.0;
            temp1 = 0.0;            
            
            for(kIter=0;kIter<nStates;kIter++)
            {
                temp1 += x_out[kIter] * A[kIter + nStates*jState];
            }

            for (kIter = 0; kIter < nInputs; kIter++)
            {
                temp1 += U[iStep + nTime*kIter] * C[jState + nStates *kIter];
            }
                     
            x_out[jState + nStates] = x_out[jState] + timeStep * temp1;
        
            if( dcmTypeB != 0 )
            {
                // B matrix update
                for(kIter=0;kIter<nInputs;kIter++)
                {
                    temp1 = 0.0;
                    for(mState=0; mState < nStates; mState++)
                    {
                        temp1 += x_out[mState] * 
                            B[mState + nStates*(jState + nStates*kIter)];
                    }
                    x_out[jState + nStates] += timeStep * 
                        U[iStep +nTime*kIter] * temp1;
                }
            }
            
            if( dcmTypeD != 0 )
            {
                // D matrix update
                for( kIter=0; kIter < nStates; kIter++ )         
                {
                     temp1 = 0.0;
                     for(mState=0; mState<nStates; mState++)
                     {
                         temp1 += x_out[mState] * 
                            D[mState + nStates*(jState + nStates*kIter)];
                     }
                     x_out[jState + nStates] += timeStep * 
                        x_out[kIter] * temp1;
                }
            }
            /// update s
            s_out[jState+nStates] = s_out[jState] +
                timeStep * (x_out[jState] -
                kappa[jState]*s_out[jState] - 
                gamma*(f1_out[jState] - 1.0));

            // update f 
            oldf1[jState] += timeStep * 
                (s_out[jState]/f1_out[jState]);
            f1_out[jState+nStates] = exp(oldf1[jState]);
            
            // update v 
            
            temp1 = pow(v1_out[jState],alphainv);
            oldv1[jState] += timeStep*( (f1_out[jState] -
                 temp1) / (tau[jState]*v1_out[jState]));
            v1_out[jState+nStates] = exp(oldv1[jState]);
            
            // update q 

            temp2 = (1.0 - pow((1.0-rho[jState]), 
                (1.0/f1_out[jState])))/rho[jState];
            oldq1[jState]= oldq1[jState] + 
                timeStep * ((f1_out[jState]*temp2 - 
                    temp1*q1_out[jState] / 
                    v1_out[jState]) / 
                    (tau[jState] * q1_out[jState]));    
            q1_out[jState + nStates] = exp(oldq1[jState]);

            y_out[jState] = forward_model(q1_out[jState + nStates], 
                v1_out[jState + nStates], rho[jState], epsilon[jState]);

            // Go to the next state
            jState += 1; //omp_get_num_threads();

        }


        x_out += nStates;
        s_out += nStates;
        f1_out += nStates;
        v1_out += nStates;
        q1_out += nStates;
        y_out += nStates;
        //#pragma omp barrier 
    }

    free(oldf1);
    free(oldv1);
    free(oldq1);

}


#define V0 4.0
#define F_OFF 40.3
#define ECHO_TIME 0.04
#define RELAXATION_RATE 25.0
#define OX_EXT 0.4


MPFLOAT
forward_model(MPFLOAT q, MPFLOAT v, MPFLOAT rho, MPFLOAT epsilon)
{
    MPFLOAT coeff1 = 4.3 * F_OFF * ECHO_TIME * OX_EXT; 
    MPFLOAT coeff2 = epsilon * RELAXATION_RATE * OX_EXT * ECHO_TIME;
    MPFLOAT coeff3 = 1 - epsilon;


    return V0 * (coeff1 * (1 - q) + coeff2 * (1 - (q / v)) + coeff3 * (1 - v));
}


