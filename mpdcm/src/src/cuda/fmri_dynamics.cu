//
// Author: Eduardo Aponte
// Copyright 2015 by Eduardo Aponte <aponteeduardo@gmail.com>
//
// Licensed under GNU General Public License 3.0 or later.
// Some rights reserved. See COPYING, AUTHORS.
//
// Revision log:
//


#include "mpdcm.hcu"
#include "fmri_dynamics.hcu"

// General functions

__device__
MPFLOAT
fmri_dx(dbuff *x, dbuff *y, dbuff *u, ThetaFmri *theta, PThetaFmri *ptheta, 
    int i)
{
    MPFLOAT dx = 0;
    MPFLOAT bt = 0;
    int nx = x->nr;
    int j;
    int k;
    int o;

    o = INDEX_X * x->nr;

    // A

    for (j = 0; j < nx; j++)
    {
        bt = 0;
        if ( theta->fD == MF_TRUE )
        {
            int ol = (nx + 1) * j;
            int oj = theta->sD->j[ol + i]; 
            for (k = 0; k < theta->sD->j[ol + i + 1] - oj;  k++)
                bt = fma(x->arr[o + theta->sD->i[oj + k]], 
                    theta->sD->v[oj + k], 
                    bt);
        }
        dx = fma(theta->A[i * nx + j] + bt, x->arr[o + j], dx);
    }

    for (j = 0; j < u->nr; j++)
    {
        int ol = (nx + 1) * j;
        int oj = theta->sB->j[ol + i];

        if (  u->arr[j] == 0  )
            continue;

        
         
        bt = 0;
        for (k = 0; k < theta->sB->j[ol + i + 1] - oj;  k++)
            bt = fma(x->arr[o + theta->sB->i[oj + k]], 
                    theta->sB->v[oj + k], 
                    bt);
        
        // C

        dx = fma(theta->C[i * u->nr + j] + bt, u->arr[j], dx);

      }
     
    return dx;
}


__device__
MPFLOAT
fmri_A(dbuff *x, dbuff *y, dbuff *u, ThetaFmri *theta, PThetaFmri *ptheta, 
    int i)
{
    MPFLOAT dx = 0;
    int nx = x->nr;
    MPFLOAT *vx = x->arr + INDEX_X * x->nr;
    MPFLOAT *A = theta->A + i * nx;
    int j;

    for (j = 0; j < nx; j++)
        dx = fma(A[j], vx[j], dx);

    return dx;
}


__device__
MPFLOAT
fmri_fB(dbuff *x, dbuff *y, dbuff *u, ThetaFmri *theta, PThetaFmri *ptheta, 
    int i, int us, int ue)
{
    MPFLOAT dx = 0;
    MPFLOAT bt = 0;
    unsigned long int nx = x->nr;
    unsigned long int j;
    unsigned long int k;
    unsigned long int o;

    o = INDEX_X * x->nr;

    for (j = us; j < ue; j++)
    {
        unsigned long int ol = nx * j;
        unsigned long int oj = theta->sB->j[ol + i];

        bt = 0;
        for (k = oj; k < theta->sB->j[ol + i + 1];  k++)
            bt = fma(x->arr[o + theta->sB->i[k]], 
                    theta->sB->v[k], bt);

        dx = fma(bt, u->arr[j], dx);        
      }

    return dx;
}


__device__
MPFLOAT
fmri_B(dbuff *x, dbuff *y, dbuff *u, ThetaFmri *theta, PThetaFmri *ptheta, 
    int i)
{
    MPFLOAT dx = 0;
    MPFLOAT bt = 0;
    int nx = x->nr;
    int j;
    int k;
    int o;

    o = INDEX_X * x->nr;

    for (j = 0; j < u->nr; j++)
    {
        int ol = (nx + 1) * j;
        int oj = theta->sB->j[ol + i];

        if (  u->arr[j] == 0  )
            continue;

        bt = 0;
        for (k = 0; k < theta->sB->j[ol + i + 1] - oj;  k++)
            bt = fma(x->arr[o + theta->sB->i[oj + k]], 
                    theta->sB->v[oj + k], 
                    bt);

        dx = fma(bt, u->arr[j], dx);
        
      }

    return dx;
}



__device__
MPFLOAT
fmri_C(dbuff *x, dbuff *y, dbuff *u, ThetaFmri *theta, PThetaFmri *ptheta, 
    int i)
{
    MPFLOAT dx = 0;
    MPFLOAT *C = theta->C + i * u->nr;
    MPFLOAT *vu = u->arr;
    int j;

    for (j=0; j < u->nr; j++)
    {
        dx = fma(C[j], vu[j], dx);
    }

    return dx;
}

__device__
MPFLOAT
fmri_fD(dbuff *x, dbuff *y, dbuff *u, ThetaFmri *theta, PThetaFmri *ptheta, 
    int i, int us, int ue)
{
    MPFLOAT dx = 0;
    MPFLOAT bt = 0;
    unsigned long int nx = x->nr;
    unsigned long int j;
    unsigned long int k;
    unsigned long int o;

    o = INDEX_X * x->nr;

    for (j = us; j < ue; j++)
    {
        unsigned long int ol = nx * j;
        unsigned long int oj = theta->sD->j[ol + i];

        bt = 0;
        for (k = oj; k < theta->sD->j[ol + i + 1];  k++)
            bt = fma(x->arr[o + theta->sD->i[k]], 
                    theta->sD->v[k], bt);
            //bt += x->arr[o + theta->sB->i[k]] * theta->sB->v[k];

        dx = fma(bt, x->arr[j], dx);        
      }

    return dx;
}


__device__
MPFLOAT
fmri_D(dbuff *x, dbuff *y, dbuff *u, ThetaFmri *theta, PThetaFmri *ptheta, 
    int i)
{
    MPFLOAT dx = 0;
    MPFLOAT bt = 0;
    int nx = x->nr;
    int j;
    int k;
    int o;

    o = INDEX_X * x->nr;

    for (j = 0; j < nx; j++)
    {
        bt = 0;
        if ( theta->fD == MF_TRUE )
        {
            int ol = (nx + 1) * j;
            int oj = theta->sD->j[ol + i]; 
            for (k = 0; k < theta->sD->j[ol + i + 1] - oj;  k++)
                bt = fma(x->arr[o + theta->sD->i[oj + k]], 
                    theta->sD->v[oj + k], 
                    bt);
        }
        dx = fma(bt, x->arr[o + j], dx);
    }

    return dx;
}

__device__
int
fmri_dL(dbuff *x, dbuff *y, dbuff *u, ThetaFmri *theta, PThetaFmri *ptheta, 
    int j, MPFLOAT *dxacc)

{
    
    int toffset = threadIdx.x * DIM_FMRI_X;
    int maxx = threadIdx.x - y->nr * (blockDim.x / y->nr);
    int unr = u->nr;

    if ( maxx <= 0 )
    {
        switch ( threadIdx.y )
        {
            case INDEX_X:
                dxacc[toffset] = fmri_A(x, y, u, theta, ptheta, j);
                break;
            case INDEX_F:
                dxacc[toffset + 1] = fmri_fB(x, y, u, theta, ptheta, j,
                    0, (unr + 2)/3);
                break;
            case INDEX_S:
                dxacc[toffset + 2] = fmri_fB(x, y, u, theta, ptheta, j,
                    (unr + 2)/3, 2 * ((unr + 2)/3));
                break;
            case INDEX_V:
                dxacc[toffset + 3] = fmri_fB(x, y, u, theta, ptheta, j,
                    2 * ((unr + 2)/3), unr);
                break;
            case INDEX_Q:
                    dxacc[toffset + 4] = fmri_C(x, y, u, theta, ptheta, j);
                break;
        }
    }
    return 0;
}

__device__
int
fmri_dNonlinear(dbuff *x, dbuff *y, dbuff *u, ThetaFmri *theta, 
    PThetaFmri *ptheta, int j, MPFLOAT *dxacc)
{

    int toffset = threadIdx.x * DIM_FMRI_X;
    int maxx = threadIdx.x - y->nr * (blockDim.x / y->nr);
    int unr = u->nr;
    int unx = y->nr;

    if ( maxx <= 0 )
    {
        switch ( threadIdx.y )
        {
            case INDEX_X:
                dxacc[toffset] = fmri_A(x, y, u, theta, ptheta, j); 
                dxacc[toffset] += fmri_C(x, y, u, theta, ptheta, j);
                break;
            case INDEX_F:
                dxacc[toffset + 1] = fmri_fB(x, y, u, theta, ptheta, j,
                    0, unr/2);
                break;
            case INDEX_S:
                dxacc[toffset + 2] = fmri_fB(x, y, u, theta, ptheta, j,
                unr/2, unr);
            break;
            case INDEX_V:
                dxacc[toffset + 3] = fmri_fD(x, y, u, theta, ptheta, j,
                    0, unx/2);
                break;
            case INDEX_Q:
                    dxacc[toffset + 4] = fmri_fD(x, y, u, theta, ptheta, j,
                    unx/2, unx);
                break;
    }
}

return 0;
}


__device__ 
MPFLOAT 
fmri_ds(dbuff *x, dbuff *y, dbuff *u, ThetaFmri *theta, PThetaFmri *ptheta, 
    int i)
{
    MPFLOAT ds;

    ds = x->arr[INDEX_X * x->nr + i] - 
        theta->K[i] * x->arr[INDEX_S * x->nr + i] -
        theta->gamma * (exp(x->arr[INDEX_F * x->nr + i]) - 1);

    return ds;
}

__device__
MPFLOAT
fmri_df(dbuff *x, dbuff *y, dbuff *u, ThetaFmri *theta,  PThetaFmri *ptheta, 
    int i)
{
    MPFLOAT df;

    //ThetaDCM *theta = (ThetaDCM *) p_theta;
    //PThetaDCM *ptheta = (PThetaDCM  *) p_ptheta;

    df = x->arr[INDEX_S * x->nr + i] * exp(-x->arr[INDEX_F * x->nr + i]);

    return df;
}

__device__
MPFLOAT 
fmri_dv(dbuff *x, dbuff *y, dbuff *u, ThetaFmri *theta, PThetaFmri *ptheta, 
    int i)
{
    MPFLOAT dv;

    dv = exp(x->arr[INDEX_F * x->nr + i] - x->arr[INDEX_V * x->nr + i] - 
            theta->tau[i]) -
        exp(x->arr[INDEX_V * x->nr + i] * theta->alpha - theta->tau[i]);

    return dv;
}

__device__
MPFLOAT
fmri_dq(dbuff *x, dbuff *y, dbuff *u, ThetaFmri *theta, PThetaFmri *ptheta,
    int i)
{
    MPFLOAT dq = 0; 
    MPFLOAT f = exp(-x->arr[INDEX_F * x->nr + i]);
    MPFLOAT v;
    MPFLOAT lnE0; 
    
    v = exp(x->arr[INDEX_V * x->nr + i] * theta->alpha - theta->tau[i]);
    lnE0 = theta->lnE0;
    
    dq = (1 - exp(f * theta->ln1_E0)) * exp(x->arr[INDEX_F * x->nr + i] -
        lnE0 - theta->tau[i] -  x->arr[INDEX_Q * x->nr + i]) - v;
        
    return dq;
}

__device__ 
MPFLOAT
fmri_lk1(dbuff *x, dbuff *y, dbuff *u, ThetaFmri *theta, PThetaFmri *ptheta,
    int i)
{
    MPFLOAT l;
    MPFLOAT q = exp(x->arr[INDEX_Q * x->nr + i]);

    l = theta->k1 * ( 1 - q );
    
    return l;
}


__device__
MPFLOAT
fmri_lk2(dbuff *x, dbuff *y, dbuff *u, ThetaFmri *theta, PThetaFmri *ptheta, 
    int i)
{
    MPFLOAT l;
    MPFLOAT qv = exp(x->arr[INDEX_Q * x->nr + i] - x->arr[INDEX_V *x->nr + i]);

    l = theta->k2 * ( 1 - qv);

    return l;
}

__device__
MPFLOAT
fmri_lk3(dbuff *x, dbuff *y, dbuff *u, ThetaFmri *theta, PThetaFmri *ptheta,
    int i)
{
    MPFLOAT l;
    MPFLOAT v = exp(x->arr[INDEX_V * x->nr + i]);

    l = theta->k3 * ( 1 - v);

    return l;
}

__device__
void
fmri_upy(dbuff *nx, dbuff *y, dbuff *u, ThetaFmri *theta, PThetaFmri *ptheta,
    dbuff *ox)
{
    // Network node
    int j = threadIdx.x%y->nr;
    // V0 has been already multiplied with k1, k2, and k3

    switch( threadIdx.y )
    {
        case INDEX_LK1 :
            ox->arr[ INDEX_LK1 * ox->nr + j] = 
                fmri_lk1(nx, y, u, theta, ptheta, j);
            break;
        case INDEX_LK2:
            ox->arr[ INDEX_LK2 * ox->nr + j] =
                fmri_lk2(nx, y, u, theta, ptheta, j);
            break;
        case INDEX_LK3 :
            ox->arr[ INDEX_LK3 * ox->nr + j] =
                fmri_lk3(nx, y, u, theta, ptheta, j);
            break;
    }

}

__device__
int
fmri_store_y(dbuff *nx, dbuff *y, dbuff *u, ThetaFmri *theta, PThetaFmri *ptheta)
{
    int j = threadIdx.x % y->nr;

    if ( threadIdx.y == 0 )
        // V0 has been already multiplied with k1, k2, and k3
        y->arr[j] = 
            (nx->arr[INDEX_LK1 * y->nr + j] +
            nx->arr[ INDEX_LK2 * y->nr + j] +
            nx->arr[ INDEX_LK3 * y->nr + j]);

    return 0;
}
 
