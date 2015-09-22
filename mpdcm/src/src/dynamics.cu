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

// General functions

__device__
MPFLOAT
dcm_dx(dbuff x, dbuff y, dbuff u, void *p_theta,
     void *p_ptheta, int i)
{
    MPFLOAT dx = 0;
    MPFLOAT bt = 0;
    int j;
    int k;
    int p;
    int o;

    ThetaDCM *theta = (ThetaDCM *) p_theta;
    o = INDEX_X * x.dim;

    // A
    for (j = 0; j < x.dim; j++)
    {
        dx = fma(theta->A[i + x.dim*j], x.arr[o + j], dx);
        if ( theta->fD == MF_TRUE )
        {
            bt = 0;
            k = x.dim * x.dim * j + i;
            for (p = 0; p < x.dim; p++)
                bt = fma(theta->D[k + x.dim * p], x.arr[o + p], bt);
            dx = fma(bt, x.arr[o + j], dx);
        }
    }

    for (j = 0; j < u.dim; j++)
    {
        if (  u.arr[j] == 0  )
            continue;
        // B
        bt = 0;
        k = x.dim * x.dim * j + i;
        for (p = 0; p < x.dim; p++){
            bt = fma(theta->B[k + x.dim * p], x.arr[o + p], bt);
        }
        // C
        dx = fma(theta->C[i + x.dim * j] + bt, u.arr[j], dx);
    }

    return dx;
}

__device__ 
MPFLOAT 
dcm_ds(dbuff x, dbuff y, dbuff u, void *p_theta,
    void *p_ptheta, int i)
{
    MPFLOAT ds;

    ThetaDCM *theta = (ThetaDCM *) p_theta;
    //PThetaDCM *ptheta = (PThetaDCM  *) p_ptheta;

    ds = x.arr[INDEX_X * x.dim + i] - 
        theta->K[i] * x.arr[INDEX_S * x.dim + i] -
        theta->gamma * (exp(x.arr[INDEX_F * x.dim + i]) - 1);

    return ds;
}

__device__
MPFLOAT
dcm_df(dbuff x, dbuff y, dbuff u, void *p_theta, 
    void *p_ptheta, int i)
{
    MPFLOAT df;

    //ThetaDCM *theta = (ThetaDCM *) p_theta;
    //PThetaDCM *ptheta = (PThetaDCM  *) p_ptheta;

    df = x.arr[INDEX_S * x.dim + i]*exp(-x.arr[INDEX_F * x.dim + i]);

    return df;
}

__device__
MPFLOAT 
dcm_dv(dbuff x, dbuff y, dbuff u, void *p_theta,
    void *p_ptheta, int i)
{
    MPFLOAT dv;

    ThetaDCM *theta = (ThetaDCM *) p_theta;
    //PThetaDCM *ptheta = (PThetaDCM  *) p_ptheta;

    dv = exp(x.arr[INDEX_F * x.dim + i] - x.arr[INDEX_V * x.dim + i] - 
            theta->tau[i]) -
        exp(x.arr[INDEX_V * x.dim + i] * theta->alpha - theta->tau[i]);

    return dv;
}

__device__
MPFLOAT
dcm_dq(dbuff x, dbuff y, dbuff u, void *p_theta, 
    void *p_ptheta, int i)
{
    MPFLOAT dq = 0;
    MPFLOAT f = exp(-x.arr[INDEX_F * x.dim + i]);
    MPFLOAT v;
    MPFLOAT lnE0; 
    ThetaDCM *theta = (ThetaDCM *) p_theta;
    
    v = exp(x.arr[INDEX_V * x.dim + i] * theta->alpha - theta->tau[i]);
    lnE0 = theta->lnE0;

    //    PThetaDCM *ptheta = (PThetaDCM  *) p_ptheta;

    dq = (1 - exp(f * theta->ln1_E0))*exp(x.arr[INDEX_F * x.dim + i] -
        lnE0 - theta->tau[i] -  x.arr[INDEX_Q * x.dim + i]) - v;

    return dq;
}

__device__ 
MPFLOAT
dcm_lk1(dbuff x, dbuff y, dbuff u, void *p_theta,
            void *p_ptheta, int i)
{
    MPFLOAT l;
    MPFLOAT q = exp(x.arr[INDEX_Q * x.dim + i]);

    ThetaDCM *theta = (ThetaDCM *) p_theta;
    //PThetaDCM *ptheta = (PThetaDCM *) p_ptheta;

    l = theta->k1 * ( 1 - q);

    return l;
}


__device__
MPFLOAT
dcm_lk2(dbuff x, dbuff y, dbuff u, void *p_theta,
            void *p_ptheta, int i)
{
    MPFLOAT l;
    MPFLOAT qv = exp(x.arr[INDEX_Q * x.dim + i] - x.arr[INDEX_V *x.dim + i]);

    ThetaDCM *theta = (ThetaDCM *) p_theta;
    //PThetaDCM *ptheta = (PThetaDCM *) p_ptheta;

    l = theta->k2 * ( 1 - qv);

    return l;
}

__device__
MPFLOAT
dcm_lk3(dbuff x, dbuff y, dbuff u, void *p_theta,
            void *p_ptheta, int i)
{
    MPFLOAT l;

    MPFLOAT v = exp(x.arr[INDEX_V * x.dim + i]);

    ThetaDCM *theta = (ThetaDCM *) p_theta;
    //PThetaDCM *ptheta = (PThetaDCM *) p_ptheta;

    l = theta->k3 * ( 1 - v);

    return l;
}

__device__
void
dcm_upy(dbuff ox, dbuff y, dbuff u, void *theta,
    void *ptheta, dbuff nx)
{
    // Network node
    int j = threadIdx.x%y.dim;

    switch( threadIdx.y )
    {
        case INDEX_LK1 :
            nx.arr[ INDEX_LK1 * nx.dim + j] =
                dcm_lk1(ox, y, u, theta, ptheta, j);
            break;
        case INDEX_LK2:
            nx.arr[ INDEX_LK2 * nx.dim + j] =
                dcm_lk2(ox, y, u, theta, ptheta, j);
            break;
        case INDEX_LK3 :
            nx.arr[ INDEX_LK3 * nx.dim + j] =
                dcm_lk3(ox, y, u, theta, ptheta, j);
            break;
    }

}


