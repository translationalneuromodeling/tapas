/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */

#include "erp_dynamics.hcu"

__device__
MPFLOAT
erp_saturation(MPFLOAT x, ErpColumn *theta)
{
    return 1 / ( 1 + exp( -theta->r1 * ( x - theta->r2 ))) - theta->er1r2;
}


__device__
MPFLOAT
erp_dx8(dbuff *x, MPFLOAT *s1, MPFLOAT *s7, MPFLOAT *s0, dbuff *u, 
    ErpColumn *theta, int i)
{
    int k;
    int nx = x->nc;

    MPFLOAT dx = 0;

    for (k = 0; k < nx; k++)
    {  
        dx += theta->A23[k] * s0[k + i * nx];
    }

    dx += theta->gamma3 * s0[threadIdx.x];
    dx += - theta->tau_e2 * x->arr[i * ERP_DIM_X + 8]  - 
        theta->tau_es2 * x->arr[i * ERP_DIM_X + 7];

    return dx;
    
}

__device__
MPFLOAT
erp_dx4(dbuff *x, MPFLOAT *s1, MPFLOAT *s7, MPFLOAT *s0, dbuff *u, 
    ErpColumn *theta, int i)
{

    int k;
    int nx = x->nc;

    MPFLOAT dx = 0;

    for (k = 0; k < nx ; k++ )
    {
        dx += theta->A13[k] * s0[k + i * nx];
    }

    dx += theta->gamma1 * s0[threadIdx.x];
     
    dx += - theta->tau_e2 * x->arr[i * ERP_DIM_X + 4] - 
        theta->tau_es2 * x->arr[i * ERP_DIM_X + 1];
    dx += theta->Au * u->arr[0];

    return dx;

}


__device__
MPFLOAT
erp_dx5(dbuff *x, MPFLOAT *s1, MPFLOAT *s7, MPFLOAT *s0, dbuff *u, 
    ErpColumn *theta, int i)
{
    int k;
    int nx = x->nc;
    MPFLOAT dx = 0;

    for (k = 0; k < nx; k++ )
    {
        dx += theta->A23[k] * s0[k + i * nx];
    }
    
    dx += theta->gamma2 * s1[threadIdx.x];
    dx += - theta->tau_e2 * x->arr[i * ERP_DIM_X + 5]  - 
        theta->tau_es2 * x->arr[i * ERP_DIM_X + 2];

    return dx;
}

__device__
MPFLOAT
erp_dx6(dbuff *x, MPFLOAT *s1, MPFLOAT *s7, MPFLOAT *s0, dbuff *u, 
    ErpColumn *theta, int i)
{
    MPFLOAT dx = 0;

    dx += theta->gamma4 * s7[threadIdx.x];
    dx += - theta->tau_i2 * x->arr[i * ERP_DIM_X + 6]  - 
        theta->tau_is2 * x->arr[i * ERP_DIM_X + 3];

    return dx;
}
