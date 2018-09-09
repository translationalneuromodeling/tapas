    /* aponteeduardo@gmail.com */
    /* copyright (C) 2015 */

#include "mpdcm.hcu"
#include "fmri_FmriCuda.hcu"
#include "fmri_dynamics.hcu"
#include "fmri_FmriStandardRK4Nonlinear.hcu"

__device__
FmriStandardRK4Nonlinear::FmriStandardRK4Nonlinear() : FmriCuda(DIM_FMRI_X, 4)
{

}

__device__
int FmriStandardRK4Nonlinear::update_x(dbuff *ox, dbuff *y, dbuff *u, dbuff *nx)
{

    int j = threadIdx.x % y->nr;
    int s;
    int maxx = threadIdx.x - y->nr * (blockDim.x / y->nr);
    MPFLOAT de = ptheta->de;

    // Buffers for the intermediate results. z is the estimated error.
    dbuff k1[1];
    dbuff k2[1];

    k1->arr = (ox->arr < nx->arr) ? ox->arr : nx->arr;
    k1->arr += ox->nr * num_eqs * 2;  
    k1->nr = ox->nr;

    k2->arr = (ox->arr < nx->arr) ? ox->arr : nx->arr;
    k2->arr += ox->nr * num_eqs * 3; 
    k2->nr = ox->nr;

    __shared__ MPFLOAT mu[100];
    __shared__ MPFLOAT dxacc[NUM_THREADS * DIM_FMRI_X];

    int toffset = threadIdx.x * DIM_FMRI_X;
    
    dbuff tu[1];

    tu->nc = u->nc;
    tu->nr = u->nr;
    tu->arr = mu + (threadIdx.x / y->nr) * u->nr;

    if ( threadIdx.y == 0 && maxx <= 0 )
    {   
        int k = j;
        while ( k < u->nr )
        {
            tu->arr[k] = u->arr[k];
            k += y->nr;
        }
    }
    __syncthreads();

    // Make the values to be closer in range
 
    if ( isnan( *tu->arr ) ){
        if ( threadIdx.y == 0 && maxx < 0 )
        {
            nx->arr[ INDEX_X * ox->nr + j] = NAN;
            nx->arr[ INDEX_F * ox->nr + j] = NAN;
            nx->arr[ INDEX_S * ox->nr + j] = NAN;
            nx->arr[ INDEX_V * ox->nr + j] = NAN;
            nx->arr[ INDEX_Q * ox->nr + j] = NAN;
        }

    }

    fmri_dNonlinear(ox, y, tu, theta, ptheta, j, dxacc);
    __syncthreads();

    // Follow Blum

    if ( maxx < 0 )
    {
        switch ( threadIdx.y )
        {
            case INDEX_X:
                s = INDEX_X * ox->nr + j;
                k1->arr[s] = de *
                    (dxacc[toffset] + dxacc[toffset + 1] +
                    dxacc[toffset + 2] + dxacc[toffset + 3] +
                    dxacc[toffset + 4]);
                break;
            case INDEX_F:
                s = INDEX_F * ox->nr + j;
                k1->arr[s] = de * 
                    fmri_df(ox, y, tu, theta, ptheta, j);
                break;
            case INDEX_S:
                s = INDEX_S * ox->nr + j;
                k1->arr[s] = de * 
                    fmri_ds(ox, y, tu, theta, ptheta, j);
                break;
            case INDEX_V:
                s = INDEX_V * ox->nr + j;
                k1->arr[s] = de * 
                    fmri_dv(ox, y, tu, theta, ptheta, j);
                break;
            case INDEX_Q:
                s = INDEX_Q * ox->nr + j;
                k1->arr[s] = de * 
                    fmri_dq(ox, y, tu, theta, ptheta, j);
                break;
        }
        nx->arr[s] = ox->arr[s]; 
    }

    __syncthreads();

    if ( maxx < 0 )
    {
        nx->arr[s] += k1->arr[s] * 0.16666666666666;
        k2->arr[s] = ox->arr[s] + 0.5 * k1->arr[s];
    }

    if ( threadIdx.y == 0 && maxx <= 0 )
    {   
        int k = j;
        while ( k < u->nr )
        {
            tu->arr[k] = fma(u->arr[k + tu->nr], (MPFLOAT ) 0.5, 
                    tu->arr[k] * (MPFLOAT )0.5);
            k += y->nr;
        }
    }
    __syncthreads();

    fmri_dNonlinear(k2, y, tu, theta, ptheta, j, dxacc);

    __syncthreads();


    if ( maxx < 0 )
    {
        switch ( threadIdx.y )
        {
            case INDEX_X:
                k1->arr[s] = de *
                    (dxacc[toffset] + dxacc[toffset + 1] +
                    dxacc[toffset + 2] + dxacc[toffset + 3] +
                    dxacc[toffset + 4]);
                break;
            case INDEX_F:
                k1->arr[s] = de * fmri_df(k2, y, tu, theta, ptheta, j);
                break;
            case INDEX_S:
                k1->arr[s] = de * fmri_ds(k2, y, tu, theta, ptheta, j);
                break;
            case INDEX_V:
                k1->arr[s] = de * fmri_dv(k2, y, tu, theta, ptheta, j);
                break;
            case INDEX_Q:
                k1->arr[s] = de * fmri_dq(k2, y, tu, theta, ptheta, j); 
                break;
        }
    }


    __syncthreads();

    if ( maxx < 0 )
    {
        nx->arr[s] += 0.33333333333 * k1->arr[s];
        k2->arr[s] = ox->arr[s] + 0.5 * k1->arr[s];
    }

    __syncthreads();

    fmri_dNonlinear(k2, y, tu, theta, ptheta, j, dxacc);

    __syncthreads();


    if ( maxx < 0 )
    {
        switch ( threadIdx.y )
        {
            case INDEX_X:
                k1->arr[s] = de *
                    (dxacc[toffset] + dxacc[toffset + 1] +
                    dxacc[toffset + 2] + dxacc[toffset + 3] +
                    dxacc[toffset + 4]);
                break;
            case INDEX_F:
                k1->arr[s] = de * 
                    fmri_df(k2, y, tu, theta, ptheta, j);
                break;
            case INDEX_S:
                k1->arr[s] = de * 
                    fmri_ds(k2, y, tu, theta, ptheta, j);
                break;
            case INDEX_V:
                k1->arr[s] = de * 
                    fmri_dv(k2, y, tu, theta, ptheta, j);
                break;
            case INDEX_Q:
                k1->arr[s] = de * 
                    fmri_dq(k2, y, tu, theta, ptheta, j);
                break;
        }
    }

    __syncthreads();

    if ( maxx < 0 )
    {
        nx->arr[s] += 0.3333333333 * k1->arr[s];
        k2->arr[s] = ox->arr[s] + k1->arr[s];
    }


    if ( threadIdx.y == 0 && maxx <= 0 )
    {   
        int k = j;
        while ( k < u->nr )
        {
            tu->arr[k] = u->arr[k + tu->nr];
            k += y->nr;
        }
    }

    __syncthreads();
    fmri_dNonlinear(k2, y, tu, theta, ptheta, j, dxacc);
    __syncthreads();


    if ( maxx < 0 )
    {
        switch ( threadIdx.y )
        {
            case INDEX_X:
                k1->arr[s] = de *
                    (dxacc[toffset] + dxacc[toffset + 1] +
                    dxacc[toffset + 2] + dxacc[toffset + 3] +
                    dxacc[toffset + 4]); 
                break;
            case INDEX_F:
                k1->arr[s] = de * fmri_df(k2, y, tu, theta, ptheta, j);
                break;
            case INDEX_S:
                k1->arr[s] = de * fmri_ds(k2, y, tu, theta, ptheta, j);
                break;
            case INDEX_V:
                k1->arr[s] = de * fmri_dv(k2, y, tu, theta, ptheta, j);
                break;
            case INDEX_Q:
                k1->arr[s] = de * fmri_dq(k2, y, tu, theta, ptheta, j); 
                break;
        }
    }

    __syncthreads();

    if ( maxx < 0 )
        nx->arr[s] += k1->arr[s] * 0.1666666666666666;
    __syncthreads();
    
    return 0;
}


