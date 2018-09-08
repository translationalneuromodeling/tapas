/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */

#include "mpdcm.hcu"
#include "fmri_FmriCuda.hcu"
#include "fmri_dynamics.hcu"
#include "fmri_FmriRK4Linear.hcu"

__device__
FmriRK4Linear::FmriRK4Linear() : FmriCuda(DIM_FMRI_X, PRELOC_SIZE_FMRI_RK4_X)
{

}

__device__
int FmriRK4Linear::update_x(dbuff *ox, dbuff *y, dbuff *u, dbuff *nx)
{

    int j = threadIdx.x % y->nc;
    int s;
    int maxx = threadIdx.x - y->nc * (blockDim.x / y->nc);
    dbuff tu[1];

    tu->nc = u->nc;
    tu->nr = u->nr;
    tu->arr = u->arr + tu->nc;

    // Buffers for the intermediate results. 
    dbuff k1[1];

    k1->arr = (ox->arr < nx->arr) ? ox->arr : nx->arr;
    k1->arr += ox->nc * num_eqs * 2;  
    k1->nc = ox->nc;

    __shared__ MPFLOAT mu[NUM_THREADS];
    
    tu->nc = u->nc;
    tu->nr = u->nr;
    tu->arr = mu;

    if ( threadIdx.y == 0 && threadIdx.x < u->nc )
        tu->arr[threadIdx.x] = u->arr[threadIdx.x];

    __syncthreads();

    // Make the values to be closer in range
 
    if ( isnan( *u->arr ) ){
        if ( threadIdx.y == 0 && maxx < 0 )
        {
            nx->arr[ INDEX_X * ox->nc + j] = NAN;
            nx->arr[ INDEX_F * ox->nc + j] = NAN;
            nx->arr[ INDEX_S * ox->nc + j] = NAN;
            nx->arr[ INDEX_V * ox->nc + j] = NAN;
            nx->arr[ INDEX_Q * ox->nc + j] = NAN;
        }
    }


    // Follow Blum

    if ( maxx < 0 )
    {
        switch ( threadIdx.y )
        {
            case INDEX_X:
                s = INDEX_X * ox->nc + j;
                k1->arr[s] = ptheta->de * (fmri_A(nx, y, u, theta, ptheta, j) +
                        fmri_C(nx, y, u, theta, ptheta, j));
                break;
            case INDEX_F:
                s = INDEX_F * ox->nc + j;
                k1->arr[s] = ptheta->de * fmri_df(ox, y, u, theta, ptheta, j);
                break;
            case INDEX_S:
                s = INDEX_S * ox->nc + j;
                k1->arr[s] = ptheta->de * fmri_ds(ox, y, u, theta, ptheta, j);
                break;
            case INDEX_V:
                s = INDEX_V * ox->nc + j;
                k1->arr[s] = ptheta->de * fmri_dv(ox, y, u, theta, ptheta, j);
                break;
            case INDEX_Q:
                s = INDEX_Q * ox->nc + j;
                k1->arr[s] = ptheta->de *fmri_dq(ox, y, u, theta, ptheta, j);
                break;
        }
        nx->arr[s] = ox->arr[s]; 
    }
    __syncthreads();

    if ( maxx < 0 )
    {
        nx->arr[s] += k1->arr[s] * 0.5;
        ox->arr[s] = k1->arr[s];
    }

    if ( threadIdx.y == 0 && threadIdx.x < u->nc )
        tu->arr[threadIdx.x] = 0.5 * (
                u->arr[threadIdx.x + tu->nc] + u->arr[threadIdx.x]);

    __syncthreads();

    if ( maxx < 0 )
    {
        switch ( threadIdx.y )
        {
            case INDEX_X:
                k1->arr[s] = ptheta->de * (fmri_A(nx, y, u, theta, ptheta, j) +
                        fmri_C(nx, y, u, theta, ptheta, j));
            case INDEX_F:
                k1->arr[s] = ptheta->de * fmri_df(nx, y, u, theta, ptheta, j);
                break;
            case INDEX_S:
                k1->arr[s] = ptheta->de * fmri_ds(nx, y, u, theta, ptheta, j);
                break;
            case INDEX_V:
                k1->arr[s] = ptheta->de * fmri_dv(nx, y, u, theta, ptheta, j);
                break;
            case INDEX_Q:
                k1->arr[s] = ptheta->de * fmri_dq(nx, y, u, theta, ptheta, j); 
                break;
        }
    }

    __syncthreads();

    if ( maxx < 0 )
    {
        nx->arr[s] += 0.5 * (k1->arr[s] - ox->arr[s]);
        ox->arr[s] *= 0.166666666666666666666;
    }

    __syncthreads();

    if ( maxx < 0 )
    {
        switch ( threadIdx.y )
        {
            case INDEX_X:
                k1->arr[s] = ptheta->de *
                     (fmri_A(nx, y, u, theta, ptheta, j) +
                        fmri_C(nx, y, u, theta, ptheta, j))
                    - k1->arr[s] * 0.5;
                break;
            case INDEX_F:
                k1->arr[s] = ptheta->de * fmri_df(nx, y, u, theta, ptheta, j)
                    - k1->arr[s] * 0.5;
                break;
            case INDEX_S:
                k1->arr[s] = ptheta->de * fmri_ds(nx, y, u, theta, ptheta, j)
                    - k1->arr[s] * 0.5;
                break;
            case INDEX_V:
                k1->arr[s] = ptheta->de * fmri_dv(nx, y, u, theta, ptheta, j)
                    - k1->arr[s] * 0.5;
                break;
            case INDEX_Q:
                k1->arr[s] = ptheta->de * fmri_dq(nx, y, u, theta, ptheta, j) 
                    - k1->arr[s] * 0.5;
                break;
        }
    }

    __syncthreads();

    if ( maxx < 0 )
    {
        nx->arr[s] += k1->arr[s];
        ox->arr[s] -= k1->arr[s];
    }

    if ( threadIdx.y == 0 && threadIdx.x < u->nc )
        tu->arr[threadIdx.x] = u->arr[threadIdx.x + tu->nc];

    __syncthreads();

    if ( maxx < 0 )
    {
        switch ( threadIdx.y )
        {
            case INDEX_X:
                k1->arr[s] = ptheta->de * 
                    (fmri_A(nx, y, u, theta, ptheta, j) +
                    fmri_C(nx, y, u, theta, ptheta, j))
                    + k1->arr[s] * 2;
                break;
            case INDEX_F:
                k1->arr[s] = ptheta->de * fmri_df(nx, y, tu, theta, ptheta, j)
                    + k1->arr[s] * 2;
                break;
            case INDEX_S:
                k1->arr[s] = ptheta->de * fmri_ds(nx, y, tu, theta, ptheta, j)
                    + k1->arr[s] * 2;
                break;
            case INDEX_V:
                k1->arr[s] = ptheta->de * fmri_dv(nx, y, tu, theta, ptheta, j)
                    + k1->arr[s] * 2;
                break;
            case INDEX_Q:
                k1->arr[s] = ptheta->de * fmri_dq(nx, y, tu, theta, ptheta, j) 
                    + k1->arr[s] * 2;
                break;
        }
    }

    __syncthreads();

    if ( maxx < 0 )
        nx->arr[s] += ox->arr[s] + k1->arr[s]*0.1666666666666666;
    __syncthreads();

    return 0;
}


