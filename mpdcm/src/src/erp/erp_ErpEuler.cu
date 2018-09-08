/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */

#include "erp_ErpEuler.hcu"

__device__
int ErpEuler::update_x(dbuff *ox, dbuff *y, dbuff *u, dbuff *nx)
{
    int t0, t1; // State
    int nc = y->nc;
    int j = threadIdx.x % nc; // Network
    int maxx = threadIdx.x - nc * (blockDim.x / nc);
    __shared__ MPFLOAT s0[32];
    __shared__ MPFLOAT s1[32];
    __shared__ MPFLOAT s7[32];

    MPFLOAT dx;
    MPFLOAT h = ptheta->de;

    // Make the values to be closer in range
 
    if ( isnan( *u->arr ) ){
        if ( threadIdx.y == 0 && maxx < 0 )
        {
            nx->arr[ 0 + nc * j] = NAN;
            nx->arr[ 1 + nc * j] = NAN;
            nx->arr[ 2 + nc * j] = NAN;
            nx->arr[ 3 + nc * j] = NAN;
            nx->arr[ 4 + nc * j] = NAN;
            nx->arr[ 5 + nc * j] = NAN;
            nx->arr[ 6 + nc * j] = NAN;
            nx->arr[ 7 + nc * j] = NAN;
            nx->arr[ 8 + nc * j] = NAN;
        }
    }

    
    // Start precomputing the nonlinearities

    
    if ( maxx < 0 )
    {
        switch ( threadIdx.y )
        {
            case INDEX_S1:
                s1[threadIdx.x] = erp_saturation(ox->arr[ERP_DIM_X * j + 1], theta);
                break;
            case INDEX_S7:
                s7[threadIdx.x] = erp_saturation(ox->arr[ERP_DIM_X * j + 7], theta);
                break;
            case INDEX_S0:
                s0[threadIdx.x] = erp_saturation(ox->arr[ERP_DIM_X * j], theta);
                break;
        }
    }

    __syncthreads();

    // Memory layout: x1, x2, x3, --- x9, x1, x2, x3, ... c9

    if ( maxx < 0 )
    {
        switch ( threadIdx.y )
        {
            case INDEX_X78:
                t0 = 7 + ERP_DIM_X * j;
                t1 = 8 + ERP_DIM_X * j;

                nx->arr[t0] = ox->arr[t0] + h * ox->arr[t1];

                dx = erp_dx8(ox, s1, s7, s0, u, theta, j);
                nx->arr[t1] = ox->arr[t1] + h * dx;
                break;
            case INDEX_X14:
                t0 = 1 + ERP_DIM_X * j;
                t1 = 4 + ERP_DIM_X * j;
                
                nx->arr[t0] = ox->arr[t0] + h * ox->arr[t1];
                dx = erp_dx4(ox, s1, s7, s0, u, theta, j);
                nx->arr[t1] = ox->arr[t1] + h * dx;

                // x 0

                nx->arr[ERP_DIM_X * j] = ox->arr[ERP_DIM_X * j] + 
                    h * (ox->arr[5 + ERP_DIM_X * j] - ox->arr[6 + ERP_DIM_X * j]);
                break;
            case INDEX_X52:
                t0 = 2 + ERP_DIM_X * j;
                t1 = 5 + ERP_DIM_X * j;

                nx->arr[t0] = ox->arr[t0] + h * ox->arr[t1];
                dx = erp_dx5(ox, s1, s7, s0, u, theta, j);
                nx->arr[t1] = ox->arr[t1] + h * dx;

                // Because the nonlinearity is precomputed this step is 
                // very quick
                t0 = 3 + ERP_DIM_X * j;
                t1 = 6 + ERP_DIM_X * j;

                nx->arr[t0] = ox->arr[t0] + h * ox->arr[t1];
                dx = erp_dx6(ox, s1, s7, s0, u, theta, j);
                nx->arr[t1] = ox->arr[t1] + h * dx;
            break;
        }
    }


    return 0;
}


