/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */


#include "mpdcm.hcu"
#include "fmri_FmriCuda.hcu"
#include "fmri_FmriEulerBilinear.hcu"
#include "fmri_dynamics.hcu"

__device__
FmriEulerBilinear::FmriEulerBilinear() : FmriCuda(DIM_FMRI_X, PRELOC_SIZE_FMRI_EULER_X)
{

}

// The memory layout is the following:

// If it is a 4 region DCM if work the following way:
// x_1, x_2, x_3, x_4, f_1, f_2, f_3, f_4, ....

__device__
int FmriEulerBilinear::update_x(dbuff *ox, dbuff *y, dbuff *u, dbuff *nx)
{

    int j = threadIdx.x % y->nc;
    int s;
    int maxx = threadIdx.x - y->nc * ( blockDim.x / y->nc );
    int toffset = threadIdx.x * DIM_FMRI_X; 
    int unc = u->nc;
    __shared__ MPFLOAT dxacc[NUM_THREADS * DIM_FMRI_X];

    __syncthreads();

    // Make the values to be closer in range
 
    if ( isnan( *( u->arr ) ) ){
        if ( threadIdx.y == 0 && maxx < 0 )
        {
            nx->arr[ INDEX_X * ox->nc + j] = NAN;
            nx->arr[ INDEX_F * ox->nc + j] = NAN;
            nx->arr[ INDEX_S * ox->nc + j] = NAN;
            nx->arr[ INDEX_V * ox->nc + j] = NAN;
            nx->arr[ INDEX_Q * ox->nc + j] = NAN;
        }
    }

    
    if ( maxx < 0 )
    {
        switch ( threadIdx.y )
        {  
            case INDEX_X:
                dxacc[toffset] = fmri_A(ox, y, u, theta, ptheta, j);
                break;
            case INDEX_F:
                dxacc[toffset + 1] = fmri_fB(ox, y, u, theta, ptheta, j,
                    0, (unc + 2)/3);
                break;
            case INDEX_S:
                dxacc[toffset + 2] = fmri_fB(ox, y, u, theta, ptheta, j,
                    (unc + 2)/3, 2 * ((unc + 2)/3));
                break;  
            case INDEX_V:
                dxacc[toffset + 3] = fmri_fB(ox, y, u, theta, ptheta, j,
                    2 * ((unc + 2)/3), unc);
                break;
            case INDEX_Q:
                dxacc[toffset + 4] = fmri_C(ox, y, u, theta, ptheta, j);
                break;   
        }
    }
    

    __syncthreads();

    if ( maxx < 0 )
    {
        switch ( threadIdx.y )
        {  
            case INDEX_X:
                s = INDEX_X * ox->nc + j;
                nx->arr[s] = ox->arr[s] + ptheta->de * 
                    (dxacc[toffset] + dxacc[toffset + 1] + 
                     dxacc[toffset + 2] + dxacc[toffset + 3] +
                     dxacc[toffset + 4]);
                break;
            case INDEX_F:
                s = INDEX_F * ox->nc + j;
                nx->arr[s] = ox->arr[s] + 
                    ptheta->de * fmri_df(ox, y, u, theta, ptheta, j);
                break;
            case INDEX_S:
                s = INDEX_S * ox->nc + j;
                nx->arr[s] = ox->arr[s] + 
                    ptheta->de * fmri_ds(ox, y, u, theta, ptheta, j);
                break;   
            case INDEX_V:
                s = INDEX_V * ox->nc + j;
                nx->arr[s] = ox->arr[s] + 
                    ptheta->de * fmri_dv(ox, y, u, theta, ptheta, j);
                break; 
            case INDEX_Q:
                s = INDEX_Q * ox->nc + j;
                nx->arr[s] = ox->arr[s] +
                    ptheta->de * fmri_dq(ox, y, u, theta, ptheta, j); 
                break;  
        }
    }
    
    return 0;
}


