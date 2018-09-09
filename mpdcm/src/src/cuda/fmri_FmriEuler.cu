/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */


#include "mpdcm.hcu"
#include "fmri_FmriCuda.hcu"
#include "fmri_FmriEuler.hcu"
#include "fmri_dynamics.hcu"

__device__
FmriEuler::FmriEuler() : FmriCuda(DIM_FMRI_X, PRELOC_SIZE_FMRI_EULER_X)
{

}

// The memory layout is the following:

// If it is a 4 region DCM if work the following way:
// x_1, x_2, x_3, x_4, f_1, f_2, f_3, f_4, ....

__device__
int FmriEuler::update_x(dbuff *ox, dbuff *y, dbuff *u, dbuff *nx)
{

    int j = threadIdx.x % y->nc;
    int s;
    int maxx = threadIdx.x - y->nc * ( blockDim.x / y->nc );


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
                s = INDEX_X * ox->nc + j;
                nx->arr[s] = ox->arr[s] +  
                    ptheta->de * fmri_dx(ox, y, u, theta, ptheta, j);
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


