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


// =======================================================================
// Integrators 
// =======================================================================

__device__ 
void
dcm_int_euler(dbuff x, dbuff y, dbuff u, void *p_theta,
    void *p_ptheta, int dp, unsigned int *errcode)
{
    int i;
    int j = threadIdx.x%y.dim;
    MPFLOAT *t;
    // Number of integration steps done between each data point
    int ss, dy;
    // Point where threads are not synchronized to anything
    int maxx = y.dim * (blockDim.x/y.dim);

    PThetaDCM *ptheta = (PThetaDCM *) p_ptheta;
    dbuff ox;
    dbuff nx;

    dbuff ty;
    dbuff tu;

    *errcode = 1;

    ox.dim = y.dim;
    nx.dim = y.dim;

    ox.arr = x.arr; 
    nx.arr = ox.arr + nx.dim * DIM_X;

    if ( threadIdx.x < maxx )
        memset(x.arr, 0, nx.dim * DIM_X * sizeof(MPFLOAT));

    __syncthreads();
    ty.dim = y.dim;
    tu.dim = u.dim;

    // How many samples are gonna be taken
    ss = ceil(1.0/ptheta->dt);
    dy = ceil(1.0/(ptheta->dt*ptheta->dyu));

    ty.arr = y.arr; 
    tu.arr = u.arr;

    for (i=0; i <= dp*ss; i++)
    {
        if ( threadIdx.x < maxx )
            dcm_upx_euler(ox, ty, tu, p_theta, p_ptheta, nx);
        __syncthreads();
        // Only sample every 1/ptheta->dt times
        if ( i%dy == 0 && i > 0) 
        {
            if ( threadIdx.x < maxx )
                dcm_upy(nx, ty, tu, p_theta, p_ptheta, ox);           
            __syncthreads();
            if ( threadIdx.x < maxx && threadIdx.y == 0 )
                ty.arr[j] = ox.arr[INDEX_LK1 * ox.dim + j] +
                    ox.arr[ INDEX_LK2 * ox.dim + j] +
                    ox.arr[ INDEX_LK3 * ox.dim + j];
            __syncthreads();

            ty.arr += y.dim; 
         }
        // Move one step forward
        if ( i%ss == 0 )
            tu.arr += u.dim;

        // Swap the pointers
        t = ox.arr;
        ox.arr = nx.arr;
        nx.arr = t;
    }

    *errcode = 0;
}

__device__
void
dcm_int_kr4(dbuff x, dbuff y, dbuff u, void *p_theta,
    void *p_ptheta, int dp, unsigned int *errcode)
{
    int i;
    int j = threadIdx.x%y.dim;
    MPFLOAT *t;
    // Number of integration steps done between each data point
    int ss, dy;
    // Point where threads are not synchronized to anything
    int maxx = y.dim * (blockDim.x/y.dim);

    PThetaDCM *ptheta = (PThetaDCM *) p_ptheta;
    dbuff ox;
    dbuff nx;

    dbuff ty;
    dbuff tu;

    *errcode = 1;

    ox.dim = y.dim;
    nx.dim = y.dim;

    ox.arr = x.arr; 
    nx.arr = ox.arr + nx.dim * DIM_X;

    if ( threadIdx.x < maxx )
        memset(x.arr, 0, nx.dim * DIM_X * sizeof(MPFLOAT));

    __syncthreads();
    ty.dim = y.dim;
    tu.dim = u.dim;

    // How many samples are gonna be taken
    ss = ceil(1.0/ptheta->dt);
    // Doesn't work always
    dy = ceil(1.0/(ptheta->dt*ptheta->dyu));

    ty.arr = y.arr; 
    tu.arr = u.arr;

    for (i=0; i <= dp*ss; i++)
    {
        if ( threadIdx.x < maxx )
            dcm_upx_kr4(ox, ty, tu, p_theta, p_ptheta, nx);
        __syncthreads();
        // Only sample every 1/ptheta->dt times
        if ( i%dy == 0 && i > 0 ) 
        {
            if ( threadIdx.x < maxx )
                dcm_upy(nx, ty, tu, p_theta, p_ptheta, ox);           
            __syncthreads();
            if ( threadIdx.x < maxx && threadIdx.y == 0 )
                ty.arr[j] = ox.arr[INDEX_LK1 * ox.dim + j] +
                    ox.arr[ INDEX_LK2 * ox.dim + j] +
                    ox.arr[ INDEX_LK3 * ox.dim + j];
            __syncthreads();

            ty.arr += y.dim; 
         }
        // Move one step forward
        if ( i%ss == 0 )
            tu.arr += u.dim;

        // Swap the pointers
        t = ox.arr;
        ox.arr = nx.arr;
        nx.arr = t;
    }

    *errcode = 0;
}


// Bucacki Shampinee

__device__
void
dcm_int_bs(dbuff x, dbuff y, dbuff u, void *p_theta,
    void *p_ptheta, int dp, unsigned int *errcode, thr_info tinfo)
{
    int i;
    int j = threadIdx.x%y.dim;
    MPFLOAT *t;
    MPFLOAT z;
    // Number of integration steps done between each output
    int dy;
    // Point where threads are not synchronized to anything
    int maxx = threadIdx.x - y.dim * (blockDim.x/y.dim);
    unsigned int ndt = MAXDY, odt = MAXDY;
    unsigned int dmin;

    __shared__ MPFLOAT zs[NUM_THREADS];

    PThetaDCM *ptheta = (PThetaDCM *) p_ptheta;
    dbuff ox;
    dbuff nx;

    dbuff ty;
    dbuff tu;

    *errcode = 0;

    ox.dim = y.dim;
    nx.dim = y.dim;

    ox.arr = x.arr; 
    nx.arr = ox.arr + nx.dim * DIM_X;

    if ( maxx < 0 )
        memset(x.arr, 0, nx.dim * DIM_X * sizeof(MPFLOAT));

    // Restart the errors
    if ( threadIdx.y == 0 )
        zs[threadIdx.x] = 0;

    __syncthreads();

    ty.dim = y.dim;
    tu.dim = u.dim;
    ty.arr = y.arr; 
    tu.arr = u.arr;

    // How many samples are gonna be taken

    dy = ceil(MAXDY * 1 / ptheta->dyu);
    dmin = min(dy, MAXDY);

    // SPM hack
    if ( threadIdx.x == 0 && threadIdx.y == 0 )
    ptheta->de = 2 * ptheta->dyu;
    __syncthreads();

    dcm_upx_bs0(ox, ty, tu, p_theta, p_ptheta, nx);
    __syncthreads();

    i = 0;

    while ( i <= dp * MAXDY )
    {
        dcm_upx_bs(ox, ty, tu, p_theta, p_ptheta, nx, zs, tinfo);

        __syncthreads();

        z = zs[0];

        __syncthreads();

        // Exceeded the error tolerance
        if ( z > MAXTOL && odt > MINDY )
        {
            odt >>= 1;
            if ( threadIdx.x == 0 && threadIdx.y == 0 )
                // SPM hack
                ptheta->de = 2 * ptheta->dyu * (((float ) odt)/MAXDY);   
        
            __syncthreads();
        
            continue;
        }
        
        // Below the error tolerance

        if ( z < MINTOL && odt < MAXDY )
            odt <<= 1;

        // Always sample at the right spot.
        if ( i%dmin + odt > dmin )
            ndt = dmin - i%dmin;
        else 
            ndt = odt;

        if ( threadIdx.x == 0 && threadIdx.y == 0 )
            ptheta->de = 2 * ptheta->dyu * (((float ) ndt)/MAXDY);
       
        __syncthreads();

        // Only sample every 1/ptheta->dt times
        if ( i%dy == 0  && i > 0 ) 
        {
           if ( maxx < 0 )
                dcm_upy(nx, ty, tu, p_theta, p_ptheta, ox);           
            __syncthreads();

           if ( maxx < 0 && threadIdx.y == 0 )
               ty.arr[j] = ox.arr[INDEX_LK1 * ox.dim + j] +
                   ox.arr[ INDEX_LK2 * ox.dim + j] +
                   ox.arr[ INDEX_LK3 * ox.dim + j];

           ty.arr += y.dim;
           __syncthreads(); 
        }

        if ( i%MAXDY == 0 )
            tu.arr += u.dim;

        // Swap the pointers
        t = ox.arr;
        ox.arr = nx.arr;
        nx.arr = t;

        i += ndt;
    }
    *errcode = 0;

}
