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

// The memory layout is the following:

// If it is a 4 region DCM if work the following way:
// x_1, x_2, x_3, x_4, f_1, f_2, f_3, f_4, ....


// Euler

__device__
void
dcm_upx_euler(dbuff ox, dbuff y, dbuff u, void *p_theta,
     void *p_ptheta, dbuff nx)
{

    //ThetaDCM *theta = (ThetaDCM *) p_theta;
    PThetaDCM *ptheta = (PThetaDCM *) p_ptheta;

    int j = threadIdx.x%y.dim;
    int s;
    int maxx = threadIdx.x - y.dim * (blockDim.x/y.dim);


    // Make the values to be closer in range
 
    if ( isnan( *u.arr ) ){
        if ( threadIdx.y == 0 && maxx < 0 )
        {
            nx.arr[ INDEX_X * ox.dim + j] = NAN;
            nx.arr[ INDEX_F * ox.dim + j] = NAN;
            nx.arr[ INDEX_S * ox.dim + j] = NAN;
            nx.arr[ INDEX_V * ox.dim + j] = NAN;
            nx.arr[ INDEX_Q * ox.dim + j] = NAN;
        }
    }
    if ( maxx < 0 )
    {
        switch ( threadIdx.y )
        {
            case INDEX_X:
                s = INDEX_X * ox.dim + j;
                nx.arr[s] = ox.arr[s] + 
                    ptheta->de * dcm_dx(ox, y, u, p_theta, p_ptheta, j);
                break;
            case INDEX_F:
                s = INDEX_F * ox.dim + j;
                nx.arr[s] = ox.arr[s] + 
                    ptheta->de * dcm_df(ox, y, u, p_theta, p_ptheta, j);
                break;
            case INDEX_S:
                s = INDEX_S * ox.dim + j;
                nx.arr[s] = ox.arr[s] + 
                    ptheta->de * dcm_ds(ox, y, u, p_theta, p_ptheta, j);
                break;
            case INDEX_V:
                s = INDEX_V * ox.dim + j;
                nx.arr[s] = ox.arr[s] + 
                    ptheta->de * dcm_dv(ox, y, u, p_theta, p_ptheta, j);
                break;
            case INDEX_Q:
                s = INDEX_Q * ox.dim + j;
                nx.arr[s] = ox.arr[s] + 
                    ptheta->de * dcm_dq(ox, y, u, p_theta, p_ptheta, j); 
                break;
        }
    }
}

// Runge Kutta

__device__
void
dcm_upx_kr4(dbuff ox, dbuff y, dbuff u, void *p_theta,
     void *p_ptheta, dbuff nx)
{

    //ThetaDCM *theta = (ThetaDCM *) p_theta;
    PThetaDCM *ptheta = (PThetaDCM *) p_ptheta;

    int j = threadIdx.x%y.dim;
    int s;
    int maxx = threadIdx.x - y.dim * (blockDim.x / y.dim);
    // Buffers for the intermediate results. z is the estimated error.
    dbuff k1;

    k1.arr = (ox.arr < nx.arr) ? ox.arr : nx.arr;
    k1.arr += ox.dim * DIM_X * 2;  
    k1.dim = ox.dim;


    // Make the values to be closer in range
 
    if ( isnan( *u.arr ) ){
        if ( threadIdx.y == 0 && maxx < 0 )
        {
            nx.arr[ INDEX_X * ox.dim + j] = NAN;
            nx.arr[ INDEX_F * ox.dim + j] = NAN;
            nx.arr[ INDEX_S * ox.dim + j] = NAN;
            nx.arr[ INDEX_V * ox.dim + j] = NAN;
            nx.arr[ INDEX_Q * ox.dim + j] = NAN;
        }
    }

    // Follow Blum

    if ( maxx < 0 )
    {
        switch ( threadIdx.y )
        {
            case INDEX_X:
                s = INDEX_X * ox.dim + j;
                k1.arr[s] = ptheta->de * dcm_dx(ox, y, u, p_theta, p_ptheta, j);
                break;
            case INDEX_F:
                s = INDEX_F * ox.dim + j;
                k1.arr[s] = ptheta->de * dcm_df(ox, y, u, p_theta, p_ptheta, j);
                break;
            case INDEX_S:
                s = INDEX_S * ox.dim + j;
                k1.arr[s] = ptheta->de * dcm_ds(ox, y, u, p_theta, p_ptheta, j);
                break;
            case INDEX_V:
                s = INDEX_V * ox.dim + j;
                k1.arr[s] = ptheta->de * dcm_dv(ox, y, u, p_theta, p_ptheta, j);
                break;
            case INDEX_Q:
                s = INDEX_Q * ox.dim + j;
                k1.arr[s] = ptheta->de *dcm_dq(ox, y, u, p_theta, p_ptheta, j);
                break;
        }
        nx.arr[s] = ox.arr[s]; 
    }
    __syncthreads();

    if ( maxx < 0 )
    {
        nx.arr[s] += k1.arr[s]*0.5;
        ox.arr[s] = k1.arr[s];
    }

    __syncthreads();

    if ( maxx < 0 )
    {
        switch ( threadIdx.y )
        {
            case INDEX_X:
                k1.arr[s] = ptheta->de*dcm_dx(nx, y, u, p_theta, p_ptheta, j);
                break;
            case INDEX_F:
                k1.arr[s] = ptheta->de*dcm_df(nx, y, u, p_theta, p_ptheta, j);
                break;
            case INDEX_S:
                k1.arr[s] = ptheta->de*dcm_ds(nx, y, u, p_theta, p_ptheta, j);
                break;
            case INDEX_V:
                k1.arr[s] = ptheta->de*dcm_dv(nx, y, u, p_theta, p_ptheta, j);
                break;
            case INDEX_Q:
                k1.arr[s] = ptheta->de*dcm_dq(nx, y, u, p_theta, p_ptheta, j); 
                break;
        }
    }

    __syncthreads();

    if ( maxx < 0 )
    {
        nx.arr[s] += 0.5 * (k1.arr[s] - ox.arr[s]);
        ox.arr[s] *= 0.166666666666666666666;
    }

    __syncthreads();

    if ( maxx < 0 )
    {
        switch ( threadIdx.y )
        {
            case INDEX_X:
                k1.arr[s] = ptheta->de*dcm_dx(nx, y, u, p_theta, p_ptheta, j) 
                    - k1.arr[s] * 0.5;
                break;
            case INDEX_F:
                k1.arr[s] = ptheta->de*dcm_df(nx, y, u, p_theta, p_ptheta, j)
                    - k1.arr[s] * 0.5;
                break;
            case INDEX_S:
                k1.arr[s] = ptheta->de*dcm_ds(nx, y, u, p_theta, p_ptheta, j)
                    - k1.arr[s] * 0.5;
                break;
            case INDEX_V:
                k1.arr[s] = ptheta->de*dcm_dv(nx, y, u, p_theta, p_ptheta, j)
                    - k1.arr[s] * 0.5;
                break;
            case INDEX_Q:
                k1.arr[s] = ptheta->de*dcm_dq(nx, y, u, p_theta, p_ptheta, j) 
                    - k1.arr[s] * 0.5;
                break;
        }
    }

    __syncthreads();

    if ( maxx < 0 )
    {
        nx.arr[s] += k1.arr[s];
        ox.arr[s] -= k1.arr[s];
    }

    __syncthreads();

    if ( maxx < 0 )
    {
        switch ( threadIdx.y )
        {
            case INDEX_X:
                k1.arr[s] = ptheta->de*dcm_dx(nx, y, u, p_theta, p_ptheta, j) 
                    + k1.arr[s] * 2;
                break;
            case INDEX_F:
                k1.arr[s] = ptheta->de*dcm_df(nx, y, u, p_theta, p_ptheta, j)
                    + k1.arr[s] * 2;
                break;
            case INDEX_S:
                k1.arr[s] = ptheta->de*dcm_ds(nx, y, u, p_theta, p_ptheta, j)
                    + k1.arr[s] * 2;
                break;
            case INDEX_V:
                k1.arr[s] = ptheta->de*dcm_dv(nx, y, u, p_theta, p_ptheta, j)
                    + k1.arr[s] * 2;
                break;
            case INDEX_Q:
                k1.arr[s] = ptheta->de*dcm_dq(nx, y, u, p_theta, p_ptheta, j) 
                    + k1.arr[s] * 2;
                break;
        }
    }

    __syncthreads();

    if ( maxx < 0 )
        nx.arr[s] += ox.arr[s] + k1.arr[s]*0.1666666666666666;
    __syncthreads();
}


// Bogacki Shampine

__device__
void
dcm_upx_bs(dbuff ox, dbuff y, dbuff u, void *p_theta,
     void *p_ptheta, dbuff nx, MPFLOAT *zs, thr_info tinfo)
{

    //ThetaDCM *theta = (ThetaDCM *) p_theta;
    PThetaDCM *ptheta = (PThetaDCM *) p_ptheta;

    int j = threadIdx.x%y.dim;
    int s;
    int maxx = threadIdx.x - y.dim * ( blockDim.x / y.dim );
    // Buffers for the intermediate results. z is the estimated error.
    dbuff k1, k2, z;

    k1.arr = (ox.arr < nx.arr) ? ox.arr : nx.arr;
    k2.arr = k1.arr;
    z.arr = k1.arr;

    k1.arr += ox.dim * DIM_X * 2;  
    k2.arr += ox.dim * DIM_X * 3;
    z.arr += ox.dim * DIM_X * 4;

    k1.dim = ox.dim;
    k2.dim = ox.dim;
    z.dim = ox.dim;

    // Make the values to be closer in range
 
    if ( isnan( *u.arr ) ){
        if ( threadIdx.y == 0  && maxx < 0 )
        {
            nx.arr[ INDEX_X * ox.dim + j] = NAN;
            nx.arr[ INDEX_F * ox.dim + j] = NAN;
            nx.arr[ INDEX_S * ox.dim + j] = NAN;
            nx.arr[ INDEX_V * ox.dim + j] = NAN;
            nx.arr[ INDEX_Q * ox.dim + j] = NAN;
        }
    }

    // Memory 
    if ( maxx < 0 )
    {
        switch ( threadIdx.y )
        {
            case INDEX_X:
                s = INDEX_X * ox.dim + j;
                break;
            case INDEX_F:
                s = INDEX_F * ox.dim + j;
                break;
            case INDEX_S:
                s = INDEX_S * ox.dim + j;
                break;
            case INDEX_V:
                s = INDEX_V * ox.dim + j;
                break;
            case INDEX_Q:
                s = INDEX_Q * ox.dim + j;
                break;
        }

        k2.arr[s] = ox.arr[s] + ptheta->de * k1.arr[s] * BSA1;
        nx.arr[s] = ox.arr[s] + ptheta->de * k1.arr[s] * BSB1;
        z.arr[s] = k1.arr[s] * BSZ1;
    }

    __syncthreads();
    
    if ( maxx < 0 )
    {
        switch ( threadIdx.y )
        {
            case INDEX_X:
                k1.arr[s] = dcm_dx(k2, y, u, p_theta, p_ptheta, j);
                break;
            case INDEX_F:
                k1.arr[s] = dcm_df(k2, y, u, p_theta, p_ptheta, j);
                break;
            case INDEX_S:
                k1.arr[s] = dcm_ds(k2, y, u, p_theta, p_ptheta, j);
                break;
            case INDEX_V:
                k1.arr[s] = dcm_dv(k2, y, u, p_theta, p_ptheta, j);
                break;
            case INDEX_Q:
                k1.arr[s] = dcm_dq(k2, y, u, p_theta, p_ptheta, j); 
                break;
        }
        nx.arr[s] += ptheta->de * k1.arr[s] * BSB2;
        z.arr[s] += k1.arr[s] * BSZ2;
    }
    
    __syncthreads();

    // Synchronize memory
    if ( maxx < 0 )
        k2.arr[s] = ox.arr[s] + ptheta->de * k1.arr[s] * BSA2; 

    __syncthreads();

    if ( maxx < 0 )
    {
        switch ( threadIdx.y )
        {
            case INDEX_X:
                k1.arr[s] = dcm_dx(k2, y, u, p_theta, p_ptheta, j);
                break;
            case INDEX_F:
                k1.arr[s] = dcm_df(k2, y, u, p_theta, p_ptheta, j);
                break;
            case INDEX_S:
                k1.arr[s] = dcm_ds(k2, y, u, p_theta, p_ptheta, j);
                break;
            case INDEX_V:
                k1.arr[s] = dcm_dv(k2, y, u, p_theta, p_ptheta, j);
                break;
            case INDEX_Q:
                k1.arr[s] = dcm_dq(k2, y, u, p_theta, p_ptheta, j); 
                break;
        }

        nx.arr[s] += ptheta->de * k1.arr[s] * BSB3;
        z.arr[s] += k1.arr[s] * BSZ3;
    }
    
    __syncthreads();

    if ( maxx < 0 )
    {
        switch ( threadIdx.y )
        {
            case INDEX_X:
                k1.arr[s] = dcm_dx(nx, y, u, p_theta, p_ptheta, j);
                break;
            case INDEX_F:
                k1.arr[s] = dcm_df(nx, y, u, p_theta, p_ptheta, j);
                break;
            case INDEX_S:
                k1.arr[s] = dcm_ds(nx, y, u, p_theta, p_ptheta, j);
                break;
            case INDEX_V:
                k1.arr[s] = dcm_dv(nx, y, u, p_theta, p_ptheta, j);
                break;
            case INDEX_Q:
                k1.arr[s] = dcm_dq(nx, y, u, p_theta, p_ptheta, j); 
                break;
        }
        z.arr[s] += k1.arr[s] * BSZ4;
        z.arr[s] *= ptheta->de;
        z.arr[s] = abs(z.arr[s]);
        // If there is a degeneracy don't increase the sampling rate and give
        // up.
        if ( isnan(z.arr[s]) || z.arr[s] == INFINITY )
            z.arr[s] = 0;

    }

    __syncthreads();

    if ( tinfo.cs >= tinfo.ns && threadIdx.y == 0 )
        zs[threadIdx.x] = 0;
    // This is a serious hack

    if ( maxx < 0 && threadIdx.y == 0) 
        zs[threadIdx.x] = z.arr[s];

    __syncthreads();
    if ( threadIdx.x < 16 && threadIdx.y ==  0) 
        zs[threadIdx.x] = zs[threadIdx.x] > zs[threadIdx.x + 16] ? 
            zs[threadIdx.x] : zs[threadIdx.x + 16];
    __syncthreads();
    if ( threadIdx.x < 8 && threadIdx.y ==  0) 
        zs[threadIdx.x] = zs[threadIdx.x] > zs[threadIdx.x + 8] ? 
            zs[threadIdx.x] : zs[threadIdx.x + 8];
    __syncthreads();
   if ( threadIdx.x < 4  && threadIdx.y ==  0) 
        zs[threadIdx.x] = zs[threadIdx.x] > zs[threadIdx.x + 4] ? 
            zs[threadIdx.x] : zs[threadIdx.x + 4];
    __syncthreads();
    if ( threadIdx.x < 2 && threadIdx.y ==  0) 
        zs[threadIdx.x] = zs[threadIdx.x] > zs[threadIdx.x + 2] ? 
            zs[threadIdx.x] : zs[threadIdx.x + 2];
    __syncthreads();
    if ( threadIdx.x == 0 && threadIdx.y ==  0) 
        zs[threadIdx.x] = zs[threadIdx.x] > zs[threadIdx.x + 1] ? 
            zs[threadIdx.x] : zs[threadIdx.x + 1];
    __syncthreads();
}


__device__ void dcm_upx_bs0(dbuff ox, dbuff y, dbuff u, void *p_theta,
     void *p_ptheta, dbuff nx)
{
    // Compute the value of f for the first iteration. This is neede only to
    // initilize the integrator.

    //ThetaDCM *theta = (ThetaDCM *) p_theta;
    //PThetaDCM *ptheta = (PThetaDCM *) p_ptheta;

    int j = threadIdx.x%y.dim;
    int s;
    
    // Buffers for the intermediate results.

    dbuff k1;

    int maxx = threadIdx.x - y.dim * (blockDim.x/y.dim);

    k1.arr = (ox.arr < nx.arr) ? ox.arr : nx.arr;
    k1.arr += ox.dim * DIM_X * 2;  
    k1.dim = ox.dim;

    // Make the values to be closer in range
 
    if ( isnan( *u.arr ) ){
        if ( threadIdx.y == 0  && maxx < 0 )
        {
            nx.arr[ INDEX_X * ox.dim + j] = NAN;
            nx.arr[ INDEX_F * ox.dim + j] = NAN;
            nx.arr[ INDEX_S * ox.dim + j] = NAN;
            nx.arr[ INDEX_V * ox.dim + j] = NAN;
            nx.arr[ INDEX_Q * ox.dim + j] = NAN;
        }
    }

    if ( maxx < 0 )
    {
        switch ( threadIdx.y )
        {
            case INDEX_X:
                s = INDEX_X * ox.dim + j;
                k1.arr[s] = dcm_dx(ox, y, u, p_theta, p_ptheta, j);
                break;
            case INDEX_F:
                s = INDEX_F * ox.dim + j;
                k1.arr[s] = dcm_df(ox, y, u, p_theta, p_ptheta, j);
                break;
            case INDEX_S:
                s = INDEX_S * ox.dim + j;
                k1.arr[s] = dcm_ds(ox, y, u, p_theta, p_ptheta, j);
                break;
            case INDEX_V:
                s = INDEX_V * ox.dim + j;
                k1.arr[s] = dcm_dv(ox, y, u, p_theta, p_ptheta, j);
                break;
            case INDEX_Q:
                s = INDEX_Q * ox.dim + j;
                k1.arr[s] = dcm_dq(ox, y, u, p_theta, p_ptheta, j); 
                break;
        }
    }
}

