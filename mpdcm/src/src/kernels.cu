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


// ==========================================================================
// Kernel code
// ==========================================================================

__global__
void
kdcm_euler(MPFLOAT *x, MPFLOAT *y, MPFLOAT *u, 
    void *p_theta, MPFLOAT *d_theta, void *p_ptheta, MPFLOAT *d_ptheta, 
    int nx, int ny, int nu, int dp, int nt, int nb, unsigned int *errcode)
{

    int i;
    dbuff tx, ty, tu;
    extern __shared__ MPFLOAT sx[];

    // Assign pointers to theta


    ThetaDCM *theta = (ThetaDCM *) p_theta;
    ThetaDCM *ltheta;

    PThetaDCM *ptheta = (PThetaDCM *) p_ptheta;
    __shared__ PThetaDCM lptheta[1];

    lptheta->dt = ptheta->dt;
    lptheta->dyu = ptheta->dyu;
    lptheta->de = ptheta->de;
    lptheta->mode = ptheta->mode;

    tu.dim = nu;
    tx.dim = nx; 
    ty.dim = nx;

    i = threadIdx.x/nx + (blockDim.x / nx) * blockIdx.x;
    // Iterate in chuncks
    while ( i < nb * nt )
    {
        MPFLOAT *o;

        tu.arr = u + (i/nb) * nu * dp;
        // Get the new address

        ltheta = theta + i;

        o = d_theta + i * (
            nx * nx + // A
            nx * nx * nu + // B 
            nx * nu + // C
            nx * nx * nx + // D
            nx + // Kappa (K)
            nx); // tau
        
        ltheta->A = o;
        o += nx * nx;

        ltheta->B = o;
        o += nx * nx * nu;

        ltheta->C = o; 
        o+= nx * nu;

        ltheta->D = o;
        o += nx * nx * nx;

        ltheta->K = o;
        o += nx;

        ltheta->tau = o; 

        tx.arr = sx + PRELOC_SIZE_X_EULER * DIM_X * nx * (threadIdx.x/nx);

        ty.arr = y + i * nx * ny;
        dcm_int_euler(tx, ty, tu, (void *) ltheta, (void *) lptheta, dp, 
            errcode);

        i += gridDim.x * (blockDim.x / nx );        
    }
}

__global__
void
kdcm_kr4(MPFLOAT *x, MPFLOAT *y, MPFLOAT *u, 
    void *p_theta, MPFLOAT *d_theta, void *p_ptheta, MPFLOAT *d_ptheta, 
    int nx, int ny, int nu, int dp, int nt, int nb, unsigned int *errcode)
{

    int i;
    dbuff tx, ty, tu;
    extern __shared__ MPFLOAT sx[];

    // Assign pointers to theta


    ThetaDCM *theta = (ThetaDCM *) p_theta;
    ThetaDCM *ltheta;

    PThetaDCM *ptheta = (PThetaDCM *) p_ptheta;
    __shared__ PThetaDCM lptheta[1];

    lptheta->dt = ptheta->dt;
    lptheta->dyu = ptheta->dyu;
    lptheta->de = ptheta->de;
    lptheta->mode = ptheta->mode;

    tu.dim = nu;
    tx.dim = nx; 
    ty.dim = nx;

    i = threadIdx.x/nx + (blockDim.x / nx) * blockIdx.x;
    // Iterate in chuncks
    while ( i < nb * nt )
    {
        MPFLOAT *o;

        tu.arr = u + (i/nb) * nu * dp;
        // Get the new address

        ltheta = theta + i;

        o = d_theta + i * (
            nx * nx + // A
            nx * nx * nu + // B 
            nx * nu + // C
            nx * nx * nx + // D
            nx + // Kappa (K)
            nx); // tau
        
        ltheta->A = o;
        o += nx * nx;

        ltheta->B = o;
        o += nx * nx * nu;

        ltheta->C = o; 
        o += nx * nu;

        ltheta->D = o;
        o += nx * nx * nx;

        ltheta->K = o;
        o += nx;

        ltheta->tau = o; 

        tx.arr = sx + PRELOC_SIZE_X_KR4 * DIM_X * nx * (threadIdx.x/nx);

        ty.arr = y + i * nx * ny;
        dcm_int_kr4(tx, ty, tu, (void *) ltheta, (void *) lptheta, dp, errcode);
        i += gridDim.x * (blockDim.x / nx );        
    }
}

__global__
void
kdcm_bs(MPFLOAT *x, MPFLOAT *y, MPFLOAT *u, 
    void *p_theta, MPFLOAT *d_theta, void *p_ptheta, MPFLOAT *d_ptheta, 
    int nx, int ny, int nu, int dp, int nt, int nb, unsigned int * errcode)
{
    /* 
    mem -- Prealocate shared memory. It depends on the slots that the 
        integrator needs; two for euler and 4 for Kutta-Ruge.
    fupx -- Function used to integrate the update the system. 
    */

    int i;
    dbuff tx, ty, tu;
    extern __shared__ MPFLOAT sx[];

    // Assign pointers to theta

    ThetaDCM *theta = (ThetaDCM *) p_theta;
    ThetaDCM *ltheta;

    PThetaDCM *ptheta = (PThetaDCM *) p_ptheta;
    __shared__ PThetaDCM lptheta[1];

    thr_info tinfo[1];

    tinfo->ns = nt * nb;

    lptheta->dt = ptheta->dt;
    lptheta->dyu = ptheta->dyu;
    lptheta->de = ptheta->de;
    lptheta->mode = ptheta->mode;

    tu.dim = nu;
    tx.dim = nx; 
    ty.dim = nx;

    i = threadIdx.x/nx + (blockDim.x / nx) * blockIdx.x;
    // Iterate in chuncks
    while ( i < nb * nt )
    {
        MPFLOAT *o;

        tinfo->cs = i;

        tu.arr = u + (i/nb) * nu * dp;
        // Get the new address

        ltheta = theta + i;

        o = d_theta + i * (
            nx * nx + // A
            nx * nx * nu + // B 
            nx * nu + // C
            nx * nx * nx + // D
            nx + // Kappa (K)
            nx); // tau
        
        ltheta->A = o;
        o += nx * nx;

        ltheta->B = o;
        o += nx * nx * nu;

        ltheta->C = o; 
        o += nx * nu;

        ltheta->D = o;
        o += nx * nx * nx;

        ltheta->K = o;
        o += nx;

        ltheta->tau = o; 

        tx.arr = sx + PRELOC_SIZE_X_BS * DIM_X * nx * (threadIdx.x/nx);

        ty.arr = y + i * nx * ny;
        dcm_int_bs(tx, ty, tu, (void *) ltheta, (void *) lptheta, dp, 
            errcode, *tinfo);
        i += gridDim.x * (blockDim.x / nx );        
    }
}


// ===========================================================================
// Kernel (l)auncher
// ===========================================================================


__host__
void
ldcm_euler
(MPFLOAT *x, MPFLOAT *y, MPFLOAT *u, 
    void *theta, MPFLOAT *d_theta, void *ptheta, MPFLOAT *d_ptheta, 
    int nx, int ny, int nu, int dp, int nt, int nb, unsigned int *errcode)
{

    int device;
    cudaGetDevice(&device);

    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    int num_blocks = min((nx * nt * nb + NUM_THREADS - 1)/NUM_THREADS,
        NUM_BLOCKS * props.multiProcessorCount);

    
    dim3 gthreads(NUM_THREADS, DIM_X);
    dim3 gblocks(num_blocks, 1);

    int sems;
    sems =  NUM_THREADS * DIM_X * PRELOC_SIZE_X_EULER * sizeof( MPFLOAT );


    kdcm_euler<<<gblocks, gthreads, sems>>>(x, y, u, 
        theta, d_theta, ptheta, d_ptheta, 
        nx, ny, nu, dp, nt, nb, errcode); 
}


__host__
void
ldcm_kr4(MPFLOAT *x, MPFLOAT *y, MPFLOAT *u, 
    void *theta, MPFLOAT *d_theta, void *ptheta, MPFLOAT *d_ptheta, 
    int nx, int ny, int nu, int dp, int nt, int nb, unsigned int *errcode)
{
    int device;
    cudaGetDevice(&device);

    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    int num_blocks = min((nx * nt * nb + NUM_THREADS - 1)/NUM_THREADS,
        NUM_BLOCKS * props.multiProcessorCount);
    
    dim3 gthreads(NUM_THREADS, DIM_X);
    dim3 gblocks(num_blocks, 1);

    int smems = NUM_THREADS * DIM_X * PRELOC_SIZE_X_KR4 * sizeof( MPFLOAT );

    kdcm_kr4<<<gblocks, gthreads, smems>>>(x, y, u, 
        theta, d_theta, ptheta, d_ptheta, 
        nx, ny, nu, dp, nt, nb, errcode); 
}

__host__
void 
ldcm_bs(MPFLOAT *x, MPFLOAT *y, MPFLOAT *u, 
    void *theta, MPFLOAT *d_theta, void *ptheta, MPFLOAT *d_ptheta, 
    int nx, int ny, int nu, int dp, int nt, int nb, unsigned int *errcode)
{

    int device;
    cudaGetDevice(&device);

    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    int num_blocks = min((nx * nt * nb + NUM_THREADS - 1)/NUM_THREADS,
        NUM_BLOCKS * props.multiProcessorCount);

    dim3 gthreads(NUM_THREADS, DIM_X);
    dim3 gblocks(num_blocks, 1);

    int smems = NUM_THREADS * DIM_X * PRELOC_SIZE_X_BS * sizeof( MPFLOAT );

    kdcm_bs<<<gblocks, gthreads, smems>>>(x, y, u, 
        theta, d_theta, ptheta, d_ptheta, 
        nx, ny, nu, dp, nt, nb, errcode); 
}



