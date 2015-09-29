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
kdcm_euler(kernpars pars, unsigned int *errcode)
{

    MPFLOAT *y = pars.y;
    MPFLOAT *u = pars.u;
    void *p_theta = (void *) pars.p_theta;
    MPFLOAT *d_theta = pars.d_theta;
    void *p_ptheta = (void *) pars.p_ptheta;
    int nx = pars.nx;
    int ny = pars.ny;
    int nu = pars.nu;
    int dp = pars.dp;
    int nt = pars.nt;
    int nb = pars.nb; 

    int i,j;
    dbuff tx, ty, tu;
    extern __shared__ MPFLOAT sx[];

    // Assign pointers to theta

    ThetaDCM *theta = (ThetaDCM *) p_theta;
    ThetaDCM *ltheta;

    PThetaDCM *ptheta = (PThetaDCM *) p_ptheta;
    __shared__ PThetaDCM lptheta[1];
    __shared__ MPFLOAT shA[1024];
    __shared__ MPFLOAT shC[320];

    sqsparse sB[1];
    sqsparse sD[1];

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
            nx * nu + // C
            nx + // Kappa (K)
            nx); // tau
        
        // Use shared memory
        ltheta->A = shA + nx * nx * (threadIdx.x/nx);
        
        // Transfere in parallel
        for (j = 0; j < nx; j++)
            if (threadIdx.y == 0)
                // Transpose the matrix
                ltheta->A[j * nx  + threadIdx.x % nx] = 
                    o[j + nx * (threadIdx.x%nx)];
             
        o += nx * nx;

        ltheta->C = o;
//        ltheta->C = shC + nx * nu * (threadIdx.x/nx);
//
//        for (j = 0; j < nu; j++)
//            if ( threadIdx.y == 0 )
//            ltheta->C[j * nx  + threadIdx.x % nx] = o[j * nx + threadIdx.x%nx];

        o+= nx * nu;

        ltheta->K = o;
        o += nx;

        ltheta->tau = o; 

        ltheta->sB = sB;
        ltheta->sD = sD;

        // Assign the appropriate offset

        ltheta->sB->j = pars.jB + (nx + 1) * nu * i;
        ltheta->sD->j = pars.jD + (nx + 1) * nx * i;

        ltheta->sB->i = pars.iB;
        ltheta->sD->i = pars.iD;

        ltheta->sB->v = pars.vB; 
        ltheta->sD->v = pars.vD;

        tx.arr = sx + PRELOC_SIZE_X_EULER * DIM_X * nx * (threadIdx.x/nx);

        ty.arr = y + i * nx * ny;
        dcm_int_euler(tx, ty, tu, (void *) ltheta, (void *) lptheta, dp, 
            errcode);

        i += gridDim.x * (blockDim.x / nx );        
    }
}

__global__
void
kdcm_kr4(kernpars pars, unsigned int *errcode)
{

    MPFLOAT *y = pars.y;
    MPFLOAT *u = pars.u;
    void *p_theta = (void *) pars.p_theta;
    MPFLOAT *d_theta = pars.d_theta;
    void *p_ptheta = (void *) pars.p_ptheta;
    int nx = pars.nx;
    int ny = pars.ny;
    int nu = pars.nu;
    int dp = pars.dp;
    int nt = pars.nt;
    int nb = pars.nb; 

    int i, j;
    dbuff tx, ty, tu;
    extern __shared__ MPFLOAT sx[];

    // Assign pointers to theta

    ThetaDCM *theta = (ThetaDCM *) p_theta;
    ThetaDCM *ltheta;

    PThetaDCM *ptheta = (PThetaDCM *) p_ptheta;
    __shared__ PThetaDCM lptheta[1];
    __shared__ MPFLOAT shA[1024];
    __shared__ MPFLOAT shC[320];



    // Assign the sparse matrices

    sqsparse sB[1];
    sqsparse sD[1];

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
        //int nB, nD;

        tu.arr = u + (i/nb) * nu * dp;
        // Get the new address

        ltheta = theta + i;

        o = d_theta + i * (
            nx * nx + // A
            nx * nu + // C
            nx + // Kappa (K)
            nx); // tau
        
        ltheta->A = shA + nx * nx * (threadIdx.x/nx);

        // Transfere in parallel
        for (j = 0; j < nx; j++)
            if (threadIdx.y == 0)
                // Transpose the matrix
                ltheta->A[j * nx  + threadIdx.x % nx] = 
                    o[j + nx * (threadIdx.x%nx)];
             
        o += nx * nx;

        ltheta->C = shC + nx * nu * (threadIdx.x/nx);

        for (j = 0; j < nu; j++)
            if (threadIdx.y == 0)
            ltheta->C[j * nx  + threadIdx.x % nx] = o[j * nx + threadIdx.x%nx];

        o += nx * nu;

        ltheta->K = o;
        o += nx;

        ltheta->tau = o;

        ltheta->sB = sB;
        ltheta->sD = sD;

        // Assign the appropriate offset

        ltheta->sB->j = pars.jB + (nx + 1) * nu * i;
        ltheta->sD->j = pars.jD + (nx + 1) * nx * i;

        ltheta->sB->i = pars.iB;
        ltheta->sD->i = pars.iD;

        ltheta->sB->v = pars.vB; 
        ltheta->sD->v = pars.vD;

        tx.arr = sx + PRELOC_SIZE_X_KR4 * DIM_X * nx * (threadIdx.x/nx);

        ty.arr = y + i * nx * ny;
        dcm_int_kr4(tx, ty, tu, (void *) ltheta, (void *) lptheta, dp, errcode);
        i += gridDim.x * (blockDim.x / nx );        
    }
}

__global__
void
kdcm_bs(kernpars pars, unsigned int * errcode)
{
    /* 
    mem -- Prealocate shared memory. It depends on the slots that the 
        integrator needs; two for euler and 4 for Kutta-Ruge.
    fupx -- Function used to integrate the update the system. 
    */

    MPFLOAT *y = pars.y;
    MPFLOAT *u = pars.u;
    void *p_theta = (void *) pars.p_theta;
    MPFLOAT *d_theta = pars.d_theta;
    void *p_ptheta = (void *) pars.p_ptheta;
    int nx = pars.nx;
    int ny = pars.ny;
    int nu = pars.nu;
    int dp = pars.dp;
    int nt = pars.nt;
    int nb = pars.nb; 

    int i;
    dbuff tx, ty, tu;
    extern __shared__ MPFLOAT sx[];

    // Assign pointers to theta

    ThetaDCM *theta = (ThetaDCM *) p_theta;
    ThetaDCM *ltheta;

    PThetaDCM *ptheta = (PThetaDCM *) p_ptheta;
    __shared__ PThetaDCM lptheta[1];

    thr_info tinfo[1];

    // Assign the sparse matrices

    sqsparse sB[1];
    sqsparse sD[1];

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
            nx * nu + // C
            nx + // Kappa (K)
            nx); // tau
        
        ltheta->A = o;
        o += nx * nx;

        ltheta->C = o; 
        o += nx * nu;

        ltheta->K = o;
        o += nx;

        ltheta->tau = o; 

        ltheta->sB = sB;
        ltheta->sD = sD;

        // Assign the appropriate offset

        ltheta->sB->j = pars.jB + (nx + 1) * nu * i;
        ltheta->sD->j = pars.jD + (nx + 1) * nx * i;

        ltheta->sB->i = pars.iB;
        ltheta->sD->i = pars.iD;

        ltheta->sB->v = pars.vB; 
        ltheta->sD->v = pars.vD;

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
ldcm_euler(kernpars pars, unsigned int *errcode)
{

    int device;
    cudaGetDevice(&device);

    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    int num_blocks = 
        min((pars.nx * pars.nt * pars.nb + NUM_THREADS - 1)/NUM_THREADS,
        NUM_BLOCKS * props.multiProcessorCount);

    
    dim3 gthreads(NUM_THREADS, DIM_X);
    dim3 gblocks(num_blocks, 1);

    int sems;
    sems =  NUM_THREADS * DIM_X * PRELOC_SIZE_X_EULER * sizeof( MPFLOAT );


    kdcm_euler<<<gblocks, gthreads, sems>>>(pars, errcode); 
}


__host__
void
ldcm_kr4(kernpars pars, unsigned int *errcode)
{
    int device;
    cudaGetDevice(&device);

    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    int num_blocks = 
        min((pars.nx * pars.nt * pars.nb + NUM_THREADS - 1)/NUM_THREADS,
        NUM_BLOCKS * props.multiProcessorCount);
    
    dim3 gthreads(NUM_THREADS, DIM_X);
    dim3 gblocks(num_blocks, 1);

    int smems = NUM_THREADS * DIM_X * PRELOC_SIZE_X_KR4 * sizeof( MPFLOAT );

    kdcm_kr4<<<gblocks, gthreads, smems>>>(pars, errcode); 
}

__host__
void 
ldcm_bs(kernpars pars, unsigned int *errcode)
{

    int device;
    cudaGetDevice(&device);

    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    int num_blocks = 
        min((pars.nx * pars.nt * pars.nb + NUM_THREADS - 1)/NUM_THREADS,
        NUM_BLOCKS * props.multiProcessorCount);

    dim3 gthreads(NUM_THREADS, DIM_X);
    dim3 gblocks(num_blocks, 1);

    int smems = NUM_THREADS * DIM_X * PRELOC_SIZE_X_BS * sizeof( MPFLOAT );

    kdcm_bs<<<gblocks, gthreads, smems>>>(pars, errcode); 
}



