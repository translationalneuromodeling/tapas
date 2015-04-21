/* aponteeduardo@gmail.com */
/* copyright (C) 2014 */


#include "mpdcm.hcu"

// TODO move to the header

__device__
void 
dcm_upx_euler(dbuff ox, dbuff y, dbuff u, void *p_theta, void *p_ptheta, 
    dbuff nx);

__device__ 
void
dcm_upx_kr4(dbuff ox, dbuff y, dbuff u, void *p_theta, void *p_ptheta, 
    dbuff nx);

__device__
void 
dcm_upx_bs(dbuff ox, dbuff y, dbuff u, void *p_theta, void *p_ptheta, 
    dbuff nx, double* zs, thr_info tinfo);

__device__
void 
dcm_upx_bs0(dbuff ox, dbuff y, dbuff u, void *p_theta, void *p_ptheta, 
    dbuff nx);
__device__
void
bs_maxz(dbuff z);


// General functions

__device__ double dcm_dx(dbuff x, dbuff y, dbuff u, void *p_theta,
     void *p_ptheta, int i)
{
    double dx=0;
    double bt=0;
    int j;
    int k;
    int p;
    int o;

    ThetaDCM *theta = (ThetaDCM *) p_theta;
    o = INDEX_X * x.dim;

    // A
    for (j = 0; j < x.dim; j++)
    {
        dx += theta->A[i + x.dim*j] * x.arr[o + j];
    }

    for (j = 0; j < u.dim; j++)
    {
        if (  u.arr[j] == 0  )
            continue;
        // B
        bt = 0;
        k = x.dim*x.dim*j + i;
        for (p = 0; p < x.dim; p++){
            bt += theta->B[k + x.dim*p] * x.arr[o + p];
        }
        // C
        dx += (theta->C[i + x.dim*j] + bt)*u.arr[j];
    }

    return dx;
}

__device__ double dcm_ds(dbuff x, dbuff y, dbuff u, void *p_theta,
    void *p_ptheta, int i)
{
    double ds;

    ThetaDCM *theta = (ThetaDCM *) p_theta;
    //PThetaDCM *ptheta = (PThetaDCM  *) p_ptheta;

    ds = x.arr[INDEX_X * x.dim + i] - 
        theta->K[i] * x.arr[INDEX_S * x.dim + i] -
        theta->gamma * (exp(x.arr[INDEX_F * x.dim + i]) - 1);

    return ds;
}

__device__ double dcm_df(dbuff x, dbuff y, dbuff u, void *p_theta, 
    void *p_ptheta, int i)
{
    double df;

    //ThetaDCM *theta = (ThetaDCM *) p_theta;
    //PThetaDCM *ptheta = (PThetaDCM  *) p_ptheta;

    df = x.arr[INDEX_S * x.dim + i]*exp(-x.arr[INDEX_F * x.dim + i]);

    return df;
}

__device__ double dcm_dv(dbuff x, dbuff y, dbuff u, void *p_theta,
    void *p_ptheta, int i)
{
    double dv;

    ThetaDCM *theta = (ThetaDCM *) p_theta;
    //PThetaDCM *ptheta = (PThetaDCM  *) p_ptheta;

    dv = exp(x.arr[INDEX_F * x.dim + i] - x.arr[INDEX_V * x.dim + i] - 
            theta->tau[i]) -
        exp(x.arr[INDEX_V * x.dim + i] * theta->alpha - theta->tau[i]);

    return dv;
}

__device__ double dcm_dq(dbuff x, dbuff y, dbuff u, void *p_theta, 
    void *p_ptheta, int i)
{
    double dq = 0;
    double f = exp(-x.arr[INDEX_F * x.dim + i]);
    double v;
    double lnE0; 
    ThetaDCM *theta = (ThetaDCM *) p_theta;
    
    v = exp(x.arr[INDEX_V * x.dim + i] * theta->alpha - theta->tau[i]);
    lnE0 = theta->lnE0;

    //    PThetaDCM *ptheta = (PThetaDCM  *) p_ptheta;

    dq = (1 - exp(f * theta->ln1_E0))*exp(x.arr[INDEX_F * x.dim + i] -
        lnE0 - theta->tau[i] -  x.arr[INDEX_Q * x.dim + i]) - v;

    return dq;
}

__device__ double dcm_lk1(dbuff x, dbuff y, dbuff u, void *p_theta,
            void *p_ptheta, int i)
{
    double l;
    double q = exp(x.arr[INDEX_Q * x.dim + i]);

    ThetaDCM *theta = (ThetaDCM *) p_theta;
    //PThetaDCM *ptheta = (PThetaDCM *) p_ptheta;

    l = theta->k1 * ( 1 - q);

    return l;
}


__device__ double dcm_lk2(dbuff x, dbuff y, dbuff u, void *p_theta,
            void *p_ptheta, int i)
{
    double l;
    double qv = exp(x.arr[INDEX_Q * x.dim + i] - x.arr[INDEX_V *x.dim + i]);

    ThetaDCM *theta = (ThetaDCM *) p_theta;
    //PThetaDCM *ptheta = (PThetaDCM *) p_ptheta;

    l = theta->k2 * ( 1 - qv);

    return l;
}

__device__ double dcm_lk3(dbuff x, dbuff y, dbuff u, void *p_theta,
            void *p_ptheta, int i)
{
    double l;

    double v = exp(x.arr[INDEX_V * x.dim + i]);

    ThetaDCM *theta = (ThetaDCM *) p_theta;
    //PThetaDCM *ptheta = (PThetaDCM *) p_ptheta;

    l = theta->k3 * ( 1 - v);

    return l;
}

__device__ void dcm_upy(dbuff ox, dbuff y, dbuff u, void *theta,
    void *ptheta, dbuff nx)
{
    // Network node
    int j = threadIdx.x%y.dim;

    switch( threadIdx.y )
    {
        case INDEX_LK1 :
            nx.arr[ INDEX_LK1 * nx.dim + j] =
                dcm_lk1(ox, y, u, theta, ptheta, j);
            break;
        case INDEX_LK2:
            nx.arr[ INDEX_LK2 * nx.dim + j] =
                dcm_lk2(ox, y, u, theta, ptheta, j);
            break;
        case INDEX_LK3 :
            nx.arr[ INDEX_LK3 * nx.dim + j] =
                dcm_lk3(ox, y, u, theta, ptheta, j);
            break;
    }

}

// =======================================================================
// Integrators 
// =======================================================================

__device__ void dcm_int_euler(dbuff x, dbuff y, dbuff u, void *p_theta,
    void *p_ptheta, int dp, unsigned int *errcode)
{
    int i;
    int j = threadIdx.x%y.dim;
    double *t;
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
        memset(x.arr, 0, nx.dim * DIM_X * sizeof(double));

    __syncthreads();
    ty.dim = y.dim;
    tu.dim = u.dim;

    // How many samples are gonna be taken
    ss = ceil(1.0/ptheta->dt);
    dy = ceil(1.0/(ptheta->dt*ptheta->dyu));

    ty.arr = y.arr; 
    tu.arr = u.arr;

    for (i=0; i < dp*ss; i++)
    {
        if ( threadIdx.x < maxx )
            dcm_upx_euler(ox, ty, tu, p_theta, p_ptheta, nx);
        __syncthreads();
        // Only sample every 1/ptheta->dt times
        if ( i%ss == 0 )
        {
           if ( i%dy == 0 ) 
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
            if ( i > 0 )
                tu.arr += u.dim;
        }
        // Swap the pointers
        t = ox.arr;
        ox.arr = nx.arr;
        nx.arr = t;
    }

    *errcode = 0;
}

// Runge Kutta

__device__ void dcm_int_kr4(dbuff x, dbuff y, dbuff u, void *p_theta,
    void *p_ptheta, int dp, unsigned int *errcode)
{
    int i;
    int j = threadIdx.x%y.dim;
    double *t;
    // Number of integration steps done between each data point
    int ss, dy;
    // Point where threads are not synchronized to anything
    int maxx = y.dim * (blockDim.x/y.dim);

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

    if ( threadIdx.x < maxx )
        memset(x.arr, 0, nx.dim * DIM_X * sizeof(double));

    __syncthreads();
    ty.dim = y.dim;
    tu.dim = u.dim;

    // How many samples are gonna be taken
    ss = ceil(1.0/ptheta->dt);
    dy = ceil(1.0/(ptheta->dt*ptheta->dyu));

    ty.arr = y.arr; 
    tu.arr = u.arr;

    for (i=0; i < dp*ss; i++)
    {
        //if ( threadIdx.x < maxx )
            dcm_upx_kr4(ox, ty, tu, p_theta, p_ptheta, nx);
        __syncthreads();
        // Only sample every 1/ptheta->dt times
        if ( i%ss == 0 )
        {
            if ( i%dy == 0 ) 
            {
                if ( threadIdx.x < maxx )
                    dcm_upy(nx, ty, tu, p_theta, p_ptheta, ox);           
                __syncthreads();
                if ( threadIdx.x < maxx && threadIdx.y == 0 )
                    ty.arr[j] = ox.arr[INDEX_LK1 * ox.dim + j] +
                        ox.arr[ INDEX_LK2 * ox.dim + j] +
                        ox.arr[ INDEX_LK3 * ox.dim + j];
                if ( i > 0 )
                    ty.arr += y.dim;
               __syncthreads(); 
            }
            if ( i > 0 )
                tu.arr += u.dim;
        }
        // Swap the pointers
        t = ox.arr;
        ox.arr = nx.arr;
        nx.arr = t;
    }
    *errcode = 0;
}

#define MINDY 1
#define MAXDY 64

#define MINTOL 0.000001
#define MAXTOL 0.001000


// Bucacki Shampinee

__device__ void dcm_int_bs(dbuff x, dbuff y, dbuff u, void *p_theta,
    void *p_ptheta, int dp, unsigned int *errcode, thr_info tinfo)
{
    int i;
    int j = threadIdx.x%y.dim;
    double *t;
    double z;
    // Number of integration steps done between each output
    int dy;
    // Point where threads are not synchronized to anything
    int maxx = threadIdx.x - y.dim * (blockDim.x/y.dim);
    unsigned int ndt = MAXDY, odt = MAXDY;

    __shared__ double zs[NUM_THREADS];

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
        memset(x.arr, 0, nx.dim * DIM_X * sizeof(double));

    // Restart the errors
    if ( threadIdx.y == 0 )
        zs[threadIdx.x] = 0;

    __syncthreads();

    ty.dim = y.dim;
    tu.dim = u.dim;
    ty.arr = y.arr; 
    tu.arr = u.arr;

    // How many samples are gonna be taken

    dy = MAXDY * ceil(1 / ptheta->dyu);
    ptheta->de = 2*ptheta->dyu;
    __syncthreads();

    dcm_upx_bs0(ox, ty, tu, p_theta, p_ptheta, nx);
    __syncthreads();

    i = 0;

    while ( i < dp * MAXDY + 1 )
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
                ptheta->de = 2 * ptheta->dyu * (((float ) odt)/MAXDY);   
        
            __syncthreads();
        
            continue;
        }
        
        // Below the error tolerance

        if ( z < MINTOL && odt < MAXDY )
        {
            odt <<= 1;
        }

        // Always sample at the right spot.
        if ( i%MAXDY + odt > MAXDY )
            ndt = MAXDY - i%MAXDY;
        else 
            ndt = odt;

        if ( threadIdx.x == 0 && threadIdx.y == 0 )
            ptheta->de = 2*ptheta->dyu * (((float ) ndt)/MAXDY);
       
        __syncthreads();

        // Only sample every 1/ptheta->dt times
        if ( i%MAXDY == 0 )
        {
            if ( i%dy == 0 ) 
            {
               if ( maxx < 0 )
                    dcm_upy(nx, ty, tu, p_theta, p_ptheta, ox);           
                __syncthreads();
               if ( maxx < 0 && threadIdx.y == 0 )
                   ty.arr[j] = ox.arr[INDEX_LK1 * ox.dim + j] +
                       ox.arr[ INDEX_LK2 * ox.dim + j] +
                       ox.arr[ INDEX_LK3 * ox.dim + j];
                if ( i > 0 )
                    ty.arr += y.dim;
               __syncthreads(); 
            }
            if ( i > 0 )
                tu.arr += u.dim;
        }
        // Swap the pointers
        t = ox.arr;
        ox.arr = nx.arr;
        nx.arr = t;

        i += ndt;
    }
    *errcode = 0;

}


// ==========================================================================
// Kernel code
// ==========================================================================

__global__ void kdcm_euler(double *x, double *y, double *u, 
    void *p_theta, double *d_theta, void *p_ptheta, double *d_ptheta, 
    int nx, int ny, int nu, int dp, int nt, int nb, unsigned int *errcode)
{

    int i;
    dbuff tx, ty, tu;
    extern __shared__ double sx[];

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
        double *o;

        tu.arr = u + (i/nb) * nu * dp;
        // Get the new address

        ltheta = theta + i;

        o = d_theta + i * (
            nx * nx + // A
            nx * nx * nu + // B 
            nx * nu + // C
            nx + // Kappa (K)
            nx); // tau
        
        ltheta->A = o;
        o += nx * nx;

        ltheta->B = o;
        o += nx * nx * nu;

        ltheta->C = o; 
        o+= nx * nu;

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

__global__ void kdcm_kr4(double *x, double *y, double *u, 
    void *p_theta, double *d_theta, void *p_ptheta, double *d_ptheta, 
    int nx, int ny, int nu, int dp, int nt, int nb, unsigned int *errcode)
{

    int i;
    dbuff tx, ty, tu;
    extern __shared__ double sx[];

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
        double *o;

        tu.arr = u + (i/nb) * nu * dp;
        // Get the new address

        ltheta = theta + i;

        o = d_theta + i * (
            nx * nx + // A
            nx * nx * nu + // B 
            nx * nu + // C
            nx + // Kappa (K)
            nx); // tau
        
        ltheta->A = o;
        o += nx * nx;

        ltheta->B = o;
        o += nx * nx * nu;

        ltheta->C = o; 
        o+= nx * nu;

        ltheta->K = o;
        o += nx;

        ltheta->tau = o; 

        tx.arr = sx + PRELOC_SIZE_X_KR4 * DIM_X * nx * (threadIdx.x/nx);

        ty.arr = y + i * nx * ny;
        dcm_int_kr4(tx, ty, tu, (void *) ltheta, (void *) lptheta, dp, errcode);
        i += gridDim.x * (blockDim.x / nx );        
    }
}

__global__ void kdcm_bs(double *x, double *y, double *u, 
    void *p_theta, double *d_theta, void *p_ptheta, double *d_ptheta, 
    int nx, int ny, int nu, int dp, int nt, int nb, unsigned int * errcode)
{
    /* 
    mem -- Prealocate shared memory. It depends on the slots that the 
        integrator needs; two for euler and 4 for Kutta-Ruge.
    fupx -- Function used to integrate the update the system. 
    */

    int i;
    dbuff tx, ty, tu;
    extern __shared__ double sx[];

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
        double *o;

        tinfo->cs = i;

        tu.arr = u + (i/nb) * nu * dp;
        // Get the new address

        ltheta = theta + i;

        o = d_theta + i * (
            nx * nx + // A
            nx * nx * nu + // B 
            nx * nu + // C
            nx + // Kappa (K)
            nx); // tau
        
        ltheta->A = o;
        o += nx * nx;

        ltheta->B = o;
        o += nx * nx * nu;

        ltheta->C = o; 
        o+= nx * nu;

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


__host__ void ldcm_euler
(double *x, double *y, double *u, 
    void *theta, double *d_theta, void *ptheta, double *d_ptheta, 
    int nx, int ny, int nu, int dp, int nt, int nb, unsigned int *errcode)
{

    int device;
    cudaGetDevice(&device);

    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    dim3 gthreads(NUM_THREADS, DIM_X);
    dim3 gblocks(NUM_BLOCKS * props.multiProcessorCount, 1);

    int sems;
    sems =  NUM_THREADS * DIM_X * PRELOC_SIZE_X_EULER * sizeof( double );


    kdcm_euler<<<gblocks, gthreads, sems>>>(x, y, u, 
        theta, d_theta, ptheta, d_ptheta, 
        nx, ny, nu, dp, nt, nb, errcode); 
}


__host__ void
ldcm_kr4(double *x, double *y, double *u, 
    void *theta, double *d_theta, void *ptheta, double *d_ptheta, 
    int nx, int ny, int nu, int dp, int nt, int nb, unsigned int *errcode)
{
    int device;
    cudaGetDevice(&device);

    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    dim3 gthreads(NUM_THREADS, DIM_X);
    dim3 gblocks(NUM_BLOCKS * props.multiProcessorCount, 1);

    int smems = NUM_THREADS * DIM_X * PRELOC_SIZE_X_KR4 * sizeof( double );

    kdcm_kr4<<<gblocks, gthreads, smems>>>(x, y, u, 
        theta, d_theta, ptheta, d_ptheta, 
        nx, ny, nu, dp, nt, nb, errcode); 
}

__host__ void 
ldcm_bs(double *x, double *y, double *u, 
    void *theta, double *d_theta, void *ptheta, double *d_ptheta, 
    int nx, int ny, int nu, int dp, int nt, int nb, unsigned int *errcode)
{

    int device;
    cudaGetDevice(&device);

    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    dim3 gthreads(NUM_THREADS, DIM_X);
    dim3 gblocks(NUM_BLOCKS * props.multiProcessorCount, 1);

    int smems = NUM_THREADS * DIM_X * PRELOC_SIZE_X_BS * sizeof( double );

    kdcm_bs<<<gblocks, gthreads, smems>>>(x, y, u, 
        theta, d_theta, ptheta, d_ptheta, 
        nx, ny, nu, dp, nt, nb, errcode); 
}


// ===========================================================================
// Allocate memory
// ===========================================================================

// Device alloc memory theta
__host__ 
void
dam_theta(
    void **theta, double **d_theta,
    void **pd_theta, double **dd_theta,
    int nx, int ny, int nu, int dp, int nt, int nb)
{

    int tp;

    // Allocate memory for the structures

    HANDLE_ERROR( cudaMalloc( pd_theta, nt * nb * sizeof(ThetaDCM)));
    HANDLE_ERROR( cudaMemcpy( *pd_theta, *theta, nt * nb * sizeof(ThetaDCM),
        cudaMemcpyHostToDevice ) );

    // Allocate memory for the matrices. It is assumed that the parameters
    // are in a block of contiguous memory with the for A, Bs, C, x0, epsilon,
    // Kappa, tau

    tp = nt * nb * 
        (nx * nx +      // A:
        nx * nx * nu +  // B's
        nx * nu +       // C
        nx + // kappa
        nx); // tau 

    HANDLE_ERROR( cudaMalloc( dd_theta, tp * sizeof(double) ) );
    HANDLE_ERROR( cudaMemcpy( (void *) *dd_theta, (void *) *d_theta, 
        tp * sizeof(double), cudaMemcpyHostToDevice ) );

}

// Device alloc memory ptheta
__host__ 
void 
dam_ptheta(
    void **ptheta, double **d_ptheta, 
    void **pd_ptheta, double **dd_ptheta, 
    int nx, int ny, int nu, int dp, int nt, int nb)
{

    HANDLE_ERROR( cudaMalloc( pd_ptheta, sizeof(PThetaDCM)));
    HANDLE_ERROR( cudaMemcpy( *pd_ptheta, *ptheta, sizeof(PThetaDCM),
        cudaMemcpyHostToDevice ) );
 
}

// ===========================================================================
// Host code
// ===========================================================================

extern "C"
int 
mpdcm_fmri( double *x, double *y, double *u,
    void *theta, double *d_theta,
    void *ptheta, double *d_ptheta, 
    int nx, int ny, int nu, int dp, int nt, int nb,
    klauncher launcher)
{

    double *d_x, *d_y, *d_u;
    void *pd_theta, *pd_ptheta;
    double *dd_theta, *dd_ptheta;
    unsigned int errcode[1], *d_errcode;

    // x

    d_x = 0;

    // y

    HANDLE_ERROR( cudaMalloc( (void **) &d_y,
        nx * ny * nt * nb * sizeof(double) ) );

    // u

    HANDLE_ERROR( cudaMalloc( (void**) &d_u,
        nt * nu * dp *  sizeof(double) ) );

    HANDLE_ERROR( cudaMemcpy( d_u, u, nt * nu * dp * sizeof(double),
        cudaMemcpyHostToDevice ) );

    // Error code

    HANDLE_ERROR( cudaMalloc( (void**) &d_errcode, 
        sizeof( unsigned int ) ) );


    // Theta 
    dam_theta(
        &theta, &d_theta,
        &pd_theta, &dd_theta,
        nx, ny, nu, dp, nt, nb);
    
    // PThetaDCM

    dam_ptheta(
        &ptheta, &d_ptheta,
        &pd_ptheta, &dd_ptheta,
        nx, ny, nu, dp, nt, nb); 

    // Launch the kernel
    (*launcher)(
        d_x, d_y, d_u, 
        pd_theta, dd_theta, 
        pd_ptheta, dd_ptheta,
        nx, ny, nu, dp, nt, nb, d_errcode);

    // Get y back

    HANDLE_ERROR( cudaMemcpy(y, d_y,
        nx * ny * nt * nb * sizeof(double),
        cudaMemcpyDeviceToHost) );

    HANDLE_ERROR( cudaMemcpy(errcode, d_errcode, sizeof( unsigned int ), 
        cudaMemcpyDeviceToHost) );

    if ( *errcode != 0 ) 
        printf( "Error %d in %s at line %d\n", *errcode, __FILE__, __LINE__ );

    // free the memory allocated on the GPU
    //HANDLE_ERROR( cudaFree( d_x ) );
    HANDLE_ERROR( cudaFree( d_y ) );
    HANDLE_ERROR( cudaFree( d_u ) );
    HANDLE_ERROR( cudaFree( d_errcode ) );

    HANDLE_ERROR( cudaFree( pd_theta ) );
    HANDLE_ERROR( cudaFree( dd_theta ) );

    if ( DIM_PTHETA ) HANDLE_ERROR( cudaFree( pd_ptheta ) );
    if ( DIM_DPTHETA ) HANDLE_ERROR( cudaFree( dd_ptheta ) );
    
    return 0; 
}

// =======================================================================
// Externals
// =======================================================================

extern "C"
int
mpdcm_fmri_euler( double *x, double *y, double *u,
    void *theta, double *d_theta,
    void *ptheta, double *d_ptheta, 
    int nx, int ny, int nu, int dp, int nt, int nb)
{
   int r = mpdcm_fmri(x, y, u,
        theta, d_theta,
        ptheta, d_ptheta, 
        nx, ny, nu, dp, nt, nb,
        &ldcm_euler);
    
    return r;
};

extern "C"
int
mpdcm_fmri_kr4( double *x, double *y, double *u,
    void *theta, double *d_theta,
    void *ptheta, double *d_ptheta, 
    int nx, int ny, int nu, int dp, int nt, int nb)
{
    int r = mpdcm_fmri(x, y, u,
        theta, d_theta,
        ptheta, d_ptheta, 
        nx, ny, nu, dp, nt, nb,
        &ldcm_kr4);

    return r;
}

extern "C"
int
mpdcm_fmri_bs( double *x, double *y, double *u,
    void *theta, double *d_theta,
    void *ptheta, double *d_ptheta, 
    int nx, int ny, int nu, int dp, int nt, int nb)
{
    int r = mpdcm_fmri(x, y, u,
        theta, d_theta,
        ptheta, d_ptheta, 
        nx, ny, nu, dp, nt, nb,
        &ldcm_bs);

    return r;
};

// =======================================================================
// Integrators
// =======================================================================

// The memory layout is the following:

// If it is a 4 region DCM if work the following way:
// x_1, x_2, x_3, x_4, f_1, f_2, f_3, f_4, ....


// Euler

__device__ void dcm_upx_euler(dbuff ox, dbuff y, dbuff u, void *p_theta,
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

__device__ void dcm_upx_kr4(dbuff ox, dbuff y, dbuff u, void *p_theta,
     void *p_ptheta, dbuff nx)
{

    //ThetaDCM *theta = (ThetaDCM *) p_theta;
    PThetaDCM *ptheta = (PThetaDCM *) p_ptheta;

    int j = threadIdx.x%y.dim;
    int s;
    int maxx = threadIdx.x - y.dim * (blockDim.x / y.dim);
    // Buffers for the intermediate results. z is the estimated error.
    dbuff k1, k2;

    k1.arr = (ox.arr < nx.arr) ? ox.arr : nx.arr;
    k2.arr = k1.arr;
    k1.arr += ox.dim * DIM_X * 2;  
    k2.arr += ox.dim * DIM_X * 3;

    k1.dim = ox.dim;
    k2.dim = ox.dim;

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
        nx.arr[s] = ox.arr[s] + ptheta->de * k1.arr[s] * KRW1;
        k2.arr[s] = ox.arr[s] + ptheta->de * 0.5 * k1.arr[s]; 
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

        nx.arr[s] += ptheta->de * k1.arr[s] * KRW2;
    }

    __syncthreads();

    if ( maxx < 0 )
        k2.arr[s] = ox.arr[s] + ptheta->de * 0.5 * k1.arr[s]; 

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

        nx.arr[s] += ptheta->de * k1.arr[s]*KRW3;

    }

    __syncthreads();

    if ( maxx < 0 )
        k2.arr[s] = ox.arr[s] + ptheta->de * k1.arr[s];

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
        nx.arr[s] += ptheta->de * k1.arr[s] * KRW4;
    }
}


// Bogacki Shampine

__device__ void dcm_upx_bs(dbuff ox, dbuff y, dbuff u, void *p_theta,
     void *p_ptheta, dbuff nx, double *zs, thr_info tinfo)
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
    }

    __syncthreads();

    if ( maxx < 0 && threadIdx.y == 0) 
        zs[threadIdx.x] = z.arr[s];
//    if ( tinfo.cs >= tinfo.ns && threadIdx.y == 0 )
//        zs[threadIdx.x] = 0;
    // This is a serious hack
    if ( threadIdx.x >= z.dim && threadIdx.y == 0 )
        zs[threadIdx.x] = 0;

    __syncthreads();
    if ( threadIdx.x < 16 ) 
        zs[threadIdx.x] = zs[threadIdx.x] > zs[threadIdx.x + 16] ? 
            zs[threadIdx.x] : zs[threadIdx.x + 16];
    __syncthreads();
    if ( threadIdx.x < 8 ) 
        zs[threadIdx.x] = zs[threadIdx.x] > zs[threadIdx.x + 8] ? 
            zs[threadIdx.x] : zs[threadIdx.x + 8];
    __syncthreads();
   if ( threadIdx.x < 4 ) 
        zs[threadIdx.x] = zs[threadIdx.x] > zs[threadIdx.x + 4] ? 
            zs[threadIdx.x] : zs[threadIdx.x + 4];
    __syncthreads();
    if ( threadIdx.x < 2 ) 
        zs[threadIdx.x] = zs[threadIdx.x] > zs[threadIdx.x + 2] ? 
            zs[threadIdx.x] : zs[threadIdx.x + 2];
    __syncthreads();
    if ( threadIdx.x == 0 ) 
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

