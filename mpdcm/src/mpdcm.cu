/* aponteeduardo@gmail.com */
/* copyright (C) 2014 */


#include "mpdcm.hcu"

#define DIM_PTHETA 3
#define DIM_DPTHETA 0

#define DIM_X 5 
#define INDEX_X 0
#define INDEX_F 1
#define INDEX_S 2
#define INDEX_V 3
#define INDEX_Q 4

#define PRELOC_SIZE_X 2

#define DIM_THETA 7

#define INDEX_V0 0
#define INDEX_E0 1
#define INDEX_K1 2
#define INDEX_K2 3
#define INDEX_K3 4
#define INDEX_ALPHA 5
#define INDEX_GAMMA 6

#define NUM_THREADS 64

typedef struct
{
    int dim;
    double *arr;
} dbuff;

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
    PThetaDCM *ptheta = (PThetaDCM  *) p_ptheta;

    o = INDEX_X * x.dim;


    switch ( ptheta->mode )
    {
        case ( 'f' ):
            // Very innefficient 
            // A
            if ( theta->fA == MF_TRUE )
            {
                for (j = 0; j < x.dim; j++){
                    dx += theta->A[i + x.dim*j] * x.arr[o + j];
                }
            }
            // B
            if ( theta->fB == MF_TRUE )
            {
                for (j = 0; j < u.dim; j++)
                {
                    if (u.arr[j] == 0)
                        continue;
                    bt = 0;
                    k = x.dim*x.dim*j + i;
                    for (p = 0; p < x.dim; p++){
                        bt += theta->B[k + x.dim*p] * x.arr[o + p];
                    }
                    dx += bt*u.arr[j];
                }
            }
            // C
            if ( theta->fC == MF_TRUE )
            {
                for (j = 0; j < u.dim; j++)
                {
                    dx += theta->C[i + x.dim*j] * u.arr[j];
                }
            }         
        break;
        case ( 'c' ):

            k = x.dim * i;
            // A
            if ( theta->fA == MF_TRUE )
            {
                for (j = 0; j < x.dim; j++)
                {
                    dx += theta->A[k + j] * x.arr[o + j];
                }
            }
            // B
            if (theta->fB == MF_TRUE )
            {
                for (j = 0; j < u.dim; j++)
                {
                    if (u.arr[j] == 0)
                        continue;
                    bt = 0;
                    k = x.dim*x.dim*j + x.dim * i;
                    for (p = 0; p < x.dim; p++){
                        bt += theta->B[k + p] * x.arr[o + p];
                    }
                    dx += bt*u.arr[j];
                }
            }
            // C
            k = i*u.dim;
            if ( theta->fC == MF_TRUE )
            {
                for (j = 0; j < u.dim; j++)
                {
                    dx += theta->C[k + j] * u.arr[j];
                }
            }            
        break;
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
        exp(theta->K[i]) * x.arr[INDEX_S * x.dim + i] -
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

    dv = exp(x.arr[INDEX_F * x.dim + i] - x.arr[INDEX_V * x.dim + i]) -
        exp(x.arr[INDEX_V * x.dim + i] * theta->alpha);

    return dv/exp(theta->tau[i]);
}

__device__ double dcm_dq(dbuff x, dbuff y, dbuff u, void *p_theta, 
    void *p_ptheta, int i)
{
    double dq = 0;
    double f = exp(x.arr[INDEX_F * x.dim + i]);
    double v;
    double lnE0; 
    ThetaDCM *theta = (ThetaDCM *) p_theta;
    
    v = exp(x.arr[INDEX_V * x.dim + i] * theta->alpha);
    lnE0 = theta->lnE0;

    //    PThetaDCM *ptheta = (PThetaDCM  *) p_ptheta;

    dq = exp(x.arr[INDEX_F * x.dim + i] - x.arr[INDEX_Q * x.dim + i] - lnE0) -
        (exp(x.arr[INDEX_F * x.dim + i] + theta->ln1_E0/f - 
            x.arr[INDEX_Q * x.dim + i] - lnE0))  -  v;

    return dq;
}

__device__ double dcm_l(dbuff x, dbuff y, dbuff u, void *p_theta, 
    void *p_ptheta, int i)
{
    double l;
    double q = exp(x.arr[INDEX_Q * x.dim + i]);
    double v = exp(x.arr[INDEX_V * x.dim + i]);

    ThetaDCM *theta = (ThetaDCM *) p_theta;
    //PThetaDCM *ptheta = (PThetaDCM *) p_ptheta;

    l = theta->V0 * ( 
        theta->k1 * ( 1 - q) +
        theta->k2 * ( 1 - q/v) +
        theta->k3 * ( 1 - v)); 
    return l;
}

__device__ void dcm_upx(dbuff ox, dbuff y, dbuff u, void *p_theta,
     void *p_ptheta, dbuff nx)
{

    //ThetaDCM *theta = (ThetaDCM *) p_theta;
    PThetaDCM *ptheta = (PThetaDCM *) p_ptheta;

    int j = threadIdx.x%y.dim;
    int ss;
    double dt;

    // Make the values to be closer in range

    ss = ceil(1.0/(ptheta->dt * ptheta->dyu));
    dt = 1.0/((double ) ss);
 
    if ( isnan( *u.arr ) ){
        if ( threadIdx.y == 0 )
        {
            nx.arr[ INDEX_X * ox.dim + j] = NAN;
            nx.arr[ INDEX_F * ox.dim + j] = NAN;
            nx.arr[ INDEX_S * ox.dim + j] = NAN;
            nx.arr[ INDEX_V * ox.dim + j] = NAN;
            nx.arr[ INDEX_Q * ox.dim + j] = NAN;
        }
    }

    switch ( threadIdx.y )
    {
        case INDEX_X:
            nx.arr[INDEX_X * ox.dim + j] = ox.arr[ INDEX_X * ox.dim + j] + 
                dt * dcm_dx(ox, y, u, p_theta, p_ptheta, j);
            break;
        case INDEX_F:
            nx.arr[ INDEX_F * ox.dim + j] = ox.arr[ INDEX_F * ox.dim + j] + 
                dt * dcm_df(ox, y, u, p_theta, p_ptheta, j);
            break;
        case INDEX_S:
            nx.arr[ INDEX_S * ox.dim + j] = ox.arr[ INDEX_S * ox.dim + j] + 
                dt * dcm_ds(ox, y, u, p_theta, p_ptheta, j);
            break;
        case INDEX_V:
            nx.arr[ INDEX_V * ox.dim + j] = ox.arr[ INDEX_V * ox.dim + j] + 
                dt * dcm_dv(ox, y, u, p_theta, p_ptheta, j);
            break;
        case INDEX_Q:
            nx.arr[ INDEX_Q * ox.dim + j] = ox.arr[ INDEX_Q * ox.dim + j] + 
                dt * dcm_dq(ox, y, u, p_theta, p_ptheta, j); 
            break;
    }

}

__device__ void dcm_upy(dbuff x, dbuff y, dbuff u, void *theta,
     void *ptheta)
{
    // Network node
    int nn = threadIdx.x%y.dim;
    if ( isnan( *u.arr ) )
    {
        y.arr[nn] = NAN;
    } else 
    {
        y.arr[nn] = dcm_l(x, y, u, theta, ptheta, nn);
    }
}

__device__ void dcm_int(dbuff x, dbuff y, dbuff u, void *p_theta,
    void *p_ptheta, int dp)
{
    int i;
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

    ox.dim = y.dim;
    nx.dim = y.dim;

    ox.arr = x.arr; 
    nx.arr = ox.arr + nx.dim * DIM_X;

    if ( threadIdx.x < maxx )
        memset(x.arr, 0, nx.dim * DIM_X * sizeof(double));
    //if ( threadIdx.x < maxx )
    //    ox.arr[threadIdx.y + DIM_X * threadIdx.x%y.dim] = 0;

    __syncthreads();
    ty.dim = y.dim;
    tu.dim = u.dim;

    // How many samples are gonna be taken
    ss = ceil(1.0/ptheta->dt);
    dy = ceil(1.0/(ptheta->dt * ptheta->dyu));

    ty.arr = y.arr; 
    tu.arr = u.arr;

    for (i=0; i < dp*ss; i++)
    {
        if ( threadIdx.x < maxx )
            dcm_upx(ox, ty, tu, p_theta, p_ptheta, nx);
        __syncthreads();
        // Only sample every 1/ptheta->dt times
        if ( i%ss == 0 )
        {
            if ( i%dy == 0 ) 
            {
                if ( threadIdx.x < maxx && threadIdx.y == 0 )
                {
                    dcm_upy(nx, ty, tu, p_theta, p_ptheta);
                }
                ty.arr += y.dim; 
            }
            if ( i > 0 )
                tu.arr += u.dim;
        }
        __syncthreads();
        // Swap the pointers
        t = ox.arr;
        ox.arr = nx.arr;
        nx.arr = t;
    }
}

// Kernel code
__global__ void kdcm_fmri(double *x, double *y, double *u, 
    void *p_theta, double *d_theta, void *p_ptheta, double *d_ptheta, 
    int nx, int ny, int nu, int dp, int nt, int nb )
{
    
    int i;
    dbuff tx, ty, tu;
    __shared__ double sx[NUM_THREADS * DIM_X * PRELOC_SIZE_X];

    // Assign pointers to theta


    ThetaDCM *theta = (ThetaDCM *) p_theta;
    ThetaDCM *ltheta;

    PThetaDCM *ptheta = (PThetaDCM *) p_ptheta;
    __shared__ PThetaDCM lptheta[1];

    lptheta->dt = ptheta->dt;
    lptheta->dyu = ptheta->dyu;
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

        tx.arr = sx + PRELOC_SIZE_X * DIM_X * nx * (threadIdx.x/nx);

        ty.arr = y + i * nx * ny;

        dcm_int(tx, ty, tu, (void *) ltheta, (void *) lptheta, dp);
        i += gridDim.x * (blockDim.x / nx );        
    }
}

// Kernel (l)auncher

__host__ void ldcm_fmri(double *x, double *y, double *u, 
    void *theta, double *d_theta, void *ptheta, double *d_ptheta, 
    int nx, int ny, int nu, int dp, int nt, int nb )
{

    dim3 gthreads(NUM_THREADS, DIM_X);
    dim3 gblocks(8, 1);

    kdcm_fmri<<<gblocks, gthreads>>>(x, y, u, 
        theta, d_theta, ptheta, d_ptheta, 
        nx, ny, nu, dp, nt, nb ); 
}


// TODO the function signature is not ideal...
// Device alloc memory theta
__host__ void dam_theta(
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
__host__ void dam_ptheta(
    void **ptheta, double **d_ptheta, 
    void **pd_ptheta, double **dd_ptheta, 
    int nx, int ny, int nu, int dp, int nt, int nb)
{

    HANDLE_ERROR( cudaMalloc( pd_ptheta, sizeof(PThetaDCM)));
    HANDLE_ERROR( cudaMemcpy( *pd_ptheta, *ptheta, sizeof(PThetaDCM),
        cudaMemcpyHostToDevice ) );
 
}

// Host code
extern "C"
int mpdcm_fmri( double *x, double *y, double *u,
    void *theta, double *d_theta,
    void *ptheta, double *d_ptheta, 
    int nx, int ny, int nu, int dp, int nt, int nb)
{

    double *d_x, *d_y, *d_u;
    void *pd_theta, *pd_ptheta;
    double *dd_theta, *dd_ptheta;


    // x

    d_x = 0;
    //HANDLE_ERROR( cudaMalloc( (void **) &d_x, 
    //    nx * DIM_X * PRELOC_SIZE_X  * nt * nb * sizeof(double) ) );

    // y

    HANDLE_ERROR( cudaMalloc( (void **) &d_y,
        nx * ny * nt * nb * sizeof(double) ) );

    // u

    HANDLE_ERROR( cudaMalloc( (void**) &d_u,
        nt * nu * dp *  sizeof(double) ) );
    HANDLE_ERROR( cudaMemcpy( d_u, u, nt * nu * dp * sizeof(double),
        cudaMemcpyHostToDevice ) );

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
    ldcm_fmri(
        d_x, d_y, d_u, 
        pd_theta, dd_theta, 
        pd_ptheta, dd_ptheta,
        nx, ny, nu, dp, nt, nb );

    // Get y back

    HANDLE_ERROR( cudaMemcpy(y, d_y,
        nx * ny * nt * nb * sizeof(double),
        cudaMemcpyDeviceToHost) );


    // free the memory allocated on the GPU
    //HANDLE_ERROR( cudaFree( d_x ) );
    HANDLE_ERROR( cudaFree( d_y ) );
    HANDLE_ERROR( cudaFree( d_u ) );

    HANDLE_ERROR( cudaFree( pd_theta ) );
    HANDLE_ERROR( cudaFree( dd_theta ) );

    if ( DIM_PTHETA ) HANDLE_ERROR( cudaFree( pd_ptheta ) );
    if ( DIM_DPTHETA ) HANDLE_ERROR( cudaFree( dd_ptheta ) );
    
    return 0; 
}
