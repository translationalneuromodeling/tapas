/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */

#include "mpdcm.hcu"
#include "fmri_FmriCuda.hcu"
#include "fmri_dynamics.hcu"

__device__
FmriCuda::FmriCuda(int dims, int locs) :
    num_eqs(dims),
    preloc_size(locs)
{

}

__device__
int
FmriCuda::update_y(dbuff *ox, dbuff *y, dbuff *u, dbuff *ny)
{

    fmri_upy(ox, y, u, theta, ptheta, ny);
    return 0;
}


__device__
int
FmriCuda::store_y(dbuff *nx, dbuff *y, dbuff *u, dbuff *ox)
{
    fmri_store_y(nx, y, u, theta, ptheta);
    return 0;
}


__device__
int
FmriCuda::set_x(int i, DataArray *x, dbuff *bx)
{

    bx->arr = x->data_device + preloc_size * num_eqs * x->nir * 
        ( threadIdx.x / x->nir );

    return 0;
}

__device__
int
FmriCuda::set_u(int i, DataArray *u, dbuff *bu)
{
    bu->nc = u->nic;
    bu->nr = u->nir;

    bu->arr = u->data_device +
        ((unsigned long int) i) * (unsigned long int) (bu->nc * bu->nr);

    return 0;
}

__device__
int
FmriCuda::set_y(int i, DataArray *y, dbuff *by)
{

    by->nc = y->nic;
    by->nr = y->nir;
    
    by->arr = y->data_device +
        ((unsigned long int) i) * (unsigned long int) (by->nc * by->nr);

    return 0;
}

__device__
int
FmriCuda::set_theta(int i, ThetaFmriArray *atheta)
{

    set_theta_fields(i, atheta);
    set_theta_linear(i, atheta); 
    set_theta_sparse(i, atheta);
    return 0;

}

__device__
int
FmriCuda::set_theta_fields(int i, ThetaFmriArray *atheta)
{

    ThetaFmri *ctheta = atheta->data_device + i;
    theta->dim_x = ctheta->dim_x;
    theta->dim_u = ctheta->dim_u;
    theta->fA  = ctheta->fA;
    theta->fB  = ctheta->fB;
    theta->fC  = ctheta->fC;
    theta->V0 = ctheta->V0;
    theta->E0 = ctheta->E0;
    theta->ln1_E0 = ctheta->ln1_E0;
    theta->lnE0 = ctheta->lnE0;
    theta->k1 = ctheta->k1;
    theta->k2 = ctheta->k2;
    theta->k3 = ctheta->k3;
    theta->alpha = ctheta->alpha;
    theta->gamma = ctheta->gamma;

    return 0;
}

__device__
int
FmriCuda::set_theta_linear(int i, ThetaFmriArray *atheta)
{
    MPFLOAT *dtheta = atheta->linear.data_device;

    int nx = theta->dim_x;
    int nu = theta->dim_u;
    unsigned long int o = nx * nx + nx * nu + nx + nx; // A, C, Kappa (K) tau
    
    dtheta += i * o;

    theta->A = dtheta;
    dtheta += nx * nx;

    theta->C = dtheta;
    dtheta += nx * nu;

    theta->K = dtheta;
    dtheta += nx;

    theta->tau = dtheta;

    return 0;
}

__device__
int
FmriCuda::set_theta_sparse(int i, ThetaFmriArray *atheta)
{
 
    int nx = theta->dim_x;
    int nu = theta->dim_u;

    theta->sB->i = atheta->sB.i_device;
    theta->sD->i = atheta->sD.i_device;

    theta->sB->v = atheta->sB.v_device;
    theta->sD->v = atheta->sD.v_device;

    theta->sB->j = atheta->sB.j_device + nx * nu * (unsigned long int ) i;
    theta->sD->j = atheta->sD.j_device + nx * nx * (unsigned long int ) i;

    return 0;
}

__device__
int
FmriCuda::set_ptheta(int i, PThetaFmri *aptheta)
{

    ptheta->dt = aptheta->dt;
    ptheta->dyu = aptheta->dyu;
    ptheta->de = aptheta->de;
    ptheta->mode = aptheta->mode;

    return 0;
} 

__device__
int
FmriCuda::integrate(dbuff *x, dbuff *y, dbuff *u)
{
    int i = 0;
    MPFLOAT *t;
    // Number of integration steps done between each data point
    int dy;
    // Number of data points
    int dp = u->nc;
    // Point where threads are not synchronized to anything
    int maxx = y->nr * ( blockDim.x / y->nr );

    dbuff ox[1];
    dbuff nx[1];

    dbuff ty[1];
    dbuff tu[1];

    ox->nr = y->nr;
    nx->nr = y->nr;

    ox->arr = x->arr; 
    nx->arr = ox->arr + nx->nr * num_eqs;

    __syncthreads();
    
    ty->nc = y->nc;
    tu->nr = u->nr;
    ty->nr = y->nr;

    ty->arr = y->arr; 
    tu->arr = u->arr;

    // How many samples are gonna be taken
    dy = ceil( 1.0 / ( ptheta->dt * ptheta->dyu ) );

    if ( threadIdx.x < maxx && threadIdx.y == 0)
        update_y(ox, ty, tu, nx);

    __syncthreads();
    // Tranfer the results 
    if ( threadIdx.x < maxx && threadIdx.y == 0)
        store_y(nx, ty, tu, ox);
    __syncthreads();

    ty->arr += y->nr; 
    
    __syncthreads();


    for ( i = 1; i < dp ; i++)
    {

        if ( threadIdx.x < maxx )
            update_x(ox, ty, tu, nx);
        __syncthreads();

        // Only sample every 1/ptheta->dt times
        if ( i % dy == 0 ) 
        {
            if ( threadIdx.x < maxx && threadIdx.y == 0)
            // Temporarily put the provisory results in the old x
                update_y(nx, ty, tu, ox);
            __syncthreads();
            // Tranfer the results 
            if ( threadIdx.x < maxx && threadIdx.y == 0)
                store_y(ox, ty, tu, nx);
            __syncthreads();

            ty->arr += y->nr; 
        }

        // Move one step forward
        tu->arr += u->nr;

        // Swap the pointers
        t = ox->arr;
        ox->arr = nx->arr;
        nx->arr = t;
    }

    return 0;
}


