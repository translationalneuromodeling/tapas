/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */

#include "mpdcm.hcu"
#include "mpdcm_Integrator.hcu"

__device__
Integrator::Integrator(int dims, int locs) :
    num_eqs(dims),
    preloc_size(locs)
{

}

__device__
int
Integrator::set_x(int i, DataArray *x, dbuff *bx)
{

    bx->arr = x->data_device + preloc_size * num_eqs * x->nic * 
        ( threadIdx.x / x->nic );

    return 0;
}

__device__
int
Integrator::set_u(int i, DataArray *u, dbuff *bu)
{
    bu->nc = u->nic;
    bu->nr = u->nir;

    bu->arr = u->data_device + ( i / (u->nc * u->nr)) * bu->nc * bu->nr;

    return 0;
}

__device__
int
Integrator::set_y(int i, DataArray *y, dbuff *by)
{

    by->nc = y->nic;
    by->nr = y->nir;
    
    by->arr = y->data_device + i * by->nc * by->nr;

    return 0;
}

__device__
int
Integrator::set_ptheta(int i, PTheta *aptheta)
{
    ptheta->dt = aptheta->dt;
    ptheta->dyu = aptheta->dyu;
    ptheta->de = aptheta->de;
    ptheta->mode = aptheta->mode;

    return 0;
}



__device__
int
Integrator::integrate(dbuff *x, dbuff *y, dbuff *u)
{
    int i;
    //int j = threadIdx.x%y.dim;
    MPFLOAT *t;
    // Number of integration steps done between each data point
    int ss, dy;
    // Number of data points
    int dp = y->nr;
    // Point where threads are not synchronized to anything
    int maxx = y->nc * ( blockDim.x / y->nc );

    dbuff ox[1];
    dbuff nx[1];

    dbuff ty[1];
    dbuff tu[1];

    ox->nc = y->nc;
    nx->nc = y->nc;

    ox->arr = x->arr; 
    nx->arr = ox->arr + nx->nc * num_eqs;

    
    if ( threadIdx.x < maxx )
        memset(x->arr, 0, nx->nc * num_eqs * sizeof(MPFLOAT));

    __syncthreads();
    
    ty->nc = y->nc;
    tu->nc = u->nc;

    ty->arr = y->arr; 
    tu->arr = u->arr;

    // How many samples are gonna be taken
    ss = ceil( 1.0 / ptheta->dt );
    dy = ceil( 1.0 / ( ptheta->dt * ptheta->dyu ) );
  
    for (i=0; i < dp*ss; i++)
    {
        if ( threadIdx.x < maxx )
            update_x(ox, ty, tu, nx);
        __syncthreads();
        // Only sample every 1/ptheta->dt times
        if ( i%dy == 0 ) 
        {
           if ( threadIdx.x < maxx )
               // Temporarily put the provisory results in the old x
               update_y(nx, ty, tu, ox);
            __syncthreads();
            ty->arr += y->nc; 
         }
        // Move one step forward
        if ( i % ss == 0 )
            tu->arr += u->nc;

        // Swap the pointers
        t = ox->arr;
        ox->arr = nx->arr;
        nx->arr = t;
    }

    return 0;
}
