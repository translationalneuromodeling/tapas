/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */

#include "mpdcm.hcu"
#include "fmri_mpdcm_Integrator.hcu"

__device__
Integrator::Integrator(int dims, int locs) :
    num_eqs(dims),
    preloc_size(locs)
{

}

__device__
int
Integrator::update_y(dbuff *x, dbuff *y, dbuff *u, dbuff *ny)
{

    return 0;
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



