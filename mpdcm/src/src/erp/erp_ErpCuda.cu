/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */


#include "erp_ErpCuda.hcu"


__device__
ErpCuda::ErpCuda() : Integrator(ERP_DIM_X, ERP_EULER_PRELOC_SIZE_X)
{

}

__device__
int
ErpCuda::update_y(dbuff *x, dbuff *y, dbuff *u, dbuff *ny)
{

    return 0;
}

__device__
int
ErpCuda::store_y(dbuff *x, dbuff *y, dbuff *u, dbuff *ny)
{

    int j = threadIdx.x % y->nc;

    y->arr[j] = x->arr[j * ERP_DIM_X + 1];
        //erp_saturation(x->arr[j * ERP_DIM_X], theta);

    return 0;
}


__device__
int
ErpCuda::set_theta(int i, ErpColumnArray *atheta)
{

    set_theta_fields(i, atheta);
    set_theta_A(i, atheta);

    return 0;
}

__device__
int
ErpCuda::set_theta_fields(int i, ErpColumnArray *atheta)
{

    i = threadIdx.x + blockIdx.x * blockDim.x / atheta->dim_x;

    theta->dim_x = atheta->data_device[i].dim_x;
    theta->dim_u = atheta->data_device[i].dim_u;
    theta->Au = atheta->data_device[i].Au;
    theta->gamma1 = atheta->data_device[i].gamma1;
    theta->gamma2 = atheta->data_device[i].gamma2;
    theta->gamma3 = atheta->data_device[i].gamma3;
    theta->gamma4 = atheta->data_device[i].gamma3;
    theta->r1 = atheta->data_device[i].r1;
    theta->r2 = atheta->data_device[i].r2;
    theta->er1r2 = atheta->data_device[i].er1r2;
    theta->tau_e2 = atheta->data_device[i].tau_e2;
    theta->tau_es2 = atheta->data_device[i].tau_es2;
    theta->tau_i2 = atheta->data_device[i].tau_i2;
    theta->tau_is2 = atheta->data_device[i].tau_is2;

    return 0;    
}


__device__
int
ErpCuda::set_theta_A(int i, ErpColumnArray *erpArray)
{
    // Can be largely optimize
    
    int dx = erpArray->dim_x; 

    int offset = dx * ( threadIdx.x + blockIdx.x * blockDim.x / dx);

    theta->A13 = erpArray->linear_A13.data_device + offset;
    theta->A23 = erpArray->linear_A23.data_device + offset;

    return 0;
}

