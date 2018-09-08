/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */

#include "erp_ErpEuler.hcu"
//#include "erp_FmriRK4.hcu"
#include "kernels.hcu"
#include "erp_kernels.hcu"
#include "erp.hcu"
#include "mpdcm.hcu"

__global__
void
kernel_erp_euler(DataArray y, DataArray u, ErpColumnArray theta, 
        ErpPTheta ptheta)
{
    
    __shared__ MPFLOAT sx[
    ERP_EULER_PRELOC_SIZE_X * ERP_DIM_BLOCKS_X * ERP_DIM_X];

    ErpEuler erp;
    
    DataArray x;
    
    x.data_device = sx;
    x.nc = y.nc;
    x.nr = y.nr;

    x.nic = theta.dim_x; // The number of columns is the number of
    // number of cortical columns (?)
    x.nir = erp.preloc_size; // So what the fuck is this?
    
    kernel_launcher(x, y, u, theta, ptheta, erp);

}

/*
__global__
void
kernel_erp_rk4(DataArray y, DataArray u, ThetaFmriArray theta, PThetaFmri ptheta)
{
    
    __shared__ MPFLOAT sx[
    PRELOC_SIZE_FMRI_RK4_X * NUM_THREADS * DIM_FMRI_X];
    //__shared__ MPFLOAT shA[1024];
    //__shared__ MPFLOAT shC[320];


    FmriRK4 erp;
    
    DataArray x;
    
    x.data_device = sx;
    x.nc = y.nc;
    x.nr = y.nr;

    x.nic = y.nic;
    x.nir = erp.preloc_size;

    kernel_launcher(x, y, u, theta, ptheta, erp);

}

*/
