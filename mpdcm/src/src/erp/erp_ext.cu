/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */

#include "erp.hcu"
#include "mpdcm.hcu"
#include "erp_kernels.hcu"

int
cuda_erp_euler(DataArray y, DataArray u, ErpColumnArray theta,
    ErpPTheta ptheta)
{

    int device;
    cudaGetDevice(&device);

    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    
    int num_blocks = 10;
    /*
        min((y.nc * theta.nc * theta.nr + (
            ERP_DIM_BLOCKS_X * ERP_DIM_BLOCKS_Y) - 1)/
            (ERP_DIM_BLOCKS_X * ERP_DIM_BLOCKS_Y),
            NUM_BLOCKS * props.multiProcessorCount);
    */

    dim3 gthreads(ERP_DIM_BLOCKS_X, ERP_DIM_BLOCKS_Y);
    dim3 gblocks(num_blocks, 1);


    kernel_erp_euler<<<gblocks, gthreads>>>(y, u, theta, ptheta);

    return 0;
}
/*
int
cuda_erp_rk4(DataArray y, DataArray u, ThetaFmriArray theta,
    PThetaFmri ptheta)
{

    int device;
    cudaGetDevice(&device);

    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    int num_blocks =
        min((y.nc * theta.nc * theta.nr + NUM_THREADS - 1)/NUM_THREADS,
            NUM_BLOCKS * props.multiProcessorCount);


    dim3 gthreads(NUM_THREADS, DIM_FMRI_X);
    dim3 gblocks(num_blocks, 1);

    kernel_erp_rk4<<<gblocks, gthreads>>>(y, u, theta, ptheta);

    return 0;
}
*/
