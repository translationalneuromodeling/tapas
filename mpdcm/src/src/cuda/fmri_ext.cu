/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */

#include "mpdcm.hcu"
#include "fmri.hcu"
#include "fmri_ext.hcu"
#include "fmri_kernels.hcu"
#include "fmri_FmriEuler.hcu"
#include "fmri_FmriRK4.hcu"
#include "fmri_FmriRK4Bilinear.hcu"
#include "fmri_FmriStandardRK4Bilinear.hcu"
#include "fmri_FmriRK4Nonlinear.hcu"
#include "fmri_FmriStandardRK4Nonlinear.hcu"
#include "fmri_FmriEulerLarge.hcu"
#include "fmri_FmriEulerBilinear.hcu"
#include "fmri_FmriEulerNonlinear.hcu"

int
cuda_fmri_euler(DataArray y, DataArray u, ThetaFmriArray theta,
    PThetaFmri ptheta)
{

    cuda_template_kernel<FmriEuler, FMRI_EULER_PRELOC_X>(y, u, theta, ptheta);
    
    return 0;
}

int
cuda_fmri_euler_large(DataArray y, DataArray u, ThetaFmriArray theta,
    PThetaFmri ptheta)
{

    
    cuda_template_kernel<FmriEulerLarge, FMRI_EULER_PRELOC_X>(y, u, theta, 
            ptheta);

    return 0;
}

int
cuda_fmri_euler_bilinear(DataArray y, DataArray u, ThetaFmriArray theta,
    PThetaFmri ptheta)
{

    cuda_template_kernel<FmriEulerBilinear, FMRI_EULER_PRELOC_X>(y, 
            u, theta, ptheta);

    return 0;
}

int
cuda_fmri_euler_nonlinear(DataArray y, DataArray u, ThetaFmriArray theta,
    PThetaFmri ptheta)
{

    cuda_template_kernel<FmriEulerNonlinear, FMRI_EULER_PRELOC_X>(y, u, 
            theta, ptheta);

    return 0;
}


int
cuda_fmri_rk4(DataArray y, DataArray u, ThetaFmriArray theta,
    PThetaFmri ptheta)
{

    cuda_template_kernel<FmriRK4, FMRI_RK4_PRELOC_X>(y, u, 
            theta, ptheta);

    return 0;
}

int
cuda_fmri_rk4_bilinear(DataArray y, DataArray u, ThetaFmriArray theta,
    PThetaFmri ptheta)
{

    cuda_template_kernel<FmriRK4Bilinear, FMRI_RK4_PRELOC_X>(y, u, 
            theta, ptheta);

    return 0;
}

int
cuda_fmri_std_rk4_bilinear(DataArray y, DataArray u, ThetaFmriArray theta,
    PThetaFmri ptheta)
{

    cuda_template_kernel<FmriStandardRK4Bilinear, FMRI_STD_RK4_PRELOC_X>(y, u, 
            theta, ptheta);

    return 0;
}

int
cuda_fmri_std_rk4_nonlinear(DataArray y, DataArray u, ThetaFmriArray theta,
    PThetaFmri ptheta)
{

    cuda_template_kernel<FmriStandardRK4Nonlinear, FMRI_STD_RK4_PRELOC_X>(y, u, 
            theta, ptheta);

    return 0;
}

int
cuda_fmri_rk4_nonlinear(DataArray y, DataArray u, ThetaFmriArray theta,
    PThetaFmri ptheta)
{

    cuda_template_kernel<FmriRK4Nonlinear, FMRI_RK4_PRELOC_X>(y, u, 
            theta, ptheta);

    return 0;
}
