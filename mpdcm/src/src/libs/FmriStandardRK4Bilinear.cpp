/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */


#include "mpdcm.hcu"
#include "cuda/fmri_ext.hcu"
#include "libs/FmriStandardRK4Bilinear.hpp"

namespace Host
{

FmriStandardRK4Bilinear::FmriStandardRK4Bilinear()
{
}

int
FmriStandardRK4Bilinear::launch_kernel(const DataArray *y, const DataArray *u,
    const ThetaFmriArray *theta, const PThetaFmri *ptheta)
{
    cuda_fmri_std_rk4_bilinear(*y, *u, *theta, *ptheta);
}

} // Host
