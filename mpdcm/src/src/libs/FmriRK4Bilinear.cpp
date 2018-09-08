/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */


#include "mpdcm.hcu"
#include "cuda/fmri_ext.hcu"
#include "libs/FmriRK4Bilinear.hpp"

namespace Host
{

FmriRK4Bilinear::FmriRK4Bilinear()
{
}

int
FmriRK4Bilinear::launch_kernel(const DataArray *y, const DataArray *u,
    const ThetaFmriArray *theta, const PThetaFmri *ptheta)
{
    cuda_fmri_rk4_bilinear(*y, *u, *theta, *ptheta);
}

} // Host
