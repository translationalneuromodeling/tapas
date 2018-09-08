/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */


#include "mpdcm.hcu"
#include "cuda/fmri_ext.hcu"
#include "libs/FmriRK4Nonlinear.hpp"

namespace Host
{

FmriRK4Nonlinear::FmriRK4Nonlinear()
{
}

int
FmriRK4Nonlinear::launch_kernel(const DataArray *y, const DataArray *u,
    const ThetaFmriArray *theta, const PThetaFmri *ptheta)
{
    cuda_fmri_rk4_nonlinear(*y, *u, *theta, *ptheta);
}

} // Host
