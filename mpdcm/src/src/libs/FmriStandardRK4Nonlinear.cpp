/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */


#include "mpdcm.hcu"
#include "cuda/fmri_ext.hcu"
#include "libs/FmriStandardRK4Nonlinear.hpp"

namespace Host
{

FmriStandardRK4Nonlinear::FmriStandardRK4Nonlinear()
{
}

int
FmriStandardRK4Nonlinear::launch_kernel(const DataArray *y, const DataArray *u,
    const ThetaFmriArray *theta, const PThetaFmri *ptheta)
{
    cuda_fmri_std_rk4_nonlinear(*y, *u, *theta, *ptheta);
}

} // Host
