/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */


#include "mpdcm.hcu"
#include "cuda/fmri_ext.hcu"
#include "libs/Fmri.hpp"
#include "libs/FmriRK4.hpp"

namespace Host
{

FmriRK4::FmriRK4()
{
}

int
FmriRK4::launch_kernel(const DataArray *y, const DataArray *u,
    const ThetaFmriArray *theta, const PThetaFmri *ptheta)
{
    cuda_fmri_rk4(*y, *u, *theta, *ptheta);
}

} // Host
