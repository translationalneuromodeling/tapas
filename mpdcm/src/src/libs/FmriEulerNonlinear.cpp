/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */


#include "mpdcm.hcu"
#include "cuda/fmri_ext.hcu"
#include "libs/Fmri.hpp"
#include "libs/FmriEulerNonlinear.hpp"


namespace Host
{

FmriEulerNonlinear::FmriEulerNonlinear()
{
}

int
FmriEulerNonlinear::launch_kernel(const DataArray *y, const DataArray *u,
    const ThetaFmriArray *theta, const PThetaFmri *ptheta)
{
    cuda_fmri_euler_nonlinear(*y, *u, *theta, *ptheta);
}

} // FmriEuler
