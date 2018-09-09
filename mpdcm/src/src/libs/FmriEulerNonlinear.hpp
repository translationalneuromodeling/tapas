/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */


#ifndef FMRI_EULER_NONLINEAR_H
#define FMRI_EULER_NONLINEAR_H

#include "libs/Fmri.hpp"
#include "mpdcm.hcu"

namespace Host
{


class FmriEulerNonlinear : public Fmri 
{
    public:
        FmriEulerNonlinear();

        int
        launch_kernel(const DataArray *y, const DataArray *u,
            const ThetaFmriArray *theta, const PThetaFmri *ptheta);

};

} // Host

#endif
