/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */


#ifndef FMRI_STD_RK4_NONILINEAR_H
#define FMRI_STD_RK4_NONLINEAR_H

#include "libs/Fmri.hpp"
#include "mpdcm.hcu"

namespace Host
{

class FmriStandardRK4Nonlinear: public Fmri 
{
    public:
        FmriStandardRK4Nonlinear();

        int
        launch_kernel(const DataArray *y, const DataArray *u,
            const ThetaFmriArray *theta, const PThetaFmri *ptheta);

};


} // Host
#endif
