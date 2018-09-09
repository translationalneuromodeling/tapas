/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */


#ifndef FMRI_STD_RK4_BILINEAR_H
#define FMRI_STD_RK4_BILINEAR_H

#include "libs/Fmri.hpp"
#include "mpdcm.hcu"

namespace Host
{

class FmriStandardRK4Bilinear: public Fmri 
{
    public:
        FmriStandardRK4Bilinear();

        int
        launch_kernel(const DataArray *y, const DataArray *u,
            const ThetaFmriArray *theta, const PThetaFmri *ptheta);

};


} // Host
#endif
