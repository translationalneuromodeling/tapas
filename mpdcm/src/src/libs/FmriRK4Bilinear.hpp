/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */


#ifndef FMRI_RK4_BILINEAR_H
#define FMRI_RK4_BILINEAR_H

#include "libs/Fmri.hpp"
#include "mpdcm.hcu"

namespace Host
{

class FmriRK4Bilinear: public Fmri 
{
    public:
        FmriRK4Bilinear();

        int
        launch_kernel(const DataArray *y, const DataArray *u,
            const ThetaFmriArray *theta, const PThetaFmri *ptheta);

};


} // Host
#endif
