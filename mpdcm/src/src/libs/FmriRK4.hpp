/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */


#ifndef FMRIRK4_H
#define FMRIRK4_H

#include "libs/Fmri.hpp"
#include "mpdcm.hcu"

namespace Host
{

class FmriRK4: public Fmri 
{
    public:
        FmriRK4();

        int
        launch_kernel(const DataArray *y, const DataArray *u,
            const ThetaFmriArray *theta, const PThetaFmri *ptheta);

};


} // Host
#endif
