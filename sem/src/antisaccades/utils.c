/* aponteeduardo@gmail.com */
/* copyright (C) 2016 */

#include "antisaccades.h"

// Populate an array

int
populate_parameters_prosa(const double *theta, PROSA_PARAMETERS *stheta)
{
    stheta->kp = theta[0];
    stheta->tp = theta[1];
    stheta->ka = theta[2];
    stheta->ta = theta[3];
    stheta->ks = theta[4];
    stheta->ts = theta[5];

    stheta->t0 = theta[6];
    stheta->da = theta[7];
    stheta->p0 = theta[8];

    return 0;
}


int
populate_parameters_seria(const double *theta, SERIA_PARAMETERS *stheta)
{
    stheta->kp = theta[0];
    stheta->tp = theta[1];

    stheta->ka = theta[2];
    stheta->ta = theta[3];

    stheta->ks = theta[4];
    stheta->ts = theta[5];

    stheta->kl = theta[6];
    stheta->tl = theta[7];

    stheta->t0 = theta[8];
    stheta->da = theta[9];
    stheta->p0 = theta[10];

    return 0;
}
