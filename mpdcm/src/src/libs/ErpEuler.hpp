/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */

    
#ifndef ERPEULER_H
#define ERPEULER_H
#include <iostream>
#include <matrix.h>
#include <mex.h>
#include <math.h>
#include "mpdcm.hcu"
#include "erp.hcu"
#include "Erp.hpp"


namespace Host
{
class ErpEuler : public Erp
{
public:
	int
	launch_kernel(DataArray *y, DataArray *u, ErpColumnArray *erpArray, ErpPTheta *ptheta);

};


} // Host

#endif // ERPEULER_H
