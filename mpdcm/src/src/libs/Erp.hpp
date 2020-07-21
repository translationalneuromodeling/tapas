/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */

    
#ifndef ERP_H
#define ERP_H
#include <iostream>
#include <matrix.h>
#include <mex.h>
#include <math.h>
#include "mpdcm.hcu"
#include "erp.hcu"
#include "Interface.hpp"





namespace Host
{
class Erp : public Interface
{
public:
	Erp( );

	~Erp( );

	int
	transverse_theta_array(const mxArray *matcell, ErpColumnArray *erpArray);

	int
	transverse_theta_A13(const mxArray *matcell, ErpColumnArray *erpArray);

	int
	transverse_theta_A23(const mxArray *matcell, ErpColumnArray *erpArray);

	int
	transverse_theta(const mxArray *matcell, ErpColumnArray *erpArray);

	int
	linearize_theta_fields(const mxArray *matstruct, ErpColumn *erpColumn);

	int
	linearize_theta_array(const mxArray *matcell, ErpColumnArray *erpArray);

	int
	linearize_theta(const mxArray *matcell, ErpColumnArray *erpArray);
    
    int
    linearize_theta_vector(const mxArray *matcell, ErpColumnArray *erpArray,
         const char field[], int dvector, MPFLOAT *tvector);
	int
	initialize_y(DataArray *u, ErpColumnArray *theta, ErpPTheta *ptheta, 
        DataArray *y);

	int
	set_device_theta(ErpColumnArray *erpArray);

	int
	set_host_theta(ErpColumnArray *erpArray);

	int
	clean_host_theta(ErpColumnArray *erpArray);

	int
	clean_device_theta(ErpColumnArray *erpArray);

	int
	transfer_device_theta(ErpColumnArray *erpArray);

	int
	transverse_ptheta(const mxArray *matstruct, ErpPTheta *ptheta);

};


} // Host

#endif // ERP_H
