/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */

    
#include "Erp.hpp"
#include "cuda.h"
#include "cuda_runtime.h"

namespace Host
{
Erp::Erp( )
{

}

Erp::~Erp( )
{

}

int
Erp::transverse_theta_array(const mxArray *matcell, ErpColumnArray *erpArray)
{


    unsigned int i, o, nx, nu;
    const mwSize *st = mxGetDimensions(matcell);
    mxArray *ttheta;

    erpArray->nc = st[0];
    erpArray->nr = st[1];

    erpArray->dim_x = (int ) mxGetPr(mxGetField(mxGetCell(matcell, 0), 
        0, "dim_x"))[0];
    erpArray->dim_u = (int ) mxGetPr(mxGetField(mxGetCell(matcell, 0), 
        0, "dim_u"))[0];


    return 0;
 
}

int
Erp::transverse_theta_A13(const mxArray *matcell, ErpColumnArray *erpArray)
{

    erpArray->linear_A13.nc = mxGetDimensions(matcell)[0];
    erpArray->linear_A13.nr = mxGetDimensions(matcell)[1];

    erpArray->linear_A13.nic = erpArray->dim_x;
    erpArray->linear_A13.nir = erpArray->dim_x;

    return 0;

        
}

int
Erp::transverse_theta_A23(const mxArray *matcell, ErpColumnArray *erpArray)
{

    erpArray->linear_A23.nc = mxGetDimensions(matcell)[0];
    erpArray->linear_A23.nr = mxGetDimensions(matcell)[1];

    erpArray->linear_A23.nic = erpArray->dim_x; 
    erpArray->linear_A23.nir = erpArray->dim_x;

    return 0;

        
}

int
Erp::transverse_theta(const mxArray *matcell, ErpColumnArray *erpArray)
{

    transverse_theta_array(matcell, erpArray);
    transverse_theta_A13(matcell, erpArray);
    transverse_theta_A23(matcell, erpArray);

}

int
Erp::linearize_theta(const mxArray *matcell, ErpColumnArray *erpArray)
{
    linearize_theta_array(matcell, erpArray);
    
    linearize_theta_vector(matcell, erpArray, "A13", erpArray->dim_x,
            erpArray->linear_A13.data_host);
    linearize_theta_vector(matcell, erpArray, "A23", erpArray->dim_x,
            erpArray->linear_A23.data_host);
            
}

int
Erp::linearize_theta_fields(const mxArray *matstruct, ErpColumn *erpColumn)
{
	erpColumn->dim_x = (int ) *mxGetPr(mxGetField(matstruct, 0, "dim_x"));
	erpColumn->dim_u = (int ) *mxGetPr(mxGetField(matstruct, 0, "dim_u"));
	erpColumn->Au = (MPFLOAT ) *mxGetPr(mxGetField(matstruct, 0, "Au"));
	erpColumn->gamma1 = (MPFLOAT ) *mxGetPr(mxGetField(matstruct, 0, "gamma1"));
	erpColumn->gamma2 = (MPFLOAT ) *mxGetPr(mxGetField(matstruct, 0, "gamma2"));
	erpColumn->gamma3 = (MPFLOAT ) *mxGetPr(mxGetField(matstruct, 0, "gamma3"));
	erpColumn->gamma4 = (MPFLOAT ) *mxGetPr(mxGetField(matstruct, 0, "gamma4"));
	erpColumn->r1 = (MPFLOAT ) *mxGetPr(mxGetField(matstruct, 0, "r1"));
	erpColumn->r2 = (MPFLOAT ) *mxGetPr(mxGetField(matstruct, 0, "r2"));
	erpColumn->er1r2 = (MPFLOAT ) *mxGetPr(mxGetField(matstruct, 0, "er1r2"));
	erpColumn->tau_e2 = (MPFLOAT ) *mxGetPr(mxGetField(matstruct, 0, "tau_e2"));
	erpColumn->tau_es2 = (MPFLOAT ) *mxGetPr(mxGetField(matstruct, 0, "tau_es2"));
	erpColumn->tau_i2 = (MPFLOAT ) *mxGetPr(mxGetField(matstruct, 0, "tau_i2"));
	erpColumn->tau_is2 = (MPFLOAT ) *mxGetPr(mxGetField(matstruct, 0, "tau_is2"));

return 0;

}

int
Erp::linearize_theta_array(const mxArray *matcell, ErpColumnArray *erpArray)
{

    int i;
    int l;
    int t = 0;

    for (i = 0; i < erpArray->nc * erpArray->nr; i++)
    {
        mxArray *columns = mxGetField(mxGetCell(matcell, i), 0, "columns");
        for (l = 0; l < erpArray->dim_x; l++)
        {
            linearize_theta_fields(mxGetCell(columns, l), 
                    erpArray->data_host + t);
            t++;
        }
    }

    return 0;
        
}

int
Erp::linearize_theta_vector(const mxArray *matcell, ErpColumnArray *erpArray,
        const char field[], int dvector, MPFLOAT *tvector)
{
    int i;
    int l;
    int j;
    double *vals;

    for (i = 0; i < erpArray->nc * erpArray->nr; i++)
    {
        mxArray *columns = mxGetField(mxGetCell(matcell, i), 0, "columns");
        for (l = 0; l < erpArray->dim_x; l++)
        {
            vals = mxGetPr(mxGetField(mxGetCell(columns, l), 0, field));
            for (j = 0; j < dvector; j ++)
                tvector[j] = vals[j];
            tvector += dvector;
        }
    }

    return 0;

}

int
Erp::initialize_y(DataArray *u, ErpColumnArray *theta, ErpPTheta *ptheta, DataArray *y)
{

    y->nc = theta->nc;
    y->nr = theta->nr;

    y->nic = theta->dim_x;
    y->nir = ceil(u->nir * ptheta->dyu);

    return 0;
    
}

int
Erp::set_device_theta(ErpColumnArray *erpArray)
{


    erpArray->data_host = new ErpColumn[erpArray->nc * erpArray->nr *
        erpArray->dim_x];

        
	set_host_array(&(erpArray->linear_A13));
	set_host_array(&(erpArray->linear_A23));
	return 0;

}

int
Erp::set_host_theta(ErpColumnArray *erpArray)
{


    HANDLE_CERROR(cudaMalloc(&(erpArray->data_device), 
        erpArray->nc * erpArray->nr * erpArray->dim_x *
        sizeof ( ErpColumn )));

        
	set_device_array(&(erpArray->linear_A13));
	set_device_array(&(erpArray->linear_A23));
	return 0;

}

int
Erp::clean_host_theta(ErpColumnArray *erpArray)
{


    delete [] erpArray->data_host;

        
	clean_host_array(&(erpArray->linear_A13));
	clean_host_array(&(erpArray->linear_A23));
	return 0;

}

int
Erp::clean_device_theta(ErpColumnArray *erpArray)
{

 
    HANDLE_CERROR(cudaFree(erpArray->data_device));

        
	clean_device_array(&(erpArray->linear_A13));
	clean_device_array(&(erpArray->linear_A23));
	return 0;

}

int
Erp::transfer_device_theta(ErpColumnArray *erpArray)
{


    HANDLE_CERROR(cudaMemcpy(
        erpArray->data_device, 
        erpArray->data_host, 
        erpArray->nc * erpArray->nr * erpArray->dim_x * sizeof( ErpColumn ), 
        cudaMemcpyHostToDevice));

        
	transfer_device_array(&(erpArray->linear_A13));
	transfer_device_array(&(erpArray->linear_A23));
	return 0;

}

int
Erp::transverse_ptheta(const mxArray *matstruct, ErpPTheta *ptheta)
{
	ptheta->dt = (MPFLOAT ) *mxGetPr(mxGetField(matstruct, 0, "dt"));
	ptheta->dyu = (MPFLOAT ) *mxGetPr(mxGetField(matstruct, 0, "dyu"));
    ptheta->udt = (MPFLOAT ) *mxGetPr(mxGetField(matstruct, 0, "udt"));
    // Precompute this value for efficency in cuda
    ptheta->de = ptheta->dt * ptheta->udt;

return 0;

}



} // Host
