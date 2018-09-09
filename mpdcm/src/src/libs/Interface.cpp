/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#include "mpdcm.hcu"
#include "Interface.hpp"

using namespace std;

namespace Host
{

Interface::Interface() 
{

}

Interface::~Interface()
{

}

int
Interface::transverse_array(const mxArray *oa, DataArray *a)
{
    const mwSize *sa = mxGetDimensions( oa );
    const mxArray *ta = mxGetCell(oa, 0);
    const mwSize *sta = mxGetDimensions( ta );

    a->nc = sa[0];
    a->nr = sa[1];
    a->nic = sta[0];
    a->nir = sta[1];

    return 0;
}

int
Interface::linearize_array(const mxArray *oa, DataArray *a)
{
    int i, j;
    MPFLOAT *ca = a->data_host;

    for (i = 0; i < a->nc * a->nr; i++ )
    {
        const mxArray *ta = mxGetCell(oa, i);
        double *dta = mxGetPr( ta );

        for ( j = 0; j < a->nic * a->nir ; j++)
            ca[j] = (MPFLOAT ) dta[j];

        ca += j;
    }

    return 0;
}

int
Interface::set_host_array(DataArray *a)
{
    a->data_host = new MPFLOAT[a->nc * a->nr * a->nic * a->nir];

    return 0;
}

int
Interface::set_device_array(DataArray *a)
{

    HANDLE_CERROR(cudaMalloc(&(a->data_device), a->nc * a->nr * a->nic * 
        a->nir * sizeof( MPFLOAT )));

    return 0;
}

int
Interface::transfer_device_array(DataArray *a)
{

    HANDLE_CERROR(cudaMemcpy(a->data_device, a->data_host, a->nc * a->nr *
        a->nir * a->nic * sizeof( MPFLOAT ), cudaMemcpyHostToDevice));

    return 0;
}

int
Interface::transfer_host_array(DataArray *a)
{

    HANDLE_CERROR(cudaMemcpy(a->data_host, a->data_device, a->nc * a->nr *
        a->nic * a->nir * sizeof( MPFLOAT ), cudaMemcpyDeviceToHost));

    return 0;
}

int
Interface::unroll_array(const DataArray *a, mxArray **o)
{
    int i, k;
    int nt = a->nr * a->nc;
    MPFLOAT *td = a->data_host;

    *o = mxCreateCellMatrix(a->nc, a->nr);

    for (i = 0; i < a->nr * a->nc; i++)
    {
        mxArray *ta = mxCreateDoubleMatrix(a->nic, a->nir, mxREAL);
        double *tr = mxGetPr(ta);
        for (k = 0; k < a->nic * a->nir; k++)
            tr[k] = (MPFLOAT ) td[k];
        td += k;

        mxSetCell(*o, i, ta);
    }

    return 0;
}

int
Interface::clean_host_array(DataArray *a)
{
    delete[] a->data_host;
    return 0;
}

int
Interface::clean_device_array(DataArray *a)
{

    cudaFree(a->data_device);
    return 0;
}

int
Interface::clean_device_y()
{

    clean_device_array(y);

    return 0;

}

int
Interface::clean_device_u()
{
    clean_device_array(u);

    return 0;
}   


// Particular functions for u and y

int
Interface::prepare_host_array(const mxArray *ou, DataArray *a)
{
    transverse_array(ou, a);
    set_host_array(a);
    linearize_array(ou, a);
}

int
Interface::prepare_device_array(DataArray *a)
{
    set_device_array(a);
    transfer_device_array(a);
}

int
Interface::set_host_y(const mxArray *oy)
{

    prepare_host_array(oy, y);
    return 0;
}

int
Interface::set_host_u(const mxArray *ou)
{

    prepare_host_array(ou, u);

    return 0;
}


int
Interface::set_device_y()
{
    prepare_device_array(y);

    return 0;
}

int
Interface::set_device_u()
{
    prepare_device_array(u);
    
    return 0;
}

} // Host
