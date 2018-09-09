/* aponteeduardo@gmail.com */
/* copyright (C) 2016 */

#include "SparseContainer.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

SparseContainer::~SparseContainer()                                                                                                                             {
    if ( d_alloc )
    {
        HANDLE_CERROR(cudaFree(d_sparse.i_device));
        HANDLE_CERROR(cudaFree(d_sparse.j_device));
        HANDLE_CERROR(cudaFree(d_sparse.v_device));
    }

}


void
SparseContainer::set_d_memory()
{
    HANDLE_CERROR(cudaMalloc(&(d_sparse.j_device), 
        i_cols.size() * sizeof ( unsigned long int )));
    HANDLE_CERROR(cudaMalloc(&(d_sparse.i_device), 
        i_rows.size() * sizeof ( unsigned long int )));
    HANDLE_CERROR(cudaMalloc(&(d_sparse.v_device), 
        data.size() * sizeof ( MPFLOAT )));

    d_alloc = 1;
}

void
SparseContainer::trans_h2d_memory()
{
    if ( d_alloc == 0 )
        mexErrMsgIdAndTxt("tapas:mpdcm:fmri:cuda:memory",
                            "Sparse memory not allocated yet.");
    HANDLE_CERROR(cudaMemcpy(d_sparse.j_device,
        &(i_cols[0]), 
        i_cols.size() * sizeof( unsigned long int ), cudaMemcpyHostToDevice));
 
    HANDLE_CERROR(cudaMemcpy(d_sparse.i_device, 
        &(i_rows[0]), 
        i_rows.size() * sizeof( unsigned long int ), cudaMemcpyHostToDevice));

    HANDLE_CERROR(cudaMemcpy(d_sparse.v_device, 
        &(data[0]),
        data.size() * sizeof( MPFLOAT ) , cudaMemcpyHostToDevice));

}

void
SparseContainer::trans_h2d_memory_async()
{
    if ( d_alloc == 0 )
        mexErrMsgIdAndTxt("tapas:mpdcm:fmri:cuda:memory",
                            "Sparse memory not allocated yet.");
    HANDLE_CERROR(cudaMemcpyAsync(d_sparse.j_device,
        &(i_cols[0]), 
        i_cols.size() * sizeof( unsigned long int ), cudaMemcpyHostToDevice));
 
    HANDLE_CERROR(cudaMemcpyAsync(d_sparse.i_device, 
        &(i_rows[0]), 
        i_rows.size() * sizeof( unsigned long int ), cudaMemcpyHostToDevice));

    HANDLE_CERROR(cudaMemcpyAsync(d_sparse.v_device, 
        &(data[0]),
        data.size() * sizeof( MPFLOAT ) , cudaMemcpyHostToDevice));

}
