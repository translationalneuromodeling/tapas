/* aponteeduardo@gmail.com */
/* copyright (C) 2016 */

#ifndef MATCONTAINER_H
#define MATCONTAINER_H

#include <matrix.h>
#include <mex.h>
#include "mpdcm.hcu"
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

template < class T_Cont, class T_Data >
struct MatContainer
{

    std::vector< T_Data > data;
    T_Cont d_array;

    int d_alloc;

    ~MatContainer();
    MatContainer() : d_alloc(0) {d_array.data_device = 0;};

    void
    set_d_memory();

    void
    trans_h2d_memory();

    void
    trans_h2d_memory_async();

    void
    trans_d2h_memory();

};

template < class T_Cont, class T_Data >
MatContainer< T_Cont, T_Data >::~MatContainer()
{
    if ( d_alloc )
    {
        HANDLE_CERROR(cudaFree(d_array.data_device));
        d_array.data_device = 0;
    }

    d_alloc = 0;
}

template < class T_Cont, class T_Data >
void
MatContainer< T_Cont, T_Data >::set_d_memory()
{
    HANDLE_CERROR(cudaMalloc(&(d_array.data_device), 
                data.size() * sizeof( T_Data ) ));
    d_alloc = 1;

}

template < class T_Cont, class T_Data >
void
MatContainer< T_Cont, T_Data >::trans_h2d_memory()
{

    HANDLE_CERROR(cudaMemcpy(d_array.data_device, 
        &(data[0]), data.size() * sizeof( T_Data ),
       cudaMemcpyHostToDevice ));

}

template < class T_Cont, class T_Data >
void
MatContainer< T_Cont, T_Data >::trans_h2d_memory_async()
{

    HANDLE_CERROR(cudaMemcpyAsync(d_array.data_device, 
        &(data[0]), data.size() * sizeof( T_Data ),
       cudaMemcpyHostToDevice ));

}

template < class T_Cont, class T_Data >
void
MatContainer< T_Cont, T_Data >::trans_d2h_memory()
{

    HANDLE_CERROR(cudaMemcpy(&(data[0]), d_array.data_device, 
       data.size() * sizeof( T_Data ),
       cudaMemcpyDeviceToHost ));
}


#endif // MATCONTAINER_H

