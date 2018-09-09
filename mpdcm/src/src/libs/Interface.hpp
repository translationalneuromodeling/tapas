/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */


# ifndef INTERFACE_H_
# define INTERFACE_H_

#include "mpdcm.hcu"
#include <matrix.h>
#include <mex.h>

namespace Host {

class Interface{
    public:
        Interface();
        virtual ~Interface();
        
        int linearize_array(const mxArray *, DataArray *);
        // Linearize the memory and transfers it. 
       
        int set_host_array(DataArray *);
        // Sets the memory of an array
         
        int set_device_array(DataArray *);
        // Sets the memory of an array into the device memory
      
        int transverse_array(const mxArray *, DataArray*);
        // Transverse a data array
        
        int transfer_device_array(DataArray *);
        // Transers the device to local memory
        
        int transfer_host_array(DataArray *);
        // Gets the memory of an array inot the host memory
                
        int clean_host_array(DataArray *);
        // Deletes the memory from the host.

        int clean_device_array(DataArray *);
        // Deletes the memory from a device array

        int set_host_y(const mxArray *);
        // Creates the remote memory and transfers
        
        int set_host_u(const mxArray *);
        // Creates the remote memora and transfers

        int set_device_y();
        // Creates the remote memory and transfers
        
        int set_device_u();
        // Creates the remote memory and transfers

        int set_y(const mxArray *y);
        // Sets y in the host
        
        int get_y(mxArray *y);
        // Sets u in the host

        int set_u(const mxArray *u);

        int prepare_host_array(const mxArray*, DataArray *);

        int prepare_device_array(DataArray *);

        int clean_device_u();
        int clean_device_y();

        int unroll_array(const DataArray *, mxArray **);

        DataArray y[1];
        DataArray u[1];
};

} // Host


# endif
