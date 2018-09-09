/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */


# ifndef FMRI_H
# define FMRI_H

#include "mpdcm.hcu"
#include <matrix.h>
#include <mex.h>
 
#include "Interface.hpp"

namespace Host
{

class Fmri : public Interface 
{
    public:
        Fmri();
        ~Fmri();

        int init_y(const DataArray *, const ThetaFmriArray *, 
            const PThetaFmri *, DataArray *);
        /// Inilizes the data array used for recovering the data.

        int transverse_theta(const mxArray*, ThetaFmriArray*);
        // Transverse the parameters thera

        int linearize_theta(const mxArray *, ThetaFmriArray *);
        // Linearize theta
        
        int linearize_theta_fields(const mxArray*, ThetaFmri *);
        // Linearize the fields of theta

        int linearize_theta_linear(const mxArray*, DataArray *);
        // Linearize the matrices in theta

        int set_host_theta(ThetaFmriArray *);
        // Sets the memory of theta in the host

        int set_device_theta(ThetaFmriArray *);
        // Sets the memory of theta in the devcie 
       
        int clean_host_theta(ThetaFmriArray *);
        // Deletes the local memory of theta

        int clean_device_theta(ThetaFmriArray *);
        //Deletes device memory of theta. 

        int transfer_device_theta(ThetaFmriArray *);
        // Copy the contents of the sparse matrices

        int transverse_ptheta(const mxArray*, PThetaFmri *);
        // Transverse the priots

        // Sets the sparse matrices

        int transverse_theta_sparse(const mxArray* , const char *, 
            sqsparse *);
        // Transverse the sparse matrixes in theta
        // Second argument is the field
        
        int linearize_theta_sparse(const mxArray *, const char *, sqsparse *);
        // Linearizes an sparse field in theta
        
        int launch_kernel(const DataArray *, const DataArray *, 
            const ThetaFmriArray *, const PThetaFmri *);
        // Launches the kernel.

};

} // Host

# endif

