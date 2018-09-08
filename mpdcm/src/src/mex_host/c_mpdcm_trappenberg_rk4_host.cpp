//
// Author: Eduardo Aponte
// Copyright 2015 by Eduardo Aponte <aponteeduardo@gmail.com>
//
// Licensed under GNU General Public License 3.0 or later.
// Some rights reserved. See COPYING, AUTHORS.
//
// Revision log:
//

#include "mpdcm.h"
#include "matrix.h"
#include "mex.h"
#include "host/InterfaceTrappenberg.hpp"
#include "host/IntegratorHost.hpp"
#include "host/UpdateRK4.hpp"
#include "host/DynamicsTrappenberg.hpp"

#include "host/utilities.hpp"


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    if (nlhs > 1)
    {
        mexErrMsgIdAndTxt("tapas:mpdcm:trappenberg:input",
            "Only one output supported");
    }

    if (nrhs != 3)
    {
        mexErrMsgIdAndTxt("tapas:mpdcm:trappenberg:input",
            "Only three arguments supported");        
    }


    if ( nlhs == 0 )
        return;
    
    utils::run_interface_host_kernel<Host::InterfaceTrappenberg< Host::IntegratorHost< Host::UpdateRK4<Host::DynamicsTrappenberg>, Host::DynamicsTrappenberg > > >(prhs[0], prhs[1], prhs[2], plhs);

}

