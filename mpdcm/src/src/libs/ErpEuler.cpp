/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */


#include "includes/erp.hcu"
#include "ErpEuler.hpp"
#include "erp/erp_ext.hcu"
#include "cuda.h"
#include "cuda_runtime.h"

namespace Host
{
int
ErpEuler::launch_kernel(DataArray *y, DataArray *u, ErpColumnArray *erpArray, ErpPTheta *ptheta)
{


        cuda_erp_euler(*y, *u, *erpArray, *ptheta);
        HANDLE_CERROR(cudaGetLastError());

        return 0;

        
}



} // Host
