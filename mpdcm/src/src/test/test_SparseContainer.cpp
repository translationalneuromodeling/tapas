/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */

#include "matrix.h"
#include "mex.h"
#include "libs/SparseContainer.hpp"
#include "libs/utilities.hpp"
#include <iostream>
#include <string>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int k;
    if (nlhs > 1)
    {
        mexErrMsgIdAndTxt("CUDA_DCM:cuda_check_theta:input",
            "Only one output supported");
    }

    if (nrhs != 1)
    {
        mexErrMsgIdAndTxt("CUDA_DCM:cuda_check_theta:input",
            "Only three arguments supported");
    }

    SparseContainer container;
    utils::linearize_cube(prhs[0], "B", container);

    std::cout << "b:  " << container.d_sparse.j_device << " ";
    container.set_d_memory();
    std::cout << "a:  " << container.d_sparse.j_device << "\n";

    container.trans_h2d_memory();

    //for ( k = 0; k < container.i_cols.size(); k++)
    //    std::cout << container.i_cols[k] << " ";

    //std::cout << "\n";
}

 
