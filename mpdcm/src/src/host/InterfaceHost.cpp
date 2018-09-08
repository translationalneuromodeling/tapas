/* aponteeduardo@gmail.com */
/* copyright (C) 2016 */

#include "host/InterfaceHost.hpp"

namespace Host
{

std::vector< arma::Mat < double > >
InterfaceHost::populate_vec_mats(const mxArray *cmats)
{
    const mwSize *dims = mxGetDimensions(cmats);
    int nvals = dims[0] * dims[1], i;
    std::vector< arma::Mat < double > > vmats;


    vmats.reserve(nvals);

    for (i = 0; i < nvals ; i++)
    {
        const mwSize *tdims = mxGetDimensions(mxGetCell(cmats, i));
        arma::Mat< double > tmat(mxGetPr(mxGetCell(cmats, i)), 
            tdims[0], tdims[1]);
        vmats.push_back(tmat);
        
    } 
    return vmats;
}

int
InterfaceHost::clone_cell_mat(const mxArray *origin, mxArray **target)
{
    const mwSize *dims = mxGetDimensions(origin);
    *target = mxCreateCellMatrix(dims[0], dims[1]);

    return 0;
}


int
InterfaceHost::populate_cell_mats(
        const std::vector <arma::Mat < double > > vmats,
        mxArray **cmats)
{
    int nc = vmats.size();
    int i, j;

    for (i = 0; i < nc ; i++)
    {
        mxArray *ta = mxCreateDoubleMatrix(vmats[i].n_rows, vmats[i].n_cols,
                mxREAL);
        double *tmat = mxGetPr(ta);
        for (j = 0; j < vmats[i].size() ; ++j )
        {
            tmat[j] = (double ) vmats[i](j);
        }
        mxSetCell(*cmats, i, ta);
    }
    
    return 0;
}


}
