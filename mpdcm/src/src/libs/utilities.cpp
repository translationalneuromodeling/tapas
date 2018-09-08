/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */

#include "utilities.hpp"
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <vector>
#include <iostream>

namespace utils 
{

// Linearize a nxnxm matrix and return a container.
int
linearize_cube(const mxArray *atheta, const char *field, 
        SparseContainer &container)
{
    int i = 0;
    unsigned long int rcol = 0;
    const mwSize *sab = mxGetDimensions(atheta);
    unsigned long int nt = (long int) sab[0] * sab[1];

    int dim_x = (int ) *mxGetPr(mxGetField(mxGetCell(atheta, i), 0, "dim_x"));

    container.d_sparse.n = nt;
    container.d_sparse.dim_x = dim_x;
    
    for (i = 0; i < nt; i++)
    {
        
        int j;
        const mxArray *cube = mxGetField(mxGetCell(atheta, i), 0, field);
        const mwSize *cdims = mxGetDimensions(cube);

        // TODO: CHECK OVER FLOW ERROR
        arma::Cube< double > acube(mxGetPr(cube), cdims[0], cdims[1], 
            cdims[2]);
        // Reserve memory or increase it.
        if ( container.data.size() == container.data.capacity() )
        {
            if ( container.data.capacity() > 0 )
            {
                container.data.reserve(2 * container.data.capacity());
                container.i_rows.reserve(2 * container.i_rows.capacity());
                container.i_cols.reserve(2 * container.i_cols.capacity());

            } else 
            {
                // Make an estimation of how many values are missing.
                int k;
                long int nvals = 0;
                for (k = 0; k < cdims[2] ; k++)
                {
                    arma::SpMat< double > sqmat(acube.slice(0));
                    nvals += sqmat.n_nonzero;
                }
                container.data.reserve(nvals * nt);
                container.i_cols.reserve(cdims[1] * nt + 1); 
                container.i_rows.reserve(nvals * nt);
                container.i_cols.push_back(0);
            }
        }
         
        for (j = 0; j < cdims[2]; j++)
        {
            int l;
            arma::Mat< MPFLOAT > fmat = 
                arma::conv_to< arma::Mat< MPFLOAT > >::from(acube.slice(j));

            // The Matrix is transposed for efficiency reasons
            arma::SpMat< MPFLOAT > sqmat(fmat.t());

            for (l = 0; l < sqmat.n_nonzero; l++)
            {
                container.data.push_back(sqmat.values[l]);
                container.i_rows.push_back(sqmat.row_indices[l]);
            }
            for (l = 1; l < sqmat.n_cols + 1; l++)
            {
                container.i_cols.push_back(rcol + sqmat.col_ptrs[l]);
            }
            rcol += sqmat.n_nonzero;
        }
        
    }  
    return 0;
}

ThetaFmri
copy_theta_fields(const mxArray *mtheta)
{
    ThetaFmri theta;
    theta.dim_x = (int ) *mxGetPr(mxGetField(mtheta, 0, "dim_x"));
    theta.dim_u = (int ) *mxGetPr(mxGetField(mtheta, 0, "dim_u"));


    theta.fA = (int ) *mxGetPr(mxGetField(mtheta, 0, "fA")) ?
            MF_TRUE : MF_FALSE;
    theta.fB = (int ) *mxGetPr(mxGetField(mtheta, 0, "fB")) ?
            MF_TRUE : MF_FALSE;
    theta.fC = (int ) *mxGetPr(mxGetField(mtheta, 0, "fC")) ?
            MF_TRUE : MF_FALSE;
    theta.fD = (int ) *mxGetPr(mxGetField(mtheta, 0, "fD")) ?
            MF_TRUE : MF_FALSE;

    theta.V0 = (MPFLOAT ) *mxGetPr(mxGetField(mtheta, 0, "V0"));
    theta.E0 = (MPFLOAT ) *mxGetPr(mxGetField(mtheta, 0, "E0"));

    //For efficiency reasons some values are prepared.

    theta.ln1_E0 = log(1 - theta.E0);
    theta.lnE0 = log(theta.E0);
    theta.k1 = (MPFLOAT ) *mxGetPr(mxGetField(mtheta, 0, "k1")) *
        theta.V0;
    theta.k2 = (MPFLOAT ) *mxGetPr(mxGetField(mtheta, 0, "k2")) *
        theta.V0;
    theta.k3 = (MPFLOAT ) *mxGetPr(mxGetField(mtheta, 0, "k3")) *
        theta.V0;

    theta.alpha = (MPFLOAT ) *mxGetPr(mxGetField(mtheta, 0, "alpha"));
    theta.alpha = 1/theta.alpha - 1;
    theta.gamma = (MPFLOAT ) *mxGetPr(mxGetField(mtheta, 0, "gamma"));

    return theta;
}

int
gen_theta_container(const mxArray *theta, 
    MatContainer<ThetaFmriArray, ThetaFmri> &container)
{
    int i = 0;
    int rcol = 0; // Current col
    const mwSize *sab = mxGetDimensions(theta);
    mwSize nt = sab[0] * sab[1];

    // Be careful with this.
    container.d_array.nr = sab[0];
    container.d_array.nc = sab[1];

    container.data.reserve(nt);
    for (i =0; i < nt ; i++)
    {
        ThetaFmri tt = copy_theta_fields(mxGetCell(theta, i));
        container.data.push_back(tt);
    }

    return 0;
}

int
gen_array_ACKtau(const mxArray *theta, 
    MatContainer< DataArray, MPFLOAT > &container)
{
    int i = 0, k;
    const mwSize *sab = mxGetDimensions(theta);
    mwSize nt = sab[0] * sab[1];
    int dim_x = (int ) *mxGetPr(mxGetField(mxGetCell(theta, 0), 0, "dim_x"));
    int dim_u = (int ) *mxGetPr(mxGetField(mxGetCell(theta, 0), 0, "dim_u"));

    // Be careful with this.
    container.d_array.nr = sab[0];
    container.d_array.nc = sab[1];

    container.d_array.nir = dim_x;
    container.d_array.nic = dim_x + dim_u + dim_x + dim_x;

    container.data.reserve(nt * dim_x * (dim_x + dim_u + 2));
    for (i =0; i < nt ; i++)
    {
        double *ta;
        // Transform the data format and transpose
        mxArray *ttheta = mxGetCell(theta, i);
        arma::Mat< double > mta(mxGetPr(mxGetField(ttheta, 0, "A")), dim_x,
                dim_x, 1);
        arma::Mat < MPFLOAT > msa =
            arma::conv_to< arma::Mat< MPFLOAT > >::from(mta);
        msa = msa.t(); 

        for (k = 0; k < dim_x * dim_x; k++)
            container.data.push_back(msa[k]);

        arma::Mat< double > mtc(mxGetPr(mxGetField(ttheta, 0, "C")), dim_x,
                dim_u, 1);
        arma::Mat < MPFLOAT > msc =
            arma::conv_to< arma::Mat< MPFLOAT > >::from(mtc);
        msc = msc.t(); 

        for (k = 0; k < dim_x * dim_u; k++)
            container.data.push_back(msc[k] / 16.0);

        ta = mxGetPr(mxGetField(ttheta, 0, "K"));
        for (k = 0; k < dim_x; k++)
            container.data.push_back((MPFLOAT ) ta[k]);

        ta = mxGetPr(mxGetField(ttheta, 0, "tau"));
        for (k = 0; k < dim_x; k++)
            container.data.push_back((MPFLOAT ) ta[k]);
    }
   
    return 0;
}

int
gen_u_container(const mxArray *u,
    MatContainer<DataArray, MPFLOAT > &container)
{
    int i = 0, k;
    const mwSize *sab = mxGetDimensions(u);
    mwSize nt = sab[0] * sab[1];
    const mwSize *dimu = mxGetDimensions(mxGetCell(u, 0));
    mwSize np = dimu[0] * dimu[1];

    container.data.reserve(np * nt);
    container.d_array.nr = sab[0];
    container.d_array.nc = sab[1];
    container.d_array.nir = dimu[0];
    container.d_array.nic = dimu[1];

    for (i = 0; i < nt; i++)
    {
        // Use arma to interface
        arma::Mat< double > mu(mxGetPr(mxGetCell(u, i)), dimu[0], dimu[1], 1);
        for (k = 0; k < np; k++)
            container.data.push_back((MPFLOAT ) mu[k]);
    }

    return 0;
}

int
gen_y_container(const mxArray *u, 
        const mxArray *theta, 
        const mxArray *ptheta,
        MatContainer< DataArray, MPFLOAT> &container)
{
    int i = 0, k;
    const mwSize *sab = mxGetDimensions(theta);
    mwSize nt = sab[0] * sab[1];
    const mwSize *dimu = mxGetDimensions(mxGetCell(u, 0));
    mwSize np = dimu[0] * dimu[1];
    int dim_x = (int ) *mxGetPr(mxGetField(mxGetCell(theta, 0), 0, "dim_x"));
    int ny = ceil(*mxGetPr(mxGetField(ptheta, 0, "dyu")) * dimu[1]);

    container.d_array.nr = sab[0];
    container.d_array.nc = sab[1];

    container.d_array.nir = dim_x;
    container.d_array.nic = ny;

    container.data.resize(ny * dim_x * nt, 0);

    return 0;
}

int
unroll_container(const MatContainer< DataArray, MPFLOAT> &y, mxArray **o)
{
    int i, k, td = 0;
    int nt = y.d_array.nr * y.d_array.nc;

    *o = mxCreateCellMatrix(y.d_array.nr, y.d_array.nc);

    for (i = 0; i < nt; i++)
    {
        mxArray *ta = mxCreateDoubleMatrix(y.d_array.nir, y.d_array.nic, 
            mxREAL);
        double *tr = mxGetPr(ta);
        for (k = 0; k < y.d_array.nir * y.d_array.nic; k++)
            tr[k] = (MPFLOAT ) y.data[td + k];
        td += k;

        mxSetCell(*o, i, ta);
    }                                                                                                                                                        
    return 0;
}


} // namespace utils


