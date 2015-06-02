//
// Author: Eduardo Aponte
// Copyright 2015 by Eduardo Aponte <aponteeduardo@gmail.com>
//
// Licensed under GNU General Public License 3.0 or later.
// Some rights reserved. See COPYING, AUTHORS.
//
// Revision log:
//

#include "c_mpdcm.h"

// Wrappers

void 
c_mpdcm_fmri_euler(mxArray **y, const mxArray **u,
    const mxArray **theta, const mxArray **ptheta)
{

    c_mpdcm_prepare_input(y, *u, *theta, *ptheta, &mpdcm_fmri_euler);
}

void 
c_mpdcm_fmri_kr4(mxArray **y, const mxArray **u,
    const mxArray **theta, const mxArray **ptheta)
{

    c_mpdcm_prepare_input(y, *u, *theta, *ptheta, &mpdcm_fmri_kr4);
}

void 
c_mpdcm_fmri_bs(mxArray **y, const mxArray **u,
    const mxArray **theta, const mxArray **ptheta)
{

    c_mpdcm_prepare_input(y, *u, *theta, *ptheta, &mpdcm_fmri_bs);

}


void 
c_mpdcm_prepare_theta(const mxArray *theta, ThetaDCM *ctheta, MPFLOAT *dtheta)
{
    int i, j, k;

    // Temporal array to store the calls to matlab's api.
    double *ta;
    
    ctheta->dim_x = (int ) *mxGetPr(mxGetField(theta, 0, "dim_x"));
    ctheta->dim_u = (int ) *mxGetPr(mxGetField(theta, 0, "dim_u"));

    ctheta->fA = (int ) *mxGetPr(mxGetField(theta, 0, "fA")) ?
        MF_TRUE : MF_FALSE;
    ctheta->fB = (int ) *mxGetPr(mxGetField(theta, 0, "fB")) ? 
        MF_TRUE : MF_FALSE;
    ctheta->fC = (int ) *mxGetPr(mxGetField(theta, 0, "fC")) ? 
        MF_TRUE : MF_FALSE;
    ctheta->fD = (int ) *mxGetPr(mxGetField(theta, 0, "fD")) ? 
        MF_TRUE : MF_FALSE;

    ctheta->V0 = (MPFLOAT ) *mxGetPr(mxGetField(theta, 0, "V0"));
    ctheta->E0 = (MPFLOAT ) *mxGetPr(mxGetField(theta, 0, "E0"));

    // For efficiency reasons some values are prepared.

    ctheta->ln1_E0 = log(1 - ctheta->E0);
    ctheta->lnE0 = log(ctheta->E0);
    ctheta->k1 = (MPFLOAT ) *mxGetPr(mxGetField(theta, 0, "k1")) * ctheta->V0;
    ctheta->k2 = (MPFLOAT ) *mxGetPr(mxGetField(theta, 0, "k2")) * ctheta->V0;
    ctheta->k3 = (MPFLOAT ) *mxGetPr(mxGetField(theta, 0, "k3")) * ctheta->V0;

    ctheta->alpha = (MPFLOAT ) *mxGetPr(mxGetField(theta, 0, "alpha"));
    ctheta->alpha = 1/ctheta->alpha - 1;
    ctheta->gamma = (MPFLOAT ) *mxGetPr(mxGetField(theta, 0, "gamma"));

    // A memcpy would be faster but then the float and double implementation
    // would need to be different

    i = ctheta->dim_x*ctheta->dim_x;
    ta = mxGetPr(mxGetField(theta, 0, "A"));
    for (k = 0; k < i; k++)
        dtheta[k] = (MPFLOAT ) ta[k];
    dtheta += i;

    i = ctheta->dim_x * ctheta->dim_x * ctheta->dim_u;
    ta = mxGetPr(mxGetField(theta, 0, "B"));
    for (k = 0; k < i; k++)
        dtheta[k] = (MPFLOAT ) ta[k];
    dtheta += i;

    i = ctheta->dim_x * ctheta->dim_u;
    ta = mxGetPr(mxGetField(theta, 0, "C"));
    for (k = 0; k < i; k++)
        dtheta[k] = (MPFLOAT ) ta[k];
    dtheta += i;

    // Inefficient implementation
    
    i = ctheta->dim_x * ctheta->dim_x * ctheta->dim_x;
    if ( ctheta->fD == MF_TRUE )
    {
        ta = mxGetPr(mxGetField(theta, 0, "D"));
        for (k = 0; k < i; k++)
            dtheta[k] = (MPFLOAT ) ta[k];
    } else
    {
        memset(dtheta, 0, i * sizeof( MPFLOAT ));
    }    
    memset(dtheta, 0, i * sizeof( MPFLOAT ));

    dtheta += i;
     

    i = ctheta->dim_x;
    ta = mxGetPr(mxGetField(theta, 0, "K"));
    for (k = 0; k < i; k++)
        dtheta[k] = (MPFLOAT ) ta[k];
    dtheta += i;

    i = ctheta->dim_x;
    ta = mxGetPr(mxGetField(theta, 0, "tau"));
    for (k = 0; k < i; k++)
        dtheta[k] = (MPFLOAT ) ta[k];

}

void 
c_mpdcm_prepare_ptheta(const mxArray *ptheta, void *vptheta, MPFLOAT *dptheta)
{

    PThetaDCM *cptheta = (PThetaDCM *) vptheta;
    cptheta->dt = (MPFLOAT ) *mxGetPr(mxGetField(ptheta, 0, "dt"));
    cptheta->dyu = (MPFLOAT ) *mxGetPr(mxGetField(ptheta, 0, "dyu"));
    // Precompute this value. For efficiency in cuda.
    cptheta->de = cptheta->dt * mxGetPr(mxGetField(ptheta, 0, "udt"))[0];
    cptheta->mode = 'f';
}

void 
c_mpdcm_prepare_u(const mxArray *u, MPFLOAT *cu)
{
    unsigned int i, j;
    const mwSize *su = mxGetDimensions(u);
    unsigned int nt = su[0] * su[1];

    for (i = 0; i < nt ; i++ )
    {
        const mxArray *tu = mxGetCell(u, i);    
        const mwSize *stu = mxGetDimensions(tu);
        unsigned int ntu = stu[0] * stu[1];
        double *dtu = mxGetPr(tu);

        for ( j = 0; j < ntu; j++)
            cu[j] = (MPFLOAT ) dtu[j];

        cu += ntu;
    }
}

void
c_mpdcm_transfer_y(mxArray **y, MPFLOAT *cy, int nx, int ny, int nt, int nb)
{
    unsigned int i, k;
    unsigned int tt = nt * nb;
    unsigned int td = nx * ny;
    double *ta;

    for (i=0; i < tt; i++)
    {
        mxArray *ty = mxCreateDoubleMatrix(nx, ny, mxREAL);
        ta = mxGetPr(ty);
        for (k = 0; k < td; k++)
            ta[k] = (MPFLOAT ) cy[k];
        cy += td;
        mxSetCell(*y, i, ty);
    }

}


void 
c_mpdcm_prepare_input(
    mxArray **y, const mxArray *u, const mxArray *theta, const mxArray *ptheta,
    integrator integ)
{
    // Prepares the data in the format necessary to interface with the
    // cuda library.

    const mwSize *su = mxGetDimensions( u );
    const mwSize *stheta = mxGetDimensions( theta );
    int i, o;

    MPFLOAT *cx, *cy, *cu;
    ThetaDCM *ctheta;
    PThetaDCM *cptheta;
    MPFLOAT *dtheta;
    MPFLOAT *dptheta;
    int nx, ny, nu, dp, nt, nb;
        
    nx = (int ) *mxGetPr( mxGetField(mxGetCell(theta, 0), 0, "dim_x") );
    nu = mxGetDimensions( mxGetCell( u, 0) )[0];
    dp = mxGetDimensions( mxGetCell( u, 0) )[1];
    nt = stheta[0];
    nb = stheta[1];

    ny = ceil(dp * mxGetPr(mxGetField(ptheta, 0, "dyu"))[0]) ;

    /* Offsets */

    o = nx * nx + /*A*/
        nx * nx * nu + /*B*/
        nx * nu + /*C*/
        nx * nx * nx + /*D*/
        nx + /* Kappa */
        nx; /* tau */


    /* Allocate memory */

    cy = (MPFLOAT *) malloc(nt * nb * nx * ny * sizeof( MPFLOAT ));
    if ( cy == NULL )
        mexErrMsgIdAndTxt("mpdcm:fmri:int:y:memory", "Memory error y");	

    cu = (MPFLOAT *) malloc(nt * dp * nu * sizeof( MPFLOAT ));
    if ( cu == NULL )
        mexErrMsgIdAndTxt("mpdcm:fmri:int:u:memory", "Memory error u");	

    ctheta = (ThetaDCM *) malloc( nt * nb * sizeof( ThetaDCM ) );
    if ( ctheta == NULL )
        mexErrMsgIdAndTxt("mpdcm:fmri:int:theta:memory", "Memory error theta");	

    cptheta = (PThetaDCM *) malloc( sizeof( PThetaDCM ) );
    if ( cptheta == NULL )
        mexErrMsgIdAndTxt("mpdcm:fmri:int:ptheta:memory", 
            "Memory error ptheta");	

    dtheta = (MPFLOAT *) malloc(nt * nb * o * sizeof( MPFLOAT ));
    if ( dtheta == NULL )
        mexErrMsgIdAndTxt("mpdcm:fmri:int:theta:memory",
            "Memory error theta");	

    // Prepare u
    
    c_mpdcm_prepare_u(u, cu);

    // Prepare theta

    for (i = 0; i < nt * nb ; i++ )
        c_mpdcm_prepare_theta(mxGetCell(theta, i), ctheta + i, dtheta + i*o);


    // Prepare ptheta

    c_mpdcm_prepare_ptheta(ptheta, cptheta, dptheta);

    // Run the function

    (*integ)(cx, cy, cu, ctheta, dtheta, cptheta, dptheta,
        nx, ny, nu, dp, nt, nb);

    // Tranfer results

    *y = mxCreateCellMatrix(nt, nb);
    c_mpdcm_transfer_y(y, cy, nx, ny, nt, nb);

    free(cy);
    free(cu);
    free(ctheta);
    free(cptheta);
    free(dtheta);

}


int 
c_transfer_sparse(const mxArray *ab, int *jB, int *iB, MPFLOAT *vB, int nx, 
    int o)
{
    // 
    // Input
    //      ab      -- Cell array with the matrices
    //      jB      -- Jc of the sparse matrices
    //      iB      -- Ir of the sparse matrices
    //      vB      -- Pr Of the sparse matrices
    //      o       -- Total number of previous non zero entries.
    //      nx      -- Dimensionality of the matrixes
    //      nu      -- Number of matrices in the cell array.
    // Output
    //      te      -- Return the total number of non zero elements.
    
    int i, j;
    mwSize nu = mxGetDimensions(ab)[0] * mxGetDimensions(ab)[0];

    // Iterate over matrices
    for ( i = 0; i < nu; i++)
    {
        mwSize *jc = mxGetJc(mxGetCell(ab, i));
        mwSize te = mxGetNzmax(mxGetCell(ab, i));
        // Get absolute indices
        for ( j = 0; j < nu + 1; j++ )
            jB[j] = o + jc[j];
        // Increase the pointer
        jB += nx + 1;
        // Transfer the other arrays
        memcpy(iB, mxGetIr(mxGetCell(ab, i)), sizeof( int ) * te);
        memcpy(vB, mxGetPr(mxGetCell(ab, i)), sizeof( MPFLOAT ) * te);
    }

    return jB[nx]; 
}
