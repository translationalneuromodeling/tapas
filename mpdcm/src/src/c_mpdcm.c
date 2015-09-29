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

int 
c_mpdcm_transfer_sparse(const mxArray *ab, sqsparse *sm, int *ij, int *ii)
{
    // 
    // Input
    //      ab      -- Cell array with the matrices
    //      sm      -- Concatenated sparse matrices
    //      ij      -- Index of the j array.
    //      ii      -- Index of the i array.
    // Output
    //      te      -- Return the total number of non zero elements.
    
    int i, j;
    int *jS = sm->j, *iS = sm->i;
    MPFLOAT *vS = sm->v;
    double *pr;
    const mwSize *sab = mxGetDimensions(ab);
    mwSize nu = sab[0] * sab[1]; 

    // Iterate over matrices
    for ( i = 0; i < nu; i++)
    {
        mwSize *jc = mxGetJc(mxGetCell(ab, i));
        mwSize te = mxGetNzmax(mxGetCell(ab, i));

        // Get absolute indices
        for ( j = 0; j < sm->dim_x + 1; j++ )
            jS[j + *ij] = *ii + jc[j];

        // Increase the pointer
        *ij += sm->dim_x + 1;
        
        // Transfer i
        memcpy(sm->i + *ii, mxGetIr(mxGetCell(ab, i)), sizeof( int ) * te);
        
        // Transfer v
        pr = mxGetPr(mxGetCell(ab, i));
        for (j = 0; j < te; j++)
            sm->v[j + *ii] = (MPFLOAT ) pr[j];
        
        // Update the counter
        *ii += te;
    }
    return 0;
}


void
c_mpdcm_prepare_theta(const mxArray *theta, ThetaDCM *ctheta, MPFLOAT *dtheta,
    sqsparse *sB, sqsparse *sD, int o)
{
    unsigned int i;
    const mwSize *st = mxGetDimensions(theta);
    unsigned int nt = st[0] * st[1];
    int ijB = 0, iiB = 0, ijD = 0, iiD = 0;

    for (i = 0; i < nt ; i++ )
    {
        c_mpdcm_prepare_theta_fields(mxGetCell(theta, i), ctheta + i, 
            dtheta + i * o);
        c_mpdcm_transfer_sparse(mxGetField(mxGetCell(theta, i), 0, "tB"), sB, 
            &ijB, &iiB); 
        c_mpdcm_transfer_sparse(mxGetField(mxGetCell(theta, i), 0, "tD"), sD,
            &ijD, &iiD); 
    }
}

void 
c_mpdcm_prepare_theta_fields(const mxArray *theta, ThetaDCM *ctheta, 
    MPFLOAT *dtheta)
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

    i = ctheta->dim_x * ctheta->dim_u;
    ta = mxGetPr(mxGetField(theta, 0, "C"));
    for (k = 0; k < i; k++)
        dtheta[k] = (MPFLOAT ) ta[k];
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
c_mpdcm_transverse_sparse(const mxArray *theta, int *nB, int *nD)
{
    // Transverse the data to compute the total memory that is required
    unsigned int i, j;
    const mwSize *st = mxGetDimensions(theta);
    unsigned int nt = st[0] * st[1];

    *nB = 0;
    *nD = 0;

    for (i = 0; i < nt ; i++ )
    {
        mxArray *tB = mxGetField(mxGetCell(theta, i), 0, "tB");
        mxArray *tD = mxGetField(mxGetCell(theta, i), 0, "tD");
        
        const mwSize *stB = mxGetDimensions(tB);
        const mwSize *stD = mxGetDimensions(tD);

        for (j = 0; j < stB[0]; j++ )
            *nB += mxGetNzmax(mxGetCell(tB, j));
        for (j = 0; j < stD[0]; j++ )
            *nD += mxGetNzmax(mxGetCell(tD, j));

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
    int nB, nD;
    sqsparse sB[1], sD[1];
    kernpars pars;

    nx = (int ) *mxGetPr( mxGetField(mxGetCell(theta, 0), 0, "dim_x") );
    nu = mxGetDimensions( mxGetCell( u, 0) )[0];
    dp = mxGetDimensions( mxGetCell( u, 0) )[1];
    nt = stheta[0];
    nb = stheta[1];

    ny = ceil(dp * mxGetPr(mxGetField(ptheta, 0, "dyu"))[0]) ;

    /* Offsets */

    o = nx * nx + /*A*/
        nx * nu + /*C*/
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

    // Allocate sparse matrix space 

    c_mpdcm_transverse_sparse(theta, &nB, &nD);

    sB->dim_x = nx;
    sD->dim_x = nx;
    
    sB->n = nt * nb;
    sD->n = nt * nb;

    sB->j = (int *) malloc(nt * nb * nu * (nx + 1) * sizeof( int ) );
    sD->j = (int *) malloc(nt * nb * nx * (nx + 1) * sizeof( int ) );

    sB->i = (int *) malloc(nB * sizeof( int ) );
    sD->i = (int *) malloc(nD * sizeof( int ) );

    sB->v = (MPFLOAT *) malloc(nB * sizeof( MPFLOAT ) );
    sD->v = (MPFLOAT *) malloc(nD * sizeof( MPFLOAT ) );
 
    // Prepare u
    
    c_mpdcm_prepare_u(u, cu);

    // Prepare theta

    c_mpdcm_prepare_theta(theta, ctheta, dtheta, sB, sD, o);

    // Prepare ptheta

    c_mpdcm_prepare_ptheta(ptheta, cptheta, dptheta);

    // Run the function

    pars.x = cx;
    pars.y = cy;
    pars.u = cu;
    pars.p_theta = ctheta;
    pars.d_theta = dtheta;
    pars.p_ptheta = cptheta;
    pars.d_ptheta = dptheta;

    pars.sB = sB;
    pars.sD = sD;

    pars.nx = nx;
    pars.ny = ny;
    pars.nu = nu;
    pars.dp = dp;
    pars.nt = nt;
    pars.nb = nb;

    // Run function

    (*integ)(pars);

    // Tranfer results

    *y = mxCreateCellMatrix(nt, nb);
    c_mpdcm_transfer_y(y, cy, nx, ny, nt, nb);

    free(cy);
    free(cu);
    free(ctheta);
    free(cptheta);
    free(dtheta);
    
    free(sB->j);
    free(sD->j);
    free(sB->i);
    free(sD->i);
    free(sB->v);
    free(sD->v);
    
}



