/* aponteeduardo@gmail.com */
/* copyright (C) 2014 */

#include "c_mpdcm.h"

// Signatures

void 
c_mpdcm_prepare_input(
    mxArray **y, const mxArray *u, const mxArray *theta, const mxArray *ptheta,
    int *nx, int *ny, int *nu, int *dp, int *nt, int *nb, integrator integ);

void 
c_mpdcm_prepare_theta(const mxArray *theta, ThetaDCM *ctheta, MPFLOAT *dtheta);

void
c_mpdcm_prepare_ptheta(const mxArray *ptheta, void *vptheta, MPFLOAT *dptheta);

void
c_mpdcm_transfer_y(mxArray **y, MPFLOAT *cy, int *nx, int *ny, int *nt, 
    int *nb);


// Wrappers

void 
c_mpdcm_fmri_euler(mxArray **y, const mxArray **u,
    const mxArray **theta, const mxArray **ptheta,
    int *nx, int *ny, int *nu, int *dp, int *nt, int *nb)
{
 
    c_mpdcm_prepare_input(y, *u, *theta, *ptheta, nx, ny, nu, dp, nt, nb, 
        &mpdcm_fmri_euler);
}

void 
c_mpdcm_fmri_kr4(mxArray **y, const mxArray **u,
    const mxArray **theta, const mxArray **ptheta,
    int *nx, int *ny, int *nu, int *dp, int *nt, int *nb)
{
 
    c_mpdcm_prepare_input(y, *u, *theta, *ptheta, nx, ny, nu, dp, nt, nb, 
        &mpdcm_fmri_kr4);
}

void 
c_mpdcm_fmri_bs(mxArray **y, const mxArray **u,
    const mxArray **theta, const mxArray **ptheta,
    int *nx, int *ny, int *nu, int *dp, int *nt, int *nb)
{
    int i;

    c_mpdcm_prepare_input(y, *u, *theta, *ptheta, nx, ny, nu, dp, nt, nb, 
        &mpdcm_fmri_bs);
}


void 
c_mpdcm_prepare_theta(const mxArray *theta, ThetaDCM *ctheta, MPFLOAT *dtheta)
{
    int i, j, k;

    // Temporal array to store the calls to matlab's api.
    double *ta;
    
    ctheta->dim_x = (int ) *mxGetPr(mxGetField(theta, 0, "dim_x"));
    ctheta->dim_u = (int ) *mxGetPr(mxGetField(theta, 0, "dim_u"));

    ctheta->fA = *mxGetPr(mxGetField(theta, 0, "fA")) ? MF_TRUE : MF_FALSE;
    ctheta->fB = *mxGetPr(mxGetField(theta, 0, "fB")) ? MF_TRUE : MF_FALSE;
    ctheta->fC = *mxGetPr(mxGetField(theta, 0, "fC")) ? MF_TRUE : MF_FALSE;

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

    i = ctheta->dim_x*ctheta->dim_u;
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
    cptheta->de = cptheta->dt*cptheta->dyu;
    cptheta->mode = 'f';
}

void 
c_mpdcm_prepare_u(const mxArray *u, MPFLOAT *cu)
{
    const mwSize *su = mxGetDimensions( u );
    double *du = mxGetPr(u);
    unsigned int tu = su[0] * su[1];
    unsigned int i;
   
    for ( i = 0; i < tu; i++)
        cu[i] = (MPFLOAT ) du[i];

}

void
c_mpdcm_transfer_y(mxArray **y, MPFLOAT *cy, int *nx, int *ny, int *nt, int *nb)
{
    unsigned int i, k;
    unsigned int tt = nt[0] * nb[0];
    unsigned int td = nx[0] * ny[0];
    double *ta;

    for (i=0; i < tt; i++)
    {
        mxArray *ty = mxCreateDoubleMatrix(*nx, *ny, mxREAL);
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
    int *nx, int *ny, int *nu, int *dp, int *nt, int *nb, integrator integ)
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

    *nx = (int ) *mxGetPr( mxGetField(mxGetCell(theta, 0), 0, "dim_x") );
    *nu = mxGetDimensions( mxGetCell( u, 0) )[0];
    *dp = mxGetDimensions( mxGetCell( u, 0) )[1];
    *nt = stheta[0];
    *nb = stheta[1];

    *ny = ceil(dp[0] * mxGetPr(mxGetField(ptheta, 0, "dyu"))[0]) ;

    /* Offset */

    o = nx[0] * nx[0] + /*A*/
        nx[0] * nx[0] * nu[0] + /*B*/
        nx[0] * nu[0] + /*C*/
        nx[0] + /* Kappa */
        nx[0]; /* tau */


    /* Allocate memory */

    cy = (MPFLOAT *) malloc(nt[0] * nb[0] * nx[0] * ny[0] * sizeof( MPFLOAT ));
    if ( cy == NULL )
        mexErrMsgIdAndTxt("mpdcm:fmri:int:y:memory", "Memory error y");	

    cu = (MPFLOAT *) malloc(nt[0] * dp[0] * nu[0] * sizeof( MPFLOAT ));
    if ( cu == NULL )
        mexErrMsgIdAndTxt("mpdcm:fmri:int:u:memory", "Memory error u");	

    ctheta = (ThetaDCM *) malloc( nt[0] * nb[0] * sizeof( ThetaDCM ) );
    if ( ctheta == NULL )
        mexErrMsgIdAndTxt("mpdcm:fmri:int:theta:memory", "Memory error theta");	

    cptheta = (PThetaDCM *) malloc( sizeof( PThetaDCM ) );
    if ( cptheta == NULL )
        mexErrMsgIdAndTxt("mpdcm:fmri:int:ptheta:memory", 
            "Memory error ptheta");	

    dtheta = (MPFLOAT *) malloc(nt[0] * nb[0] * o * sizeof( MPFLOAT ));
    if ( dtheta == NULL )
        mexErrMsgIdAndTxt("mpdcm:fmri:int:theta:memory",
            "Memory error theta");	

    // Prepare u and theta

    for (i = 0; i < nt[0] * nb[0] ; i++ )
    {
        if ( i/nt[0] < 1 )
            c_mpdcm_prepare_u(mxGetCell(u, i), cu + i * nu[0] * dp[0]);

        c_mpdcm_prepare_theta(mxGetCell(theta, i), ctheta + i, dtheta + i*o);
    }

    // Prepare ptheta

    c_mpdcm_prepare_ptheta(ptheta, cptheta, dptheta);

    // Run the function

    (*integ)(cx, cy, cu, ctheta, dtheta, cptheta, dptheta,
        *nx, *ny, *nu, *dp, *nt, *nb);

    // Tranfer results

    *y = mxCreateCellMatrix(*nt, *nb);
    c_mpdcm_transfer_y(y, cy, nx, ny, nt, nb);

    free(cy);
    free(cu);
    free(ctheta);
    free(cptheta);
    free(dtheta);

}
