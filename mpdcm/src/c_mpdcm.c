/* aponteeduardo@gmail.com */
/* copyright (C) 2014 */

#include "mpdcm.hcu"
#include "c_mpdcm.h"

#define DIM_X 5

void mpdcm_prepare_theta(const mxArray *theta, ThetaDCM *ctheta, double *dtheta)
{

    int o=0, i, j;
    
    ctheta->dim_x = (int ) *mxGetPr(mxGetField(theta, 0, "dim_x"));
    ctheta->dim_u = (int ) *mxGetPr(mxGetField(theta, 0, "dim_u"));

    ctheta->fA = *mxGetPr(mxGetField(theta, 0, "fA")) ? MF_TRUE : MF_FALSE;
    ctheta->fB = *mxGetPr(mxGetField(theta, 0, "fB")) ? MF_TRUE : MF_FALSE;
    ctheta->fC = *mxGetPr(mxGetField(theta, 0, "fC")) ? MF_TRUE : MF_FALSE;

    ctheta->V0 = *mxGetPr(mxGetField(theta, 0, "V0"));
    ctheta->E0 = *mxGetPr(mxGetField(theta, 0, "E0"));
    ctheta->k1 = *mxGetPr(mxGetField(theta, 0, "k1"));
    ctheta->k2 = *mxGetPr(mxGetField(theta, 0, "k2"));
    ctheta->k3 = *mxGetPr(mxGetField(theta, 0, "k3"));

    ctheta->alpha = *mxGetPr(mxGetField(theta, 0, "alpha"));
    ctheta->gamma = *mxGetPr(mxGetField(theta, 0, "gamma"));
    
    i = ctheta->dim_x*ctheta->dim_x;

    memcpy(dtheta,
        mxGetPr(mxGetField(theta, 0, "A")),
        i*sizeof(double) );

    o += i;

    for (j=0; j < ctheta->dim_u; j++)
    {
        memcpy(dtheta + o, mxGetPr(mxGetCell(mxGetField(theta, 0, "B"), j)),
            i * sizeof(double) );
        o += i;
    }

    i = ctheta->dim_x*ctheta->dim_u;
    memcpy(dtheta + o, mxGetPr(mxGetField(theta, 0, "C")), i * sizeof(double));
    o += i;

    i = ctheta->dim_x;
    memcpy(dtheta + o, mxGetPr(mxGetField(theta, 0, "epsilon")),
        i * sizeof(double));
    o += i;

    i = ctheta->dim_x;
    memcpy(dtheta + o, mxGetPr(mxGetField(theta, 0, "K")),
        i * sizeof(double));
    o += i;

    i = ctheta->dim_x;
    memcpy(dtheta + o, mxGetPr(mxGetField(theta, 0, "tau")),
        i * sizeof(double));
    o += i;

}

void mpdcm_prepare_ptheta(const mxArray *ptheta, void *vptheta, double *dptheta)
{

    PThetaDCM *cptheta = (PThetaDCM *) vptheta;
    cptheta->dt = *mxGetPr(mxGetField(ptheta, 0, "dt"));
    cptheta->dyu = *mxGetPr(mxGetField(ptheta, 0, "dyu"));
    cptheta->mode = 'f';

}

void mpdcm_prepare_u(const mxArray *u, double *cu)
{
    const mwSize *su = mxGetDimensions( u );

    memcpy(cu, mxGetPr(u), su[0] * su[1] * sizeof(double));

}

void mpdcm_prepare_input(
    mxArray **y, const mxArray *u, const mxArray *theta, const mxArray *ptheta,
    int *nx, int *ny, int *nu, int *dp, int *nt, int *nb)
    /* Prepares the data in the format necessary to interface with the
    cuda library. */
{

    const mwSize *su = mxGetDimensions( u );
    const mwSize *stheta = mxGetDimensions( theta );
    int i, o;

    double *cx, *cy, *cu;
    ThetaDCM *ctheta;
    PThetaDCM *cptheta;
    double *dtheta;
    double *dptheta;

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
        nx[0] + /* epsilon */
        nx[0] + /* Kappa */
        nx[0]; /* tau */


    /* Allocate memory */

    cy = (double *) malloc(nt[0] * nb[0] * nx[0] * ny[0] * sizeof(double));
    cu = (double *) malloc(nt[0] * dp[0] * nu[0] * sizeof(double));
    ctheta = (ThetaDCM *) malloc( nt[0] * nb[0] * sizeof(ThetaDCM) );
    cptheta = (PThetaDCM *) malloc( sizeof(PThetaDCM) );
    dtheta = (double *) malloc(nt[0] * nb[0] * o * sizeof(double));

    if ( cy == NULL ){
            mexErrMsgIdAndTxt("mpdcm:fmri:int:y:memory",
                "Memory error y");	
    }
    if ( cu == NULL ){
            mexErrMsgIdAndTxt("mpdcm:fmri:int:u:memory",
                "Memory error u");	
    }
    if ( ctheta == NULL ){
            mexErrMsgIdAndTxt("mpdcm:fmri:int:theta:memory",
                "Memory error theta");
    }
    if ( cptheta == NULL ){
            mexErrMsgIdAndTxt("mpdcm:fmri:int:ptheta:memory",
                "Memory error ptheta");	
    }
    if ( dtheta == NULL ){
            mexErrMsgIdAndTxt("mpdcm:fmri:int:theta:memory",
                "Memory error theta");	
    }

    for (i = 0; i < nt[0] * nb[0] ; i++ )
    {
        if ( i/nt[0] < 1 )
        {
            mpdcm_prepare_u(mxGetCell(u, i), cu + i * nu[0] * dp[0]);
        }

        mpdcm_prepare_theta(mxGetCell(theta, i), ctheta + i, dtheta + i*o);
    }

    mpdcm_prepare_ptheta(ptheta, cptheta, dptheta);

    /* run the function */

    /*printf("nx: %d, ny:%d, nu:%d, dp:%d, nt:%d, nb:%d", 
            *nx, *ny, *nu, *dp, *nt, *nb);*/

    mpdcm_fmri(cx, cy, cu, ctheta, dtheta, cptheta, dptheta,
        *nx, *ny, *nu, *dp, *nt, *nb);

    *y = mxCreateCellMatrix(*nt, *nb);

    for (i=0; i<nt[0]*nb[0]; i++)
    {
        mxArray *ty = mxCreateDoubleMatrix(*nx, *ny, mxREAL); 
        memcpy(mxGetPr(ty), cy + i * nx[0] * ny[0],
             nx[0] * ny[0] * sizeof(double));
        mxSetCell(*y, i, ty);
    }

    free(cy);
    free(cu);
    free(ctheta);
    free(cptheta);
    free(dtheta);

    /* free dptheta) */

}

void mpdcm_fmri_int(mxArray **y, const mxArray **u,
    const mxArray **theta, const mxArray **ptheta,
    int *nx, int *ny, int *nu, int *dp, int *nt, int *nb)
{
    int i;
    double **cu, **dtheta, **dptheta;
    void **ctheta, **cptheta;

    
    mpdcm_prepare_input(y, *u, *theta, *ptheta,
        nx, ny, nu, dp, nt, nb);
}
