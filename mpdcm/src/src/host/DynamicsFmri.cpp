/* aponteeduardo@gmail.com */
/* copyright (C) 2016 */

#include "host/DynamicsFmri.hpp"
#include <iostream>
#define INDEX_X 0
#define INDEX_F 1
#define INDEX_S 2
#define INDEX_V 3
#define INDEX_Q 4

namespace Host
{

int
DynamicsFmri::update_y(
    const arma::Mat< double > x, 
    const arma::Mat< double > u, 
    const mxArray *theta, 
    const mxArray *ptheta,
    arma::Mat< double > &y)
{
    y.zeros();  

    dynamics_g(x, u, theta, ptheta, y);

    return 0;
}   


int
DynamicsFmri::initialize_x(
    const arma::Mat< double > u, 
    const mxArray *theta, 
    const mxArray *ptheta, arma::Mat< double >& nx)
{

    nx.zeros();

    return 0;
}   


arma::Mat< double > 
DynamicsFmri::dynamics_x(
    const arma::Mat< double > x, 
    const arma::Mat< double > u, 
    const mxArray *theta, 
    const mxArray *ptheta)
{
    // Evaluate the function one at a time.

    arma::Mat< double > dx(x.n_rows, x.n_cols);
    dx.zeros();
 
    dynamics_f_dx(x, u, theta, ptheta, dx);
    dynamics_f_df(x, u, theta, ptheta, dx);
    dynamics_f_ds(x, u, theta, ptheta, dx);
    dynamics_f_dv(x, u, theta, ptheta, dx);
    dynamics_f_dq(x, u, theta, ptheta, dx);
    
    return dx;
}


int
DynamicsFmri::dynamics_f_dx(
    const arma::Mat< double > x, 
    const arma::Mat< double > u, 
    const mxArray *theta, 
    const mxArray *ptheta, 
    arma::Mat< double >& dx)
{

    dynamics_f_dxA(x, u, theta, ptheta, dx);
    dynamics_f_dxC(x, u, theta, ptheta, dx);
    if ( (int ) *mxGetPr(mxGetField(theta, 0, "fB")) )
        dynamics_f_dxB(x, u, theta, ptheta, dx);
    if ( (int ) *mxGetPr(mxGetField(theta, 0, "fD")) )
        dynamics_f_dxD(x, u, theta, ptheta, dx);

    return 0;
}

int
DynamicsFmri::dynamics_f_dxA(
    const arma::Mat< double > x, 
    const arma::Mat< double > u, 
    const mxArray *theta, 
    const mxArray *ptheta, 
    arma::Mat< double >& dx)
{
    mxArray *mA = mxGetField(theta, 0, "A");
    arma::Mat< double> A(mxGetPr(mA), mxGetDimensions(mA)[0],
                mxGetDimensions(mA)[1], 0);
    
    dx.col(INDEX_X) += A * x.col(INDEX_X);

    return 0;    
}

int
DynamicsFmri::dynamics_f_dxB(
    const arma::Mat< double > x, 
    const arma::Mat< double > u, 
    const mxArray *theta, 
    const mxArray *ptheta, 
    arma::Mat< double >& dx)
{
    int i;
    mxArray *mB = mxGetField(theta, 0, "B");
    arma::Cube< double > B(mxGetPr(mB), mxGetDimensions(mB)[0],
                mxGetDimensions(mB)[1], mxGetDimensions(mB)[2], 0);
    
    for (i = 0; i < u.n_rows; i++)
    {
        if (u(i) == 0 )
            continue;
        dx.col(INDEX_X) += u(i) * B.slice(i) * x.col(INDEX_X);
    }

    return 0;    
}

int
DynamicsFmri::dynamics_f_dxC(
    const arma::Mat< double > x, 
    const arma::Mat< double > u, 
    const mxArray *theta, 
    const mxArray *ptheta, 
    arma::Mat< double >& dx)
{
    mxArray *mC = mxGetField(theta, 0, "C");
    arma::Mat< double > C(mxGetPr(mC), mxGetDimensions(mC)[0],
                mxGetDimensions(mC)[1], 0);
    dx.col(INDEX_X) += C * u;

    return 0;    
}

int
DynamicsFmri::dynamics_f_dxD(
    const arma::Mat< double > x, 
    const arma::Mat< double > u, 
    const mxArray *theta, 
    const mxArray *ptheta, 
    arma::Mat< double >& dx)
{
    int i;
    mxArray *mD = mxGetField(theta, 0, "D");
    arma::Cube< double > D(mxGetPr(mD), mxGetDimensions(mD)[0],
                mxGetDimensions(mD)[1], mxGetDimensions(mD)[2], 0);
    
    for (i = 0; i < x.n_rows; i++)
        dx.col(INDEX_X) += x.col(INDEX_X)[i] * D.slice(i) * x.col(INDEX_X);

    return 0;    
}

int
DynamicsFmri::dynamics_f_ds(
    const arma::Mat< double > x, 
    const arma::Mat< double > u, 
    const mxArray *theta, 
    const mxArray *ptheta, 
    arma::Mat< double >& dx)
{
    double gamma = *mxGetPr(mxGetField(theta, 0, "gamma"));
    arma::Mat< double > K(mxGetPr(mxGetField(theta, 0, "K")), x.n_rows, 1, 0);

    dx.col(INDEX_S) += x.col(INDEX_X) - (K % x.col(INDEX_S)) -
        gamma * (exp(x.col(INDEX_F)) - 1);

    return 0;
}

int
DynamicsFmri::dynamics_f_df(
    const arma::Mat< double > x, 
    const arma::Mat< double > u, 
    const mxArray *theta, 
    const mxArray *ptheta,
    arma::Mat< double >& dx)
{

    dx.col(INDEX_F) += x.col(INDEX_S) % exp(-x.col(INDEX_F));
    
    return 0;
}


int
DynamicsFmri::dynamics_f_dv(
    const arma::Mat< double > x, 
    const arma::Mat< double > u, 
    const mxArray *theta, 
    const mxArray *ptheta,
    arma::Mat< double >& dx)
{
    arma::Mat< double > tau(mxGetPr(mxGetField(theta, 0, "tau")), x.n_rows,
            1, 0);
    double alpha = *mxGetPr(mxGetField(theta, 0, "alpha"));

    dx.col(INDEX_V) += exp(x.col(INDEX_F) - x.col(INDEX_V) - tau) -
        exp(alpha * x.col(INDEX_V) - tau);

    return 0;
}


int
DynamicsFmri::dynamics_f_dq(
    const arma::Mat< double > x, 
    const arma::Mat< double > u, 
    const mxArray *theta, 
    const mxArray *ptheta,
    arma::Mat< double >& dx)
{
    arma::Mat< double > tau(mxGetPr(mxGetField(theta, 0, "tau")), x.n_rows,
            1, 0);
    double alpha = *mxGetPr(mxGetField(theta, 0, "alpha"));
    double E0 = *mxGetPr(mxGetField(theta, 0, "E0"));
    double lnE0 = log(E0);
    double ln1_E0 = log(1 - E0);
 
    dx.col(INDEX_Q) += (1.0 - exp(ln1_E0 * exp(-x.col(INDEX_F)))) % 
            exp(x.col(INDEX_F) - lnE0 - tau - x.col(INDEX_Q)) - 
            exp(alpha * x.col(INDEX_V) - tau);

    return 0;
}


int
DynamicsFmri::dynamics_g(
    const arma::Mat< double > x, 
    const arma::Mat< double > u, 
    const mxArray *theta, 
    const mxArray *ptheta, 
    arma::Mat< double >& y)
{
    int i;
    double k1 = *mxGetPr(mxGetField(theta, 0, "k1"));
    double k2 = *mxGetPr(mxGetField(theta, 0, "k2"));
    double k3 = *mxGetPr(mxGetField(theta, 0, "k3"));
    double v0 = *mxGetPr(mxGetField(theta, 0, "V0"));
   
    y = k1 * ( 1 - exp(x.col(INDEX_Q))) + 
        k2 * ( 1 - exp(x.col(INDEX_Q) - x.col(INDEX_V))) +
        k3 * ( 1 - exp(x.col(INDEX_V)));

    y *= v0;

    return 0;

}

}

