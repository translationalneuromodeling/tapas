/* aponteeduardo@gmail.com */
/* copyright (C) 2017 */

#include "host/DynamicsTrappenberg.hpp"

#define INDEX_NODES 0 // Nodes of the network
#define DIM_X_1_TRAPP 1 // Size of the dimensions

namespace Host
{

int
DynamicsTrappenberg::initialize_x(
    const arma::Mat< double > u,
    const mxArray *theta,
    const mxArray *ptheta,
    arma::Mat< double >& nx)
{
    mxArray *mx0 = mxGetField(theta, 0, "x0");
    arma::Mat< double > n0(mxGetPr(mx0), mxGetDimensions(mx0)[0], 
        DIM_X_1_TRAPP, 1);
    
    nx = n0;

    return 0;

}
 
int
DynamicsTrappenberg::update_y(
    const arma::Mat< double > x,
    const arma::Mat< double > u,
    const mxArray *theta,
    const mxArray *ptheta,
    arma::Mat< double > &y)
{

    y = x.col(INDEX_NODES);

    return 0;

}

arma::Mat< double >
DynamicsTrappenberg::sigma(
    const arma::Mat< double > x,
    const arma::Mat< double > u,
    const mxArray *theta,
    const mxArray *ptheta)
{
    arma::Mat< double > beta(mxGetPr(mxGetField(theta, 0, "beta")), 
        x.n_rows, 1, 1);
    arma::Mat< double > baseline(mxGetPr(mxGetField(theta, 0, "theta")), 
        x.n_rows, 1, 1);
   
    arma::Mat< double > dx = 1/(1 + exp(-x.col(INDEX_NODES) % beta + 
                baseline));

    return dx;

}


arma::Mat< double >
DynamicsTrappenberg::dynamics_x(
    const arma::Mat< double > x,
    const arma::Mat< double > u,
    const mxArray *theta,
    const mxArray *ptheta)
{
    mxArray *mA = mxGetField(theta, 0, "A");
    mxArray *mB = mxGetField(theta, 0, "B");
    mxArray *mC = mxGetField(theta, 0, "C");
    double tau = *mxGetPr(mxGetField(theta, 0, "tau"));

    arma::Mat< double > A(mxGetPr(mA), mxGetDimensions(mA)[0], 
                mxGetDimensions(mA)[1], 1);

    arma::Mat< double > B(mxGetPr(mB), mxGetDimensions(mB)[0], 
                mxGetDimensions(mB)[1], 1);

    arma::Mat< double > C(mxGetPr(mC), mxGetDimensions(mC)[0],
                mxGetDimensions(mC)[1], 1);

    arma::Mat< double > sx = sigma(x, u, theta, ptheta); 

    arma::Mat< double > dx = (A * x + B * sx + C * u)/tau;

    return dx ;
}


}
