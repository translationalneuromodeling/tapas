/* aponteeduardo@gmail.com */
/* dschoebi@biomed.ee.ethz.ch */
/* copyright (C) 2016 */

#include "host/DynamicsErp.hpp"
#include <iostream>

#define INDEX_PCVO 8 // Pyramidal Cell: Output Voltage
#define INDEX_PCVH 2 // Pyramidal Cell: Hyperpolarizing Voltage
#define INDEX_PCVD 1 // Pyramidal Cell: Depolarizing Voltage
#define INDEX_PCCH 5 // Pyramidal Cell: Hyperpolarizing Current
#define INDEX_PCCD 4 // Pyramidal Cell: Depolarizing Current
#define INDEX_ICV 6 // Inhibitory Cell: Voltage 
#define INDEX_ICC 7 // Inhibitory Cell: Current
#define INDEX_SCV 0 // Stellate Cell: Voltage
#define INDEX_SCC 3 // Stellate Cell: Current
#define INDEX_SIGPCVO 0 // Sigmoid Transform of Pyramidal Cell Output Voltage
#define INDEX_SIGSCV 1 // Sigmoid Transform of Stellate Cell Output Voltage
#define INDEX_SIGICV 2 // Sigmoid Transform of Inhibitory Cell Output Voltage


namespace Host
{

int
DynamicsErp::update_y(
    const arma::Mat< double > x, 
    const arma::Mat< double > u, 
    const mxArray *theta, 
    const mxArray *ptheta,
    arma::Mat< double > &y)
{
 
	y = x.col(INDEX_PCVO);

    return 0;
}   


int
DynamicsErp::initialize_x(
    const arma::Mat< double > u, 
    const mxArray *theta, 
    const mxArray *ptheta,
    arma::Mat< double >& nx)
{

    nx.zeros();

    return 0;
}   


arma::Mat< double > 
DynamicsErp::dynamics_x(
    const arma::Mat< double > x, 
    const arma::Mat< double > u, 
    const mxArray *theta, 
    const mxArray *ptheta)
{
    // Evaluate the function one at a time.
	arma::Mat<double > sx = saturation(x, u, theta, ptheta);
    arma::Mat< double > dx(x.n_rows, x.n_cols);
    dx.zeros();

	dynamics_f_dxA(x, sx, u, theta, ptheta, dx);
    dynamics_f_dxG(x, sx, u, theta, ptheta, dx);
    dynamics_f_dxC(x, sx, u, theta, ptheta, dx);
  
    return dx;
}


arma::Mat< double >
DynamicsErp::saturation(const arma::Mat< double > x, const arma::Mat< double > u, 
	const mxArray *theta, const mxArray *ptheta)
{
	double er1r2 = *mxGetPr(mxGetField(theta, 0, "er1r2"));
    double r1 = *mxGetPr(mxGetField(theta, 0, "r1"));
	double r2 = *mxGetPr(mxGetField(theta, 0, "r2"));
 	arma::Mat< double > sx(x.n_rows, 3);

	sx.col(INDEX_SIGPCVO) = 1 / (1 + exp(-r1 * (x.col(INDEX_PCVO) - r2))) - er1r2; 
	sx.col(INDEX_SIGSCV) = 1 / (1 + exp(-r1 * (x.col(INDEX_SCV) - r2))) - er1r2; 
	sx.col(INDEX_SIGICV) = 1 / (1 + exp(-r1 * (x.col(INDEX_ICV) - r2))) - er1r2; 
	return sx;
}

int
DynamicsErp::dynamics_f_dxA(
    const arma::Mat< double > x,
    const arma::Mat< double > sx,
    const arma::Mat< double > u, 
    const mxArray *theta, 
    const mxArray *ptheta, 
    arma::Mat< double >& dx)
{
    mxArray *mA13 = mxGetField(theta, 0, "A13");
    mxArray *mA23 = mxGetField(theta, 0, "A23");
    arma::Mat< double > A13(mxGetPr(mA13), mxGetDimensions(mA13)[0],
                mxGetDimensions(mA13)[1], 0);
    arma::Mat< double > A23(mxGetPr(mA23), mxGetDimensions(mA23)[0],
                mxGetDimensions(mA23)[1], 0);
	arma::Mat< double > tvec = A23 * sx.col(INDEX_SIGPCVO);


    dx.col(INDEX_SCC)  += A13 * sx.col(INDEX_SIGPCVO);
    dx.col(INDEX_ICC)  += tvec;
	dx.col(INDEX_PCCD) += tvec; 
	
//	std::cout << "f_dxA has passed ..." << std::endl;
    return 0;
}



int
DynamicsErp::dynamics_f_dxG(
    const arma::Mat< double > x,
    const arma::Mat< double > sx,
    const arma::Mat< double > u, 
    const mxArray *theta, 
    const mxArray *ptheta, 
    arma::Mat< double >& dx)
{
    mxArray *mgamma1 = mxGetField(theta, 0, "gamma1");
    arma::Mat< double> gamma1(mxGetPr(mgamma1), mxGetDimensions(mgamma1)[0], 
		mxGetDimensions(mgamma1)[1], 0);
    mxArray *mgamma2 = mxGetField(theta, 0, "gamma2");
    arma::Mat< double> gamma2(mxGetPr(mgamma2), mxGetDimensions(mgamma2)[0], 
		mxGetDimensions(mgamma2)[1], 0);
    mxArray *mgamma3 = mxGetField(theta, 0, "gamma3");
    arma::Mat< double> gamma3(mxGetPr(mgamma3), mxGetDimensions(mgamma3)[0], 
		mxGetDimensions(mgamma3)[1], 0);
    mxArray *mgamma4 = mxGetField(theta, 0, "gamma4");
    arma::Mat< double> gamma4(mxGetPr(mgamma4), mxGetDimensions(mgamma4)[0], 
		mxGetDimensions(mgamma4)[1], 0);
    mxArray *mtau_e2 = mxGetField(theta, 0, "tau_e2");
    arma::Mat< double> tau_e2(mxGetPr(mtau_e2), mxGetDimensions(mtau_e2)[0], 
		mxGetDimensions(mtau_e2)[1], 0);
    mxArray *mtau_i2 = mxGetField(theta, 0, "tau_i2");
    arma::Mat< double> tau_i2(mxGetPr(mtau_i2), mxGetDimensions(mtau_i2)[0], 
		mxGetDimensions(mtau_i2)[1], 0);
    mxArray *mtau_es2 = mxGetField(theta, 0, "tau_es2");
    arma::Mat< double> tau_es2(mxGetPr(mtau_es2), mxGetDimensions(mtau_es2)[0], 
		mxGetDimensions(mtau_es2)[1], 0);
    mxArray *mtau_is2 = mxGetField(theta, 0, "tau_is2");
    arma::Mat< double> tau_is2(mxGetPr(mtau_is2), mxGetDimensions(mtau_is2)[0], 
		mxGetDimensions(mtau_is2)[1], 0);
   
      
    dx.col(INDEX_PCVO) += x.col(INDEX_PCVD) - x.col(INDEX_PCVH);
    dx.col(INDEX_PCVH) += x.col(INDEX_PCCH);
    dx.col(INDEX_PCVD) += x.col(INDEX_PCCD);
    dx.col(INDEX_SCV)  += x.col(INDEX_SCC);
    dx.col(INDEX_ICV)  += x.col(INDEX_ICC);

    dx.col(INDEX_PCCH) += gamma4 % sx.col(INDEX_SIGICV)
			  - tau_i2 % x.col(INDEX_PCCH)
			  - tau_is2 % x.col(INDEX_PCVH);
    dx.col(INDEX_PCCD) += gamma2 % sx.col(INDEX_SIGSCV)
 			  - tau_e2 % x.col(INDEX_PCCD)
 			  - tau_es2 % x.col(INDEX_PCVD);
    dx.col(INDEX_SCC)  += gamma1 % sx.col(INDEX_SIGPCVO) 
			  - tau_e2 % x.col(INDEX_SCC)
 			  - tau_es2 % x.col(INDEX_SCV);
    dx.col(INDEX_ICC)  += gamma3 % sx.col(INDEX_SIGPCVO)
 			  - tau_e2 % x.col(INDEX_ICC)
			  - tau_es2 % x.col(INDEX_ICV);

// 	std::cout << "f_dxG has passed ..." << std::endl;

    return 0;
}

int
DynamicsErp::dynamics_f_dxC(
    const arma::Mat< double > x,
    const arma::Mat< double > sx,
    const arma::Mat< double > u, 
    const mxArray *theta, 
    const mxArray *ptheta, 
    arma::Mat< double >& dx)
{
    mxArray *mC = mxGetField(theta, 0, "Au");
    arma::Mat< double > C(mxGetPr(mC), mxGetDimensions(mC)[0],
                mxGetDimensions(mC)[1], 0);
   
	dx.col(INDEX_SCC) += C * u[0];

//   	std::cout << "f_dxC has passed ..." << std::endl;

}


}

