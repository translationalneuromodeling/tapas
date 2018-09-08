/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */

# ifndef MPDCM_UTILS_H
# define MPDCM_UTILS_H

#include "mpdcm.h"
#include "mpdcm.hcu"
#include <matrix.h>
#include <mex.h>
#include <armadillo>

namespace utils{

template<class T_Interface> 
int
run_interface_host_kernel(const mxArray *u, const mxArray *theta, 
    const mxArray *ptheta, mxArray **output);

template<class T_Interface> 
int
run_interface_host_kernel(const mxArray *u, const mxArray *theta, 
    const mxArray *ptheta, mxArray **output)
{

    std::vector< arma::Mat < double > > dy;
    std::vector< arma::Mat < double > > du;

    T_Interface interface;

    du = interface.populate_vec_mats(u);
    dy = interface.init_y(u, theta, ptheta);

    interface.integrate(du, theta, ptheta, dy);

    interface.clone_cell_mat(theta, output);
    interface.populate_cell_mats(dy, output);

    return 0;
}


template<class T_Interface> 
int
run_interface_fmri_kernel(const mxArray *u, const mxArray *theta, 
    const mxArray *ptheta, mxArray **output);

template<class T_Interface> 
int
run_interface_fmri_kernel(const mxArray *u, const mxArray *theta, 
    const mxArray *ptheta, mxArray **output)
{

    std::vector< arma::Mat < double > > dy;
    std::vector< arma::Mat < double > > du;

    T_Interface interface;

    du = interface.populate_vec_mats(u);
    dy = interface.init_y(u, theta, ptheta);

    interface.integrate(du, theta, ptheta, dy);

    interface.clone_cell_mat(theta, output);
    interface.populate_cell_mats(dy, output);

    return 0;
}

template<class T_Interface> 
int
run_interface_erp_kernel(const mxArray *u, const mxArray *theta, 
    const mxArray *ptheta, mxArray **output);

template<class T_Interface> 
int
run_interface_erp_kernel(const mxArray *u, const mxArray *theta, 
    const mxArray *ptheta, mxArray **output)
{

    std::vector< arma::Mat < double > > dy;
    std::vector< arma::Mat < double > > du;

    T_Interface interface;

    du = interface.populate_vec_mats(u);
    dy = interface.init_y(u, theta, ptheta);

    interface.integrate(du, theta, ptheta, dy);

    interface.clone_cell_mat(theta, output);
    interface.populate_cell_mats(dy, output);

    return 0;
}




} // Utils namespace

# endif

