/* aponteeduardo@gmail.com */
/* copyright (C) 2015 */

# ifndef MPDCM_UTILS_H
# define MPDCM_UTILS_H

#include "mpdcm.hcu"
#include <matrix.h>
#include <mex.h>
#include <armadillo>
#include "SparseContainer.hpp"
#include "MatContainer.hpp"

namespace utils{

// Linearize a nxnxm matrix and return a container.
int
linearize_cube(const mxArray *atheta, const char *field, SparseContainer &);

int 
copy_theta_fields(const mxArray *theta, ThetaFmri &);

// Copy over the field of theta into a container.
int
gen_theta_container(const mxArray *, 
    MatContainer<ThetaFmriArray, ThetaFmri> &);

// Generate the array contiaing the fiels A, C, K and tau
int
gen_array_ACKtau(const mxArray *theta, 
    MatContainer<DataArray, MPFLOAT> &);

int
gen_u_container(const mxArray *u, MatContainer<DataArray, MPFLOAT> &);

int
gen_y_container(const mxArray *u, const mxArray *theta, 
        const mxArray *ptheta, MatContainer<DataArray, MPFLOAT> &cont_y);

int
unroll_container(const MatContainer<DataArray, MPFLOAT> &y, mxArray **o);

int linearize_sparse(const mxArray *ab, sqsparse *sm, int *ij, int *ii);
// Linearize an array of sparse matrices.
// 
// Input
//      ab      -- Cell array with the matrices
//      sm      -- Concatenated sparse matrices
//      ij      -- Index of the j array.
//      ii      -- Index of the i array.

template<class T_Container>
int
collect_container(T_Container &container_y, mxArray **output);

template<class T_Container>
int
collect_container(T_Container &container_y, mxArray **output)
{

    container_y.trans_d2h_memory();
    unroll_container(container_y, output);

    return 0;
}


template<class T_Theta, class T_Ptheta, class T_IntHost> 
int
run_kernel(const mxArray *u, const mxArray *theta, const mxArray *ptheta,
        mxArray **output);

template<class T_Theta, class T_Ptheta, class T_IntHost> 
int
run_kernel(const mxArray *u, const mxArray *theta, const mxArray *ptheta,
        mxArray **output)
{

    T_IntHost IntHost;
    T_Ptheta dptheta;
    
    IntHost.transverse_ptheta(ptheta, &dptheta);

    // Generate the u container.
    
    MatContainer< DataArray, MPFLOAT > cont_u;
    
    gen_u_container(u, cont_u);
    
    cont_u.set_d_memory();
    cont_u.trans_h2d_memory();
    
    // First transfer the linear part
    MatContainer< DataArray, MPFLOAT > cont_ackt;
    gen_array_ACKtau(theta, cont_ackt);
    
    cont_ackt.set_d_memory();
    cont_ackt.trans_h2d_memory();
    
    // Generate the container of the Fmri part 
    
    MatContainer<ThetaFmriArray, ThetaFmri> cont_theta;
    gen_theta_container(theta, cont_theta);

    // Put the pointer to the linear memory of A, C, K, and tau.
    cont_theta.d_array.linear = cont_ackt.d_array;
    
    //Now the sparse matrixes
    SparseContainer cont_B;
    linearize_cube(theta, "B", cont_B);
    
    cont_B.set_d_memory();
    cont_B.trans_h2d_memory();

    SparseContainer cont_D;
    linearize_cube(theta, "D", cont_D);
    cont_D.set_d_memory();
    cont_D.trans_h2d_memory();

    cont_theta.d_array.sB = cont_B.d_sparse;
    cont_theta.d_array.sD = cont_D.d_sparse;
    
    // Now transfer the memory
    cont_theta.set_d_memory();
    cont_theta.trans_h2d_memory();
    
    // Generate the container for y
    
    MatContainer < DataArray, MPFLOAT > cont_y;
    gen_y_container(u, theta, ptheta, cont_y); 

    cont_y.set_d_memory();

    IntHost.launch_kernel(
            &(cont_y.d_array), 
            &(cont_u.d_array), 
            &(cont_theta.d_array), 
            &dptheta);
    
    cont_y.trans_d2h_memory();
    unroll_container(cont_y, output);

    return 0;
}

template<class T_Theta, class T_Ptheta, class T_IntHost, class T_Container>
int
run_asynchronous_kernel(const mxArray *u, const mxArray *theta, 
        const mxArray *ptheta, T_Container &fmri_cont);

template<class T_Theta, class T_Ptheta, class T_IntHost, class T_Container> 
int
run_asynchronous_kernel(const mxArray *u, const mxArray *theta, 
        const mxArray *ptheta, T_Container &fmri_container)
{

    T_IntHost IntHost;
    T_Ptheta dptheta;
    
    IntHost.transverse_ptheta(ptheta, &dptheta);

    // Generate the u container.
    
    gen_u_container(u, fmri_container.u_container);
    
    fmri_container.u_container.set_d_memory();
    fmri_container.u_container.trans_h2d_memory_async();
    
    // First transfer the linear part

    gen_array_ACKtau(theta, fmri_container.ackt_container);
    
    fmri_container.ackt_container.set_d_memory();
    fmri_container.ackt_container.trans_h2d_memory_async();
    
    // Generate the container of the Fmri part 
    
    gen_theta_container(theta, fmri_container.theta_container);

    // Put the pointer to the linear memory of A, C, K, and tau.
    fmri_container.theta_container.d_array.linear = 
        fmri_container.ackt_container.d_array;
    
    //Now the sparse matrixes
    linearize_cube(theta, "B", fmri_container.B_container);
    
    fmri_container.B_container.set_d_memory();
    fmri_container.B_container.trans_h2d_memory_async();

    linearize_cube(theta, "D", fmri_container.D_container);
    fmri_container.D_container.set_d_memory();
    fmri_container.D_container.trans_h2d_memory_async();

    fmri_container.theta_container.d_array.sB = 
        fmri_container.B_container.d_sparse;

    fmri_container.theta_container.d_array.sD = 
        fmri_container.D_container.d_sparse;
    
    // Now transfer the memory
    fmri_container.theta_container.set_d_memory();
    fmri_container.theta_container.trans_h2d_memory_async();
    
    // Generate the container for y 
    gen_y_container(u, theta, ptheta, fmri_container.y_container); 
    fmri_container.y_container.set_d_memory();

    IntHost.launch_kernel(
            &(fmri_container.y_container.d_array), 
            &(fmri_container.u_container.d_array), 
            &(fmri_container.theta_container.d_array), 
            &dptheta);

    return 0;
}


template<class T_IntHost> 
int
run_host_fmri_kernel(const mxArray *u, const mxArray *theta, 
    const mxArray *ptheta, mxArray **output);

template<class T_IntHost> 
int
run_host_fmri_kernel(const mxArray *u, const mxArray *theta, 
    const mxArray *ptheta, mxArray **output)
{

    std::vector< arma::Mat < MPFLOAT > > dy;
    std::vector< arma::Mat < MPFLOAT > > du;

    T_IntHost int_host;

    du = int_host.populate_vec_mats(u);
    dy = int_host.init_y(u, theta, ptheta);

    int_host.clone_cell_mat(theta, output);
    int_host.populate_cell_mats(dy, output);

    return 0;
}


template<class T_Theta, class T_Ptheta, class T_IntHost> 
int
run_erp_kernel(const mxArray *u, const mxArray *theta, const mxArray *ptheta,
        mxArray **output);

template<class T_Theta, class T_Ptheta, class T_Host> 
int
run_erp_kernel(const mxArray *u, const mxArray *theta, const mxArray *ptheta,
        mxArray **output)
{

    
    DataArray du;
    DataArray dy;
    
    T_Theta h_theta;
    T_Ptheta h_ptheta;
    
    T_Host host;
   
    host.transverse_ptheta(ptheta, &h_ptheta);
 
    host.transverse_array(u, &du);
    host.set_host_array(&du);
    host.set_device_array(&du);
    host.linearize_array(u, &du);
    host.transfer_device_array(&du);
    
    host.transverse_theta(theta, &h_theta);
    host.set_host_theta(&h_theta);
    host.set_device_theta(&h_theta);
    host.linearize_theta(theta, &h_theta);
    host.transfer_device_theta(&h_theta); 

    host.initialize_y(&du, &h_theta, &h_ptheta, &dy);
    host.set_host_array(&dy);
    host.set_device_array(&dy);

    host.launch_kernel(&dy, &du, &h_theta, &h_ptheta);
    host.transfer_host_array(&dy);
    host.unroll_array(&dy, output);

    //Clean up
   
    host.clean_host_theta(&h_theta);
    host.clean_device_theta(&h_theta);    

    host.clean_host_array(&du);
    host.clean_device_array(&du);

    host.clean_host_array(&dy);
    host.clean_device_array(&dy);
    
}


} // Utils namespace

# endif



