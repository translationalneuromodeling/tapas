/* aponteeduardo@gmail.com */
/* copyright (C) 2016 */

#ifndef INTEGRATORHOST_H
#define INTEGRATORHOST_H

#include "mpdcm.h"
#include "mpdcm.hcu"
#include <matrix.h>
#include <mex.h>
#include <math.h>
#include <armadillo>
#include <iostream>

namespace Host
{

template <class T_update, class T_dynamics>
struct IntegratorHost
{
    const int dim_x;
    const int nts;
    const double dx;
    
    explicit IntegratorHost(int dm_x, int ns, double idx) :
        dim_x(dm_x), nts(ns), dx(idx) {};

    int
    integrate(const arma::Mat< double > u, 
        const mxArray *theta, 
        const mxArray *ptheta, 
        arma::Mat< double >& y) const;

};

template <class T_update, class T_dynamics>
int
IntegratorHost<T_update, T_dynamics>::integrate(
    const arma::Mat<double> u, 
    const mxArray *theta, 
    const mxArray *ptheta, 
    arma::Mat<double> &y) const
{
    int dy = u.n_cols / y.n_cols;
    int cy = 0, i, j;
    int nt = u.n_cols;

    arma::Mat< double > u0;
    arma::Mat< double > u1;
    arma::Mat< double > ox = T_dynamics::create_x(dim_x);
    arma::Mat< double > nx = T_dynamics::create_x(dim_x);
    arma::Mat< double > *tx, *tox = &ox, *tnx = &nx;

    T_dynamics::initialize_x(u.col(0), theta, ptheta, ox);

    arma::Mat< double > ty(y.colptr(cy), y.n_rows, 1, 0);
    T_dynamics::update_y(ox, u.col(0), theta, ptheta, ty);
    cy++;
    for (i = 1; i < nt; i++)
    {
        u0 = u.col(i - 1);
        u1 = u.col(i);

        for (j = 0; j < nts; j++)
        {
            arma::Mat< double > tu0;
            arma::Mat< double > tu1;
            if (nts == 1)
            {
                tu0 = u0;
                tu1 = u1;
            } else
            {
                tu0 = ((nts - j) / (double ) nts) * u0 + 
                    (j / (double ) nts) * u1;
                tu1 = ((nts - j + 1) / (double ) nts) * u0 + 
                    ((j + 1) / (double ) nts) * u1;
           }

            T_update::update_x(*tox, tu0, tu1, dx, theta, ptheta, *tnx);
            tx = tox;
            tox = tnx;
            tnx = tx;
        }

        if ( i % dy == 0 )
        {
            arma::Mat< double > ty(y.colptr(cy), y.n_rows, 1, 0);
            T_dynamics::update_y(*tox, u1, theta, ptheta, ty);
            cy++;
        }

    }
    
    return 0;
}


}

#endif // INTEGRATORHOST_H

