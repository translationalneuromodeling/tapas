/* aponteeduardo@gmail.com */
/* copyright (C) 2016 */

#ifndef INTERFACEHOST_H
#define INTERFACEHOST_H

#include "mpdcm.h"
#include "mpdcm.hcu"
#include <armadillo>
#include <matrix.h>

namespace Host
{

class InterfaceHost 
{  

    public:

        std::vector< arma::Mat < double > >
        populate_vec_mats(const mxArray *mvals);

        int
        populate_cell_mats(const std::vector< arma::Mat < double > >,
                mxArray **);

        int
        clone_cell_mat(const mxArray *origin, mxArray **target);

};

}
#endif // INTERFACEHOST_H

