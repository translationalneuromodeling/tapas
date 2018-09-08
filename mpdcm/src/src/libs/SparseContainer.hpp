/* aponteeduardo@gmail.com */
/* copyright (C) 2016 */

#ifndef SPARSECONTAINER_H
#define SPARSECONTAINER_H

#include <matrix.h>
#include <mex.h>
#include "mpdcm.hcu"
#include <vector>

struct SparseContainer{

    // Local memory
    std::vector< MPFLOAT > data;
    std::vector< unsigned long int > i_rows;
    std::vector< unsigned long int > i_cols;

    // Contained object
    sqsparse d_sparse;
    
    int d_alloc;

    //
    ~SparseContainer();
    // Flag whether memory has been allocated.
    SparseContainer() : d_alloc(0) {};

    void
    set_d_memory();

    void
    trans_h2d_memory();
 
    void
    trans_h2d_memory_async();
   
};

#endif // SPARSECONTAINER_H

