/* aponteeduardo@gmail.com */
/* copyright (C) 2017 */

#ifndef FMRICONTAINER_H
#define FMRICONTAINER_H

namespace fmri {

template < 
    class T_y_Container, 
    class T_u_Container, 
    class T_theta_Container,
    class T_ackt_Container,
    class T_B_Container,
    class T_D_Container>
struct FmriContainer
{
    T_y_Container y_container;
    T_u_Container u_container;
    T_theta_Container theta_container;
    T_ackt_Container ackt_container;
    T_B_Container B_container;
    T_D_Container D_container;

    FmriContainer() :
        y_container(), u_container(), theta_container(),
        ackt_container(), B_container(), D_container()
        {};

    ~FmriContainer() {};
};

}

#endif // FMRICONTAINER_H

