# README

aponteeduardo@gmail.com

## Introduction

mpdcm is a toolbox for efficient integration of the dynamical system defined
by DCM (see citations) on GPU architectures. Besides the core code, several 
routines for statistical inference are defined as part of our toolbox, 
including a map estimator, a mle estimator, variational Bayes hierarchical 
inference, thermodynamic integration, and population MCMC. All this tools 
are documented below.

## Compilation

Currently only LINUX is supported for compilation. We have compiled in a few
different platforms successfully. Most likely it will compile in MAC and 
Windows but we haven't yet tested those environments. 

Cuda version supported: 4.2 onward (testest until CUDA 7.0)

For compilation you require a matlab running environment, a cuda installation
and the supported compiler. If your standard compiler is not compatible with
your cuda installation, then, although it''s possible to compile, it might
require some manual work. For compilation you should use

 cd src
 ./configure
 make

If your cuda installation happens to be in a non-standard location, or you
would like to use a version of cuda installed in your computer but not the 
standard one use

 ./configure --with-cuda=<LOCATION OF YOUR CUDA INSTALLATION>

For example

 ./configure --with-cuda=/cluster/apps/cuda/4.2.9/

mpdcm compiles by default in single precision float numbers. Double precision
compilation is possible, but there is only a small difference in performance.


## Available routines

* c_mpdcm_fmri_int.m: 
* mpdcm_fmri_estimate.m: Thermodynamic integration for fmri with the same basic
    interface that spm.
* mpdcm_fmri_int_check_input.m: Checks that the input entered to the integrator
    fullfils the API.
* mpdcm_fmri_int.m: Integrates a set of DCM's. 
* mpdcm_fmri_k.m: k Values for DCM.
* mpdcm_fmri_llh.m: Log likelihood of the DCM.
* mpdcm_fmri_lpp.m: Log prior probability of DCM.
* mpdcm_fmri_map.m: Map estimator.
* mpdcm_fmri_tinput.m: Tranform SPM format to MPDCM compatible format.
* mpdcm_get_device.m: In case more that one Nvida card is available, gets the
    number of the active card. 
* mpdcm_num_devices.m: Returns the number of Nvidia cards.
* mpdcm_set_device.m: Sets the active card.
* mpdcm_update_kernel.m: 

# Change log:

v2.0.0.1: Fixed compilation issue in ./src/src/mpdcm.hcu.
