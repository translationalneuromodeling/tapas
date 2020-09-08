TAPAS ceode toolbox 
====================

*Current Version: Release 09.2020*

> Copyright (C) 2020  
> Dario Schöbi  
> <dschoebi(at)biomed.ee.ethz.ch>  
>  
> Translational Neuromodeling Unit (TNU)  
> Institute for Biomedical Engineering  
> University of Zurich and ETH Zurich  


Download
--------

- Please download the latest stable versions of the ceode Toolbox on GitHub as part of the 
  [TAPAS software releases of the TNU](https://github.com/translationalneuromodeling/tapas/releases).
- The latest bugfixes can be found in the 
  [development branch of TAPAS](https://github.com/translationalneuromodeling/tapas/tree/development) 
  and are announced in the [GitHub Issue Forum](https://github.com/translationalneuromodeling/tapas/issues). 


Purpose
-------

The general purpose of this Matlab toolbox is the robust estimation of convolution based Dynamic Causal Models (DCMs) for evoked responses (ERPs).
Core goals for the toolbox are: *flexibility*, *robustness*, and *quality assurance*. 
It ties modularly into the existing software suite (SPM12) for the estimation of Dynamic Causal Models for ERP.


Some highlights:
1. Efficient integration of delay differential equations underlying convolution based DCMs for ERPs for three (ERP model) and four (canonical microcircuit (CMC)) populations. The integration method uses the concept of a Continuous Extension of ODE methods (CEODE).


Installation and Usage
----------------------

1. Start MATLAB
3. Add SPM12 -> EEG toolboxes (e.g. type *spm eeg* in command window) to the MATLAB path
4. Replace *y{c} = spm_int_L(Q, M, U)* with *y{c} = tapas_ceode_int_euler(Q, M, U)* in *spm_gen_erp* (line 84)
	
    For three (ERP) / four (CMC) population DCMs
    
        Replace f = 'spm_fx_erp' with f = 'tapas_ceode_fx_erp' in spm_dcm_x_neural.m (line 53)
        Replace f = 'spm_fx_cmc' with f = 'tapas_ceode_fx_cmc' in spm_dcm_x_neural.m (line 79)

5. Run tapas_ceode_init()

    Your setup is compatible, if all of the following information is displayed:

    
    - Checking for SPM version: *You are currently using SPM version: SPM12 (XXXX)*

    - Running test scripts:
       *Successful test for your model (ERP/CMC).*
    
    - Checking integrator settings: 
       *Your setup currently runs with the tapas/ceode integrator.*
       (No compatibility warning with your model (ERP/CMC))



Tutorial
--------

To see the overall predictions by the integrators, run

    tapas_ceode_compare_integrators_erp() 
or

    tapas_ceode_compare_integrators_cmc()

This will create a figure showing the impact of the delays to the responses in a simple two region setup.
We assume a delay on the forward connection (from region 1 to region 2). 

To **change the delay magnitude**, adjust the spacing of tau in line 35 of

    tapas_ceode_compare_integrators_erp.m
and

    tapas_ceode_compare_integrators_cmc.m

respectively. 

To **change the parameters of the synthetic setup**, change

    tapas_ceode_synData_erp.m
or

    tapas_ceode_synData_cmc.m


Notes and Troubleshooting
-------------------------

### Failed test script

If only the test scripts fail, compare the scaling factors of the dynamical equations between the integration scheme. 
(see *spm_fx_erp/tapas_ceode_fx_erp* or *spm_fx_cmc/tapas_ceode_fx_cmc*, respectively).
also see: *Contact/Support*


### Changing the integration step size

By default, the stepsize of the integration for ceode is set to 1 ms .

To change the integration stepsize, either:

a) Add field DCM.M.intstep to the DCM.M structure. 
This defines a sampling rate for the integration.

b) In *tapas_ceode_int_euler.m* (line 44):
Replace default = 1E-3 to the desired subsampling rate.


### Scaling parameters for delays

To change the scaling parameters, either

a) Add field DCM.M.pF.D to the DCM.M structure.
This should define a 1x2 matrix with the desired scaling factors for delays.

b) In *tapas_ceode_compute_itau.m* (line 24):
Replace D = [2, 16] (ERP) or D = [1, 8] (CMC) to the desired scaling factors.
    


Contact/Support
---------------

In case you encouter problems with this toolbox, please refer to the [GitHub Issue Forum](https://github.com/translationalneuromodeling/tapas/issues).


Documentation
-------------

Documentation for this toolbox is provided in the following forms

1. Overview and guide to further documentation: README.md and CHANGELOG.md
    - [README.md](README.md): this file, purpose, installation, getting started, pointer to more help
    - [CHANGELOG.md](CHANGELOG.md): List of all toolbox versions and the respective release notes, 
      i.e. major changes in functionality, bugfixes etc.
2. For the general purpose documentation of DCM for ERPs, we refer to the SPM manual.


Compatibility
-------------
- MATLAB (tested with MATLAB R2017b)
- SPM12 (tested with SPM12, ver.7219 and ver.7771)


Contributors
------------

- Lead Programmer: 
    - [Dario Schöbi](https://www.tnu.ethz.ch/en/team/faculty-and-scientific-staff/schoebi),
      TNU, University of Zurich & ETH Zurich
- Coding and Revision:
    - Cao Tri Do, TNU Zürich
- Project Team: 
    - Jakob Heinzle, TNU Zurich
    - Klaas Enno Stephan, TNU Zurich



Requirements
------------
- MATLAB
- SPM12
(also see *Compatibility*)

References
----------

### Main Toolbox Reference

Schöbi D., Do C.T., Heinzle J., Stephan K.E., 
*Technical Note: A novel delay differential integration method for DCM for ERP*
(in prep)

Schöbi D.
*Dynamic causal models for inference on neuromodulatory processes in neural circuits.*
ETH Zürich (Dissertation; Chapter 2)


### Related Papers

1. Schöbi D., Jung F., Frässle S., Endepols H., Moran R.J., Friston K.J., Tittgemeyer M., Heinzle J., Stephan K.E.
Model-based prediction of muscarinic receptor function from auditory mismatch negativity responses.
bioRXiv. https://doi.org/10.1101/2020.06.08.139550



Copying/License
---------------

The ceode Toolbox is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program (see the file [LICENSE](LICENSE)).  If not, see
<http://www.gnu.org/licenses/>.
