![rDCM Logo](misc/rDCM_Logo.png?raw=true "rDCM Logo")



rDCM - regression Dynamic Causal Modeling.
========================================================================

This ReadMe file provides the relevant information on the regression dynamic causal modeling (rDCM)
toolbox.


-------------------
General information
-------------------

- Authors: Stefan Frässle (<stefanf@biomed.ee.ethz.ch>), Ekaterina I. Lomakina
- Copyright (C) 2016-2021 
- Translational Neuromodeling Unit (TNU)
- Institute for Biomedical Engineering
- University of Zurich & ETH Zurich



--------
Download
--------

- Please download the latest stable versions of the rDCM Toolbox on GitHub as part of the 
  [TAPAS software releases of the TNU](https://github.com/translationalneuromodeling/tapas/releases).
- The latest bugfixes can be found in the [GitHub Issue Forum](https://github.com/translationalneuromodeling/tapas/issues) or by request to the authors. 
- Changes between all versions will be documented in the 
  [CHANGELOG](CHANGELOG.md).



-------
Purpose
-------

The regression dynamic causal modeling (rDCM) toobox implements a novel variant 
of DCM for fMRI that enables computationally efficient inference on effective (i.e.,
directed) connectivity parameters among brain regions. Due to its
computational efficiency, inversion of large (whole-brain) networks becomes feasible.

For the accompanying technical papers, detailing the methodology presented in this toolbox,
please see [Frässle et al., 2017](https://www.sciencedirect.com/science/article/pii/S105381191730201X?via%3Dihub) 
and [Frässle et al., 2018](https://www.sciencedirect.com/science/article/pii/S1053811918304762?via%3Dihub).



------------
Installation
------------

### Matlab ###
1. Unzip the TAPAS archive in your folder of choice
2. Open Matlab
3. Add the rDCM Toolbox to your Matlab path
4. Use the [Manual](docs/Manual.pdf) and the tutorial script `tapas_rdcm_tutorial()` as starting points



---------------
Important Notes
---------------

Please note that rDCM is a method that is still in an early stage of development and thus subject to 
various limiations. Due to these limitations, requirements of rDCM in terms of 
fMRI data quality (i.e., fast TR, high SNR) are high. For data that does not
meet these conditions, the method might not give reliable results. It remains the 
responsibility of each user to ensure that his/her dataset fulfills these 
requirements. Please refer to the main toolbox references (see below) for more 
detailed explanations.



---------------
Contact/Support
---------------

We are very happy to provide support on how to use the rDCM Toolbox. However, 
due to time constraints, we might not provide a detailed answer to your request, 
but just some general pointers and templates. Before you contact us, please try the following:

1. First, look at the [Manual](docs/Manual.pdf) and the tutorial script `tapas_rdcm_tutorial()` as starting points for answers to your questions.
2. For new requests, we would like to ask you to submit them as 
   [issues](https://github.com/translationalneuromodeling/tapas/issues) on our github release page for TAPAS.



----------
Cite Me
----------

Please cite the following paper (main TAPAS reference):
1. Frässle, S., Aponte, E.A., Bollmann, S., Brodersen, K.H., Do, C.T., Harrison, O.K., Harrison, S.J., Heinzle, J., Iglesias, S., Kasper, L., Lomakina, E.I., Mathys, C., Müller-Schrader, M., Pereira, I., Petzschner, F.H., Raman, S., Schöbi, D., Toussaint, B., Weber, L.A., Yao, Y., Stephan, K.E., 2021. TAPAS: an open-source software package for Translational Neuromodeling and Computational Psychiatry. Frontiers in Psychiatry 12, 857. https://doi.org/10.3389/fpsyt.2021.680811

In addition, please cite the following key references for the rDCM Toolbox:
2. Frässle, S., Lomakina, E.I., Razi, A., Friston, K.J., Buhmann, J.M., Stephan, K.E., 2017. Regression DCM for fMRI. NeuroImage 155, 406–421. doi:10.1016/j.neuroimage.2017.02.090
3. Frässle, S., Lomakina, E.I., Kasper, L., Manjaly Z.M., Leff, A., Pruessmann, K.P., Buhmann, J.M., Stephan, K.E., 2018. A generative model of whole-brain effective connectivity. NeuroImage 179, 505-529. doi:10.1016/j.neuroimage.2018.05.058

Finally, when using rDCM for resting-state fMRI data, feel free to also cite:
4. Frässle, S., Harrison, S.J., Heinzle, J., Clementz, B.A., Tamminga, C.A., Sweeney, J.A., Gershon, E.S., Keshavan, M.S., Pearlson, G.D., Powers, A., Stephan, K.E., 2021. Regression dynamic causal modeling for resting-state fMRI. Human Brain Mapping 42, 2159-2180. https://doi.org/10.1002/hbm.25357

You can include for example the following snippet in your Methods section:
> The analysis was performed using the regression dynamic causal modeling (rDCM) toolbox (Frässle et al., 2017; 2018), open-source code available as part of the TAPAS software collection (Frässle et al., 2021).



---------------
Copying/License
---------------

The rDCM Toolbox is free software: you can redistribute it and/or
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



--------------
Acknowledgment
--------------

We would like to highlight and acknowledge that the rDCM toolbox uses some 
functions that were publised as part of the Statistical Parameteric Mapping 
([SPM](https://www.fil.ion.ucl.ac.uk/spm/software/)) toolbox. The respective 
functions are marked with the prefix `tapas_rdcm_spm_`.
