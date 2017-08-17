TAPAS PhysIO Toolbox Version 2017
=================================

> Copyright (C) 2012-2017 Lars Kasper <kasper@biomed.ee.ethz.ch>

> Translational Neuromodeling Unit (TNU)

> Institute for Biomedical Engineering

> University of Zurich and ETH Zurich

Copying
-------

The PhysIO Toolbox is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program (see the file COPYING).  If not, see
<http://www.gnu.org/licenses/>.

Purpose
-------

The PhysIO Toolbox provides model-based physiological noise correction of 
fMRI data using peripheral measures of respiration and cardiac pulsation. 
It incorporates noise models of cardiac/respiratory phase (RETROICOR, 
Glover et al. 2000), as well as heart rate variability and respiratory 
volume per time (cardiac response function, Chang et. al, 2009, respiratory 
response function, Birn et al. 2006). The toolbox is usable via the SPM 
batch editor, performs automatic pre-processing of noisy peripheral data 
and outputs nuisance regressor files directly suitable for SPM 
(multiple_regressors.txt).

Installation
------------

### Matlab ###
- Unzip the TAPAS archive
- Add tapas/physio/code to your matlab path

### SPM ###
- Certain functionality (Batch Editor GUI, pipeline dependencies, model assessment via F-contrasts) require the installation of SPM
- Afterwards, the PhysIO Toolbox has to be registered as an SPM toolbox by copying the `physio/code` folder to `spm/toolbox/physio`


Getting Started
---------------

Run `example_main_ECG3T.m` in subdirectory `Philips/ECG3T` of the toolbox example repository `physio-examples`.
See subdirectory `physio/docs` and next section of this document.


Getting Help/Documentation
--------------------------

Several documentation files are provided with this toolbox. They have the extension .md (markdown), i.e. are plain text files, but can be conveniently viewed online as the github/gitlab Wiki.
You can find them in `physio/wikidocs`.

Alternatively, a pdf and html converted version of the following files is found in `physio/docs/documentation.pdf (or .html)`

List of Documentation files:
- README.md: this file, purpose, installation, getting started, pointer to more help
- FAQ.md: Frequently asked questions (for users)
- QUICKSTART.md: Example script and how to use on test data, Intro to Batch Editor GUI
- MANUAL.md: Reference Manual (mostly for developers) listing all functions, and rationales of the toolbox, disecting its modular structure
    - not provided yet; see the old version in `physio/docs/QuickStart_PhysIO_Toolbox.pdf` for source code documentation
- HOME.md: Landing Page of Wiki. Navigation to all other files and this explanation
- EXAMPLES.md: List and explanation of all examples
- CHANGELOG.md: List of all toolbox versions and the respective release notes, i.e. major changes in functionality, bugfixes etc.


Background
----------

The PhysIO Toolbox provides physiological noise correction for fMRI-data 
from peripheral measures (ECG/pulse oximetry, breathing belt). It is 
model-based, i.e. creates nuisance regressors from the physiological 
monitoring that can enter a General Linear Model (GLM) analysis, e.g. 
SPM8/12. Furthermore, for scanner vendor logfiles (PHILIPS, GE, Siemens), 
it provides means to statistically assess peripheral data (e.g. heart rate variability) 
and recover imperfect measures (e.g. distorted R-peaks of the ECG).

Facts about physiological noise in fMRI:
- Physiological noise can explain 20-60 % of variance in fMRI voxel time 
  series (Birn2006, Hutton2011, Harvey2008)
    - Physiological noise affects a lot of brain regions (s. figure, e.g. 
      brainstem or OFC), especially next to CSF, arteries (Hutton2011). 
    - If not accounted for, this is a key factor limiting sensitivity for effects of interest.
- Physiological noise contributions increase with field strength; they 
  become a particular concern at and above 3 Tesla (Kasper2009, Hutton2011).
- In resting state fMRI, disregarding physiological noise leads to wrong 
  connectivity results (Birn2006).

Therefore, some kind of physiological noise correction is highly recommended for every statistical fMRI analysis.

Model-based correction of physiological noise: 
- Physiological noise can be decomposed into periodic time series following 
  heart rate and breathing cycle.
- The Fourier expansion of cardiac and respiratory phases was introduced as 
  RETROICOR (RETROspective Image CORrection, Glover2000, 
  see also Josephs1997).
- These Fourier Terms can enter a General Linear Model (GLM) as nuisance 
  regressors, analogous to movement parameters.
- As the physiological noise regressors augment the GLM and explain 
  variance in the time series, they increase sensitivity in all contrasts 
  of interest.

		
Features of this Toolbox
------------------------

### Physiological Noise Modeling ###

- Modeling physiological noise regressors from peripheral data 
  (breathing belt, ECG, pulse oximeter) 
    - State of the art RETROICOR cardiac and respiratory phase expansion
    - Cardiac response function (Chang et al, 2009) and respiratory response 
      function (Birn et al. 2006) modelling of heart-rate variability and 
      respiratory volume  per time influence on physiological noise
    - Flexible expansion orders to model different contributions of cardiac, 
      respiratory and interaction terms (see Harvey2008, Hutton2011)
- Data-driven noise regressors
    - PCA extraction from nuisance ROIs (CSF, white matter), similar to aCompCor (Behzadi2007)

### Automatization and Performance Assessment ###
- Automatic creation of nuisance regressors, full integration into standard 
  GLMs, tested for SPM8/12 ("multiple_regressors.mat")
- Integration in SPM Batch Editor: GUI for parameter input, dependencies to integrate physiological noise correction in preprocessing pipeline
- Performance Assessment: Automatic F-contrast and tSNR Map creation and display for groups of physiological noise regressors, using SPM GLM tools

### Flexible Read-in ###

The toolbox is dedicated to seamless integration into a clinical research s
etting and therefore offers correction methods to recover physiological 
data from imperfect peripheral measures.

- General Electric
- Philips SCANPHYSLOG files (all versions from release 2.6 to 5.3)
- Siemens VB (files `.ecg`, `.resp`, `.puls`
- Siemens VD (files (`*_ECG.log`, `*_RESP.log`, `*_PULS.log`)
- Biopac .mat-export
    - assuming the following variables (as columns): `data`, `isi`, `isi_units`, `labels`, `start_sample`, `units`
    - See `tapas_physio_read_physlogfiles_biopac_mat.m` for details
- Custom logfiles: should contain one amplitude value per line, one logfile per device. Sampling interval(s) are provided as a separate parameter to the toolbox.



Compatibility and Support
-------------------------

- Matlab Toolbox
- Input: 
    - Fully integrated to work with physiological logfiles for Philips MR systems (SCANPHYSLOG)
    - tested for General Electric (GE) log-files
    - implementation for Siemens log-files
    - also: interface for 'Custom', i.e. general heart-beat time stamps 
      & breathing volume time courses from other log formats
- Output: 
    - Nuisance regressors for mass-univariate statistical analysis with SPM5,8,12
      or as text file for export to any other package
    - raw and processed physiological logfile data
- Part of the TAPAS Software Collection of the Translational Neuromodeling Unit (TNU) Zurich:long term support and ongoing development


Contributors
------------

- Lead Programmer: 
    - Lars Kasper, TNU & MR-Technology Group, IBT, University of Zurich & ETH Zurich
- Project Team: 
    - Steffen Bollmann, Centre for Advanced Imaging, University of Queensland, Australia
    - Saskia Bollmann, Centre for Advanced Imaging, University of Queensland, Australia
- Contributors:
    - Eduardo Aponte, TNU Zurich
    - Tobias U. Hauser, FIL London, UK
    - Jakob Heinzle, TNU Zurich
    - Chloe Hutton, FIL London, UK (previously)
    - Miriam Sebold, Charite Berlin, Germany


Contact
-------
Send bug reports and suggestions either to 
1) our mailing list: tapas@sympa.ethz.ch, or
2) as an issue on our TAPAS github account : https://github.com/translationalneuromodeling/tapas/issues


References
----------

### Main Toolbox Reference ###
1. Kasper, L., Bollmann, S., Diaconescu, A.O., Hutton, C., Heinzle, J., Iglesias, S., Hauser, T.U., Sebold, M., Manjaly, Z.-M., Pruessmann, K.P., Stephan, K.E., 2017. The PhysIO Toolbox for Modeling Physiological Noise in fMRI Data. Journal of Neuroscience Methods 276, 56–72. doi:10.1016/j.jneumeth.2016.10.019

### Related Papers (Implemented noise correction algorithms) ###
2. Glover, G.H., Li, T.Q. & Ress, D. Image‐based method for retrospective correction
of PhysIOlogical motion effects in fMRI: RETROICOR. Magn Reson Med 44, 162-7 (2000).

3. Hutton, C. et al. The impact of PhysIOlogical noise correction on fMRI at 7 T.
NeuroImage 57, 101‐112 (2011).

4. Harvey, A.K. et al. Brainstem functional magnetic resonance imaging:
Disentangling signal from PhysIOlogical noise. Journal of Magnetic Resonance
Imaging 28, 1337‐1344 (2008).

5. Behzadi, Y., Restom, K., Liau, J., Liu, T.T., 2007. A component based noise
correction method (CompCor) for BOLD and perfusion based fMRI. NeuroImage 37,
90–101. doi:10.1016/j.neuroimage.2007.04.042
    
6. Birn, R.M., Smith, M.A., Jones, T.B., Bandettini, P.A., 2008. The respiration response
function: The temporal dynamics of fMRI s ignal fluctuations related to changes in
respiration. NeuroImage 40, 644–654. doi:10.1016/j.neuroimage.2007.11.059
    
7. Chang, C., Cunningham, J.P., Glover, G.H., 2009. Influence of heart rate on the
BOLD signal: The cardiac response function. NeuroImage 44, 857–869.
doi:10.1016/j.neuroimage.2008.09.029
    
8. Siegel, J.S., Power, J.D., Dubis, J.W., Vogel, A.C., Church, J.A., Schlaggar, B.L.,
Petersen, S.E., 2014. Statistical improvements in functional magnetic resonance
imaging analyses produced by censoring high-motion data points. Hum. Brain Mapp.
35, 1981–1996. doi:10.1002/hbm.22307
