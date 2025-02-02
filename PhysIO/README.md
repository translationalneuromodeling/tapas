TAPAS PhysIO Toolbox 
====================

*Current version: Release 2024a, v9.0.0*

> Copyright (C) 2012-2025  
> Lars Kasper  
> <lars.kasper@alumni.ethz.ch>  
>  
> Translational Neuromodeling Unit (TNU)  
> Institute for Biomedical Engineering  
> University of Zurich and ETH Zurich  


Download
--------

- Please download the latest stable versions of the PhysIO Toolbox on GitHub as part of the 
  [TAPAS software releases of the TNU](https://github.com/translationalneuromodeling/tapas/releases).
- Older versions are available on the [TNU website](http://www.translationalneuromodeling.org/tapas).
- The latest bugfixes can be found in the 
  [development branch of TAPAS](https://github.com/translationalneuromodeling/tapas/tree/development) 
  and are announced in the [GitHub Issue Forum](https://github.com/translationalneuromodeling/tapas/issues). 
- Changes between all versions are documented in the 
  [CHANGELOG](https://gitlab.ethz.ch/physio/physio-doc/blob/master/CHANGELOG.md).


Purpose
-------

The general purpose of this Matlab toolbox is the model-based physiological noise 
correction of fMRI data using peripheral measures of respiration and cardiac 
pulsation. It incorporates noise models of cardiac/respiratory phase (RETROICOR, 
Glover et al. 2000), as well as heart rate variability and respiratory 
volume per time (cardiac response function, Chang et. al, 2009, respiratory 
response function, Birn et al. 2006), and extended motion models. 
While the toolbox is particularly well integrated with SPM via the Batch Editor GUI, its 
simple output nuisance regressor text files can be incorporated into any major 
neuroimaging analysis package.

Core design goals for the toolbox were: *flexibility*, *robustness*, and *quality assurance* 
to enable physiological noise correction for large-scale and multi-center studies. 

Some highlights:
1. Robust automatic preprocessing of peripheral recordings via iterative peak 
   detection and Hilbert transforms, validated in noisy data and patients.
2. Flexible support of peripheral data formats (Siemens, Philips, HCP, GE, Biopac, ...) 
   and noise models (RETROICOR, RVHRCOR).
3. Fully automated noise correction and performance assessment for group studies.
4. Integration in fMRI pre-processing pipelines as SPM Toolbox (Batch Editor GUI).

The accompanying technical paper about the toolbox concept and methodology 
can be found at: https://doi.org/10.1016/j.jneumeth.2016.10.019

Details on the peak-detection-free computation of respiratory volume and rate are outlined in our publication on the Hilbert transform: https://doi.org/10.1016/j.neuroimage.2021.117787

The PhysIO Toolbox is part of the TAPAS software collection of *Translational Algorithms for Psychiatry-Advancing Science*, whose design principles are described in the following paper: https://doi.org/10.3389/fpsyt.2021.680811.

There is also a video introduction into PhysIO and the TAPAS philosophy, given as part of the MRITogether23 Workshop on open science in MRI: [*PhysIO, UniQC and a TAPAStry of Tools* (20 min YouTube Video)](https://www.youtube.com/watch?v=zkPvQjLV2Is).

Installation
------------

### Matlab
1. Unzip the TAPAS archive in your folder of choice
2. Open Matlab
3. Go to `/your/path/to/tapas/physio/code`
4. Run `tapas_physio_init()` in Matlab


*Note*: Step (4) executes the following steps, which you could do manually as well.
- Adds the `physio/code/` folder to your Matlab path
- Adds SPM to your Matlab path (you can enter it manually, if not found)
- Links the folder (Linux/Max) or copies the folder (Windows) `physio/code/` to `/your/path/to/SPM/toolbox/PhysIO`, if the PhysIO code is not already found there  

Only the first point is necessary for using PhysIO standalone with Matlab.
The other two points enable PhysIO's SPM integration, i.e., certain functionality 
(Batch Editor GUI, pipeline dependencies, model assessment via F-contrasts).


Getting Started
---------------

...following the installation, you can try out an example:

1. Download the TAPAS examples via running `tapas_download_example_data()` 
   (found in `misc`-subfolder of TAPAS)
    - The PhysIO Example files will be downloaded to `tapas/examples/<tapas-version>/PhysIO`
2. Run `siemens_vb_ppu3t_sync_first_matlab_script.m` in subdirectory `Siemens_VB/PPU3T_Sync_First`
3. See subdirectory `physio/docs` and the next two section of this document for help.

You may try any of the examples in the other vendor folders as well.


Contact/Support
---------------

We are very happy to provide support on how to use the PhysIO Toolbox. However, 
as every researcher, we only have a limited amount of time. So please excuse, if 
we might not provide a detailed answer to your request, but just some general 
pointers and templates. Before you contact us, please try the following:

1. A first look at the [FAQ](https://gitlab.ethz.ch/physio/physio-doc/-/wikis/FAQ) 
   (which is frequently extended) might already answer your questions.
2. A lot of questions (before 2018) have also been discussed on our mailinglist 
   [tapas@sympa.ethz.ch](https://sympa.ethz.ch/sympa/info/tapas), 
   which has a searchable [archive](https://sympa.ethz.ch/sympa/arc/tapas).
3. For new requests, we would like to ask you to submit them as 
   [issues](https://github.com/translationalneuromodeling/tapas/issues) on our 
   github release page for TAPAS, which is also an up-to-date resource to 
   user-driven questions (since 2018).


Documentation
-------------

Documentation for this toolbox is provided in the following forms

1. Overview and guide to further documentation: README.md and CHANGELOG.md
    - [README.md](README.md): this file, purpose, installation, getting started, pointer to more help
    - [CHANGELOG.md](CHANGELOG.md): List of all toolbox versions and the respective release notes, 
      i.e. major changes in functionality, bugfixes etc.
2. User Guide: The markdown-based [GitLab Wiki](https://gitlab.ethz.ch/physio/physio-doc/-/wikis/home), including an FAQ
    - online (and frequently updated) at http://gitlab.ethz.ch/physio/physio-doc/-/wikis/home.
    - offline (with stables releases) as part of the toolbox in folder `physio/wikidocs`: 
        - plain text `.md` markdown files
        - as single HTML and PDF  file: `documentation.{html,pdf}`
3. Within SPM: All toolbox parameters and their settings are explained in the 
   Help Window of the SPM Batch Editor
4. Within Matlab: Extensive header at the start of each `tapas_physio_*` function and commenting
    - accessible via `help` and `doc` commands from Matlab command line
    - starting point for all parameters (comments within file): `edit tapas_physio_new` 
    - also useful for developers (technical documentation)
    

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
- Uncorrected physiological noise introduces serial correlations into the residual
  voxel time series, that invalidate assumptions on noise correlations (e.g., AR(1)) 
  used in data prewhitening by all major analysis packages. This issue is particularly
  aggravated at short TR (<1s), and most of its effects can be suitably addressed
  by physiological noise correction (Bollmann2018)

Therefore, some kind of physiological noise correction is highly recommended for
every statistical fMRI analysis.

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

### Flexible Read-in

The toolbox is dedicated to seamless integration into a clinical research 
setting and therefore offers correction methods to recover physiological 
data from imperfect peripheral measures. Read-in of the following formats is 
currently supported (alphabetic order):

- BioPac `.mat` and `.txt` export files
- Brain Imaging Data Structure files ([BIDS](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/physiological-and-other-continuous-recordings.html#recommendations-for-specific-use-cases)) `*_physio.tsv[.gz]/.json` files
- Custom logfiles: should contain one amplitude value per line, one logfile per 
  device. Sampling interval(s) are provided as a separate parameter to the toolbox.
- General Electric
- Philips SCANPHYSLOG files (all versions from release 2.6 to 5.3)
- Siemens Manual Recordings (acquired via `IdeaCmdTool`), named "Siemens" files in the PhysIO interface (files `.ecg`, `.resp`, `.puls`)
- Siemens Automatic Recordings (XA60 and above, or WIP Advanced Physiologging (AdvPhysio), or CMRR C2P BOLD sequences), named "Siemens Tics" in the PhysIO interface (files `*_ECG.log`, `*_RESP.log`, `*_PULS.log`, `*_Info.log`)
- Siemens Human Connectome Project (preprocessed files `*Physio_log.txt`)

See also the 
[Wiki page on Read-In](https://gitlab.ethz.ch/physio/physio-doc/-/wikis/MANUAL_PART_READIN) 
for a more detailed list and description of the supported formats.

### Robust Preprocessing

The toolbox has been extensively tested on different recording devices, field strengths and (clinical) study populations, and can handle a wide range of data imperfections in peripheral traces. Highlights are:

- Tested on various systems and subject populations (e.g., HCP, Children, 7T, ADHD, Major Depression, Psychosis)
- Cardiac Data: Iterative peak detection via Bayesian updates of current heart rate estimates (Steffen Bollmann)
- Respiratory Data: Novel volume estimation via Hilbert transform, respects, e.g., sighs, yawns, deep breaths (Samuel J. Harrison)

### Physiological Noise Modeling

- Modeling physiological noise regressors from peripheral data 
  (breathing belt, ECG, pulse oximeter) 
    - State of the art RETROICOR cardiac and respiratory phase expansion
    - Cardiac response function (Chang et al, 2009) and respiratory response 
      function (Birn et al. 2006) modelling of heart-rate variability and 
      respiratory volume  per time influence on physiological noise
    - Flexible expansion orders to model different contributions of cardiac, 
      respiratory and interaction terms (see Harvey2008, Hutton2011)
- Data-driven noise regressors
    - PCA extraction from nuisance ROIs (CSF, white matter), similar to aCompCor 
      (Behzadi2007)

### Automatization and Performance Assessment

- Automatic Quality Assurance Diagnostic Figures (Heart beat-to-beat interval curves, respiration amplitude histograms) and flagging of 
- Automatic creation of nuisance regressors, full integration into standard 
  GLMs, tested for SPM8/12 (`multiple_regressors.mat`)
    - Data export as text file available for usage in other analysis packages
- Integration in SPM Batch Editor: GUI for parameter input, dependencies to 
  integrate physiological noise correction in preprocessing pipeline
- Performance Assessment: Automatic F-contrast and tSNR Map creation and display 
  for groups of physiological noise regressors, using SPM GLM tools via 
  `tapas_physio_report_contrasts()`.


Requirements
------------

- All specific software requirements and their versions are in a separate file
  in this folder, `requirements.txt`.
- In brief:
  - Typically, PhysIO needs Matlab to run, and a few of its toolboxes
  - Some functionality requires SPM (GUI via the SPM Batch Editor, nuisance regression, contrast reporting, 
    writing residual and SNR images).    
  - There is a standalone version within [Neurodesk](https://www.neurodesk.org/tutorials-examples/tutorials/functional_imaging/physio/) that does not require Matlab


Contributors
------------

- Lead Programmer: 
    - [Lars Kasper](https://brain-to.ca/content/lars-kasper),
      TNU & MR-Technology Group, IBT, University of Zurich & ETH Zurich
- Project Team: 
    - Johanna Bayer, Donders Institute for Brain, Cognition and Behaviour, Nijmegen, Netherlands
    - Saskia Bollmann, Centre for Advanced Imaging, University of Queensland, Australia
- Project Team Alumni:
    - Steffen Bollmann, Centre for Advanced Imaging, University of Queensland, Australia
    - Sam Harrison, TNU Zurich
- Contributors (Code):
    - Eduardo Aponte, TNU Zurich
    - Tobias U. Hauser, FIL London, UK
    - Jakob Heinzle, TNU Zurich
    - Chloe Hutton, FIL London, UK (previously)
    - Miriam Sebold, Charite Berlin, Germany
    - External TAPAS contributors are listed in the [Contributor License Agreement](https://github.com/translationalneuromodeling/tapas/blob/master/Contributor-License-Agreement.md)
- Contributors (Examples):
    - listed in [EXAMPLES.md](https://gitlab.ethz.ch/physio/physio-doc/-/wikis/EXAMPLES)


Acknowledgements
----------------

The PhysIO Toolbox ships with the following publicly available code from other
open source projects and gratefully acknowledges their use.

- `utils\tapas_physio_propval.m`
    - `propval` function from Princeton MVPA toolbox (GPL)
      a nice wrapper function to create flexible propertyName/value optional
      parameters
- `utils\tapas_physio_fieldnamesr.m`
    - recursive parser for field names of a structure
    - Matlab file exchange, adam.tudorjones@pharm.ox.ac.uk

We further acknowledge the generous support from the MathWorks (MA, USA), who 
supported Johanna Bayer for the [2022 MATLAB Community Toolbox Training Program](https://www.incf.org/blog/second-summer-training-project-collaboration-between-incf-and-mathworks) 
established in collaboration with the International Neuroinformatics Coordinating Facility ([INCF](https://www.incf.org/)).


Cite Me
-------

### Main Toolbox and TAPAS Reference

Please cite the following papers in all of your publications that utilized the 
PhysIO Toolbox. 

1. Kasper, L., Bollmann, S., Diaconescu, A.O., Hutton, C., Heinzle, J., Iglesias, 
S., Hauser, T.U., Sebold, M., Manjaly, Z.-M., Pruessmann, K.P., Stephan, K.E., 2017. 
The PhysIO Toolbox for Modeling Physiological Noise in fMRI Data. 
Journal of Neuroscience Methods 276, 56-72. https://doi.org/10.1016/j.jneumeth.2016.10.019
    - *main PhysIO Toolbox reference*
2. Frässle, S., Aponte, E.A., Bollmann, S., Brodersen, K.H., Do, C.T., Harrison, O.K., Harrison, S.J., Heinzle, J., Iglesias, S., Kasper, L., Lomakina, E.I., Mathys, C., Müller-Schrader, M., Pereira, I., Petzschner, F.H., Raman, S., Schöbi, D., Toussaint, B., Weber, L.A., Yao, Y., Stephan, K.E., 2021. TAPAS: an open-source software package for Translational Neuromodeling and Computational Psychiatry. Frontiers in Psychiatry 12, 857. https://doi.org/10.3389/fpsyt.2021.680811
    - *main TAPAS software collection reference*

You can include the following snippet in your Methods section,
along with a brief description of the physiological noise models used: 

> The analysis was performed using the Matlab PhysIO Toolbox ([1], version x.y.z,
> open-source code available as part of the TAPAS software collection: [2], 
> <https://www.translationalneuromodeling.org/tapas>)

If you process respiratory traces, for example, to compute respiratory volume per time (RVT) or 
RETROICOR regressors, please also cite:

3. Harrison, S.J., Bianchi, S., Heinzle, J., Stephan, K.E., Iglesias, S., Kasper L., 2021.
A Hilbert-based method for processing respiratory timeseries.
NeuroImage, 117787. https://doi.org/10.1016/j.neuroimage.2021.117787
    - *superior RVT computation, preprocessing of respiratory traces*

Our [FAQ](https://gitlab.ethz.ch/physio/physio-doc/-/wikis/FAQ#3-how-do-i-cite-physio) 
contains a suggestion for a more comprehensive RETROICOR-related methods paragraph.
See the [main TAPAS README](../README.md) for more details on citing TAPAS itself.


### Related Papers (Implemented noise correction algorithms and optimal parameter choices)

The following sections list papers that 
- first implemented specific noise correction algorithms
- determined optimal parameter choices for these algorithms, depending on the
  targeted application
- demonstrate the impact of physiological noise and the importance of its correction

It is loosely ordered by the dominant physiological noise model used in the 
paper. The list is by no means complete, and we are happy to add any relevant papers 
suggested to us. 

#### RETROICOR 
4. Glover, G.H., Li, T.Q. & Ress, D. Image-based method for retrospective correction
of Physiological motion effects in fMRI: RETROICOR. Magn Reson Med 44, 162-7 (2000).

5. Hutton, C. et al. The impact of Physiological noise correction on fMRI at 7 T.
NeuroImage 57, 101-112 (2011).

6. Harvey, A.K. et al. Brainstem functional magnetic resonance imaging:
Disentangling signal from PhysIOlogical noise. Journal of Magnetic Resonance
Imaging 28, 1337-1344 (2008).

7. Bollmann, S., Puckett, A.M., Cunnington, R., Barth, M., 2018. 
Serial correlations in single-subject fMRI with sub-second TR. 
NeuroImage 166, 152-166. https://doi.org/10.1016/j.neuroimage.2017.10.043

#### aCompCor / Noise ROIs 
8. Behzadi, Y., Restom, K., Liau, J., Liu, T.T., 2007. A component based noise
correction method (CompCor) for BOLD and perfusion based fMRI. NeuroImage 37,
90-101. https://doi.org/10.1016/j.neuroimage.2007.04.042

#### RVT
9. Birn, R.M., Smith, M.A., Jones, T.B., Bandettini, P.A., 2008. The respiration response
function: The temporal dynamics of fMRI signal fluctuations related to changes in
respiration. NeuroImage 40, 644-654. doi:10.1016/j.neuroimage.2007.11.059
10. Jo, H.J., Saad, Z.S., Simmons, W.K., Milbury, L.A., Cox, R.W., 2010. 
Mapping sources of correlation in resting state FMRI, with artifact detection 
and removal. NeuroImage 52, 571-582. https://doi.org/10.1016/j.neuroimage.2010.04.246  
    - *regressor delay suggestions*

#### HRV
11. Chang, C., Cunningham, J.P., Glover, G.H., 2009. Influence of heart rate on the
BOLD signal: The cardiac response function. NeuroImage 44, 857-869.
doi:10.1016/j.neuroimage.2008.09.029
12. Shmueli, K., van Gelderen, P., de Zwart, J.A., Horovitz, S.G., Fukunaga, M., 
Jansma, J.M., Duyn, J.H., 2007. Low-frequency fluctuations in the cardiac rate 
as a source of variance in the resting-state fMRI BOLD signal. 
NeuroImage 38, 306-320. https://doi.org/10.1016/j.neuroimage.2007.07.037  
    - *regressor delay suggestions*

#### Motion (Censoring, Framewise Displacement)
13. Siegel, J.S., Power, J.D., Dubis, J.W., Vogel, A.C., Church, J.A., Schlaggar, B.L.,
Petersen, S.E., 2014. Statistical improvements in functional magnetic resonance
imaging analyses produced by censoring high-motion data points. Hum. Brain Mapp.
35, 1981-1996. https://doi.org/10.1002/hbm.22307

14. Power, J.D., Barnes, K.A., Snyder, A.Z., Schlaggar, B.L., Petersen, S.E., 2012. 
Spurious but systematic correlations in functional connectivity MRI networks 
arise from subject motion. NeuroImage 59, 2142-2154. 
https://doi.org/10.1016/j.neuroimage.2011.10.018
    - *definition of framewise displacement*

Copying/License
---------------

The PhysIO Toolbox is free software: you can redistribute it and/or
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
