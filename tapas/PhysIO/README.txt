TAPAS PhysIO Toolbox Version 2015

************************************************************************
Copyright (C) 2012-2015 Lars Kasper <kasper@biomed.ee.ethz.ch>
Translational Neuromodeling Unit (TNU)
Institute for Biomedical Engineering
University of Zurich and ETH Zurich
------------------------------------------------------------------------

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
************************************************************************

PURPOSE

The PhysIO Toolbox provides model-based physiological noise correction of 
fMRI data using peripheral measures of respiration and cardiac pulsation. 
It incorporates noise models of cardiac/respiratory phase (RETROICOR, 
Glover et al. 2000), as well as heart rate variability and respiratory 
volume per time (cardiac response function, Chang et. al, 2009, respiratory 
response function, Birn et al. 2006). The toolbox is usable via the SPM 
batch editor, performs automatic pre-processing of noisy peripheral data 
and outputs nuisance regressor files directly suitable for SPM 
(multiple_regressors.txt).

BACKGROUND

The PhysIO Toolbox provides physiological noise correction for fMRI-data 
from peripheral measures (ECG/pulse oximetry, breathing belt). It is 
model-based, i.e. creates nuisance regressors from the physiological 
monitoring that can enter a General Linear Model (GLM) analysis, e.g. 
SPM8/12. Furthermore, for PHILIPS SCANPHYSLOG logfiles, it provides means 
to statistically assess peripheral data (e.g. heart rate variability) and 
recover imperfect measures (e.g. distorted R-peaks of the ECG).

Facts about physiological noise in fMRI:
- Physiological noise can explain 20-60 % of variance in fMRI voxel time 
  series (Birn2006, Hutton2011, Harvey2008.
	- Physiological noise affects a lot of brain regions (s. figure, e.g. 
      brainstem or OFC), especially next to CSF, arteries (Hutton2011). 
	- If not accounted for, this is a key factor limiting sensitivity for 
      effects of interest.
- Physiological noise contributions increase with field strength; they 
  become a particular concern at and above 3 Tesla (Kasper2009, Hutton2011).
- In resting state fMRI, disregarding physiological noise leads to wrong 
  connectivity results (Birn2006).

=> Some kind of physiological noise correction is highly recommended for 
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
		

FEATURES OF THIS TOOLBOX

Physiological Noise Modeling :
- Modeling physiological noise regressors from peripheral data 
  (breathing belt, ECG, pulse oximeter) 
- State of the art RETROICOR cardiac and respiratory phase expansion
- Cardiac response function (Chang et al, 2009) and respiratory response 
  function (Birn et al. 2006) modelling of heart-rate variability and 
  respiratory volume  per time influence on physiological noise
- Flexible expansion orders to model different contributions of cardiac, 
  respiratory and interaction terms (see Harvey2008, Hutton2011)
- Automatic creation of nuisance regressors, full integration into standard 
  GLMs, tested for SPM8/12 ("multiple_regressors.mat")

Philips SCANPHYSLOG-file handling:
The toolbox is dedicated to seamless integration into a clinical research s
etting and therefore offers correction methods to recover physiological 
data from imperfect peripheral measures.
For Philips SCANPHYSLOG-files, this includes
- Automatic alignment of scan volume timing and physiological time series 
  from logged gradient timecourses
- Automatic detection of ECG-R-peak events from raw ECG-signal, even if 
  online detection (and logging) was unsuccessful


COMPATIBILITY & SUPPORT

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
- Part of the TNU Software Edition: long term support and ongoing development

 
DOWNLOADS & RELEASE INFORMATION

- Current Release: 

PhysIO_Toolbox_15 (Code | Examples)
January 30th, 2015

revision: $Rev: 667 $

Minor Release Notes (r666):
- Compatibility tested for SPM12, small bugfixes Batch Dependencies
- Cleaner Batch Interface with grouped sub-menus (cfg_choice)
- new model: 'none' to just read out physiological raw data and preprocess,
  without noise modelling 
- Philips: Scan-timing via gradient log now automatized (gradient_log_auto)
- Siemens: Tics-Logfile read-in (proprietary, needs Siemens-agreement)
- All peak detections (cardiac/respiratory) now via auto_matched algorithm
- Adapt plots/saving for Matlab R2014b

Major Release Notes (r534):
- Read-in of Siemens plain text log files; new example dataset for Siemens
- Speed up and debugging of auto-detection method for noisy cardiac data => new method thresh.cardiac.initial_cpulse_select.method = ???auto_matched???
- Error handling for temporary breathing belt failures (Eduardo Aponte, TNU Zurich)
- slice-wise regressors can be created by setting sqpar.onset_slice to a index vector of slices

Major Release Notes (r497):
- SPM matlabbatch GUI implemented (Call via Batch -> SPM -> Tools -> TAPAS PhysIO Toolbox)
- improved, automatic heartbeat detection for noisy ECG now standard for ECG and Pulse oximetry (courtesy of Steffen Bollmann)
- QuickStart-Manual and PhysIO-Background presentation expanded/updated
- job .m/.mat-files created for all example datasets
- bugfixes cpulse-initial-select method-handling (auto/manual/load)

Major Release Notes (r429):
- Cardiac and Respiratory response function regressors integrated in workflow (heart rate and breathing volume computation)
- Handling of Cardiac and Respiratory Logfiles only
- expanded documentation (Quickstart.pdf and Handbook.pdf)
- read-in of custom log files, e.g. for BrainVoyager peripheral data
- more informative plots and commenting (especially in tapas_physio_new).

Minor Release Notes (r354):
- computation of heart and breathing rate in Philips/PPU/main_PPU.m
- prefix of functions with tapas_*

Major Release Notes (r241):
- complete modularization of reading/preprocessing/regressor creation for peripheral physiological data
- manual selection of missed heartbeats in ECG/pulse oximetry (courtesy of Jakob Heinzle)
- support for logfiles from GE scanners (courtesy of Steffen Bollmann, KiSpi Zuerich)
- improved detection of pulse oximetry peaks (courtesy of Steffen Bollmann)
- improved documentation
- consistent function names (prefixed by "physio_")

NOTE: Your main_ECG/PPU.m etc. scripts from previous versions (<=r159) will not work with this one any more. Please adapt one of the example scripts for your needs (~5 min of work). The main benefit of this version is a complete new variable structure that is more sustainable and makes the code more readable.


Lead Programmer: Lars Kasper, TNU & MR-Technology Group, IBT, University & ETH Zurich

Contributors: 
Steffen Bollmann, Children's Hospital Zurich & ETH Zurich
Jakob Heinzle, TNU Zurich
Eduardo Aponte, TNU Zurich

Send bug reports and suggestions to: kasper@biomed.ee.ethz.ch


TUTORIAL

Run main_ECG3T.m in subdirectory "examples" of the toolbox
See subdirectory "manual"


REFERENCES

Birn, Rasmus M., Jason B. Diamond, Monica A. Smith, and Peter A. Bandettini. 2006. Separating Respiratory-variation-related Fluctuations from Neuronal-activity-related Fluctuations in fMRI. NeuroImage 31 (4) (July 15): 1536?1548. 	doi:10.1016/j.neuroimage.2006.02.048.

Glover, G H, T Q Li, and D Ress. 2000. Image-based Method for Retrospective Correction of Physiological Motion Effects in fMRI: RETROICOR. Magnetic Resonance in Medicine: Official Journal of the Society of Magnetic Resonance in Medicine 44 (1) (July): 162(7). doi:10893535.

Harvey, Ann K., Kyle T.S. Pattinson, Jonathan C.W. Brooks, Stephen D. Mayhew, Mark Jenkinson, and Richard G. Wise. 2008. Brainstem Functional Magnetic Resonance Imaging: Disentangling Signal from Physiological Noise. Journal of Magnetic 		Resonance Imaging 28 (6): 1337?1344. doi:10.1002/jmri.21623.

Hutton, C., O. Josephs, J. Stadler, E. Featherstone, A. Reid, O. Speck, J. Bernarding, and N. Weiskopf. 2011. The Impact of Physiological Noise Correction on fMRI at 7 T. NeuroImage 57 (1) (July 1): 101?112. 	doi:10.1016/j.neuroimage.2011.04.018.

Josephs, O., Howseman, A.M., Friston, K., Turner, R., 1997. Physiological noise modelling for multi-slice EPI fMRI using SPM. Proceedings of the 5th Annual Meeting of ISMRM, Vancouver, Canada, p. 1682

Kasper, Lars, Sarah Marti, S. Johanna Vannesjo, Chloe Hutton, Ray Dolan, Nikolaus Weiskopf, Klaas Enno Stephan, and Klaas Paul Pruessmann. 2009. Cardiac Artefact Correction for Human Brainstem fMRI at 7 Tesla. In Proc. Org. Hum.  Brain Mapping 		15, 395. San Francisco.


VERSION OF THIS FILE
$Id: README.txt 667 2015-01-31 11:45:17Z kasperla $
