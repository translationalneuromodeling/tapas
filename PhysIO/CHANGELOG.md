RELEASE INFORMATION
===============================

Current Release
---------------

PhysIO_Toolbox_R2018.1.1

September 05, 2018

Bugfix Release Notes (R2018.1.1)
-------------------------------

### Changed
- documentation.{html,pdf} export nicer with different FAQ numbering

Major Release Notes (R2018.1)
-----------------------------

### Added
- initialization function `tapas_physio_init()` to check Matlab paths, including SPM for batch processing
- Extended motion diagnostics via Framewise displacement (Power et al., 2012)
    - Outlier motion models generate 'spike' regressors from FD outliers (gitlab issue  #)
- Censoring of intervals with bad physiological recordings in RETROICOR regressors (github issue #11, gitlab #36)
- Added examples of Siemens VD (Tics Format, Prisma) and Human Connectome Project (HCP) format

### Changed
- Updated read-in examples of all vendors (Siemens, Philips, GE) to latest PhysIO Toolbox version.
- Updated `README.md` to reflect changes to example download, new references
- Extended Wiki documentation, in particular examples and read-in formats

Minor Release Notes (R2017.3)
-----------------------------

- Included references to external [ETH gitlab physio-doc repo and wiki](https://gitlab.ethz.ch/physio/physio-doc)
- New Human Connectome Project reader for preprocessed Siemens 3-column logfiles (`*Physio_log.txt`)
- Updated Siemens Reader for Multiband patches(CMRR), versions EJA_1
    - including multi-echo data (4,5 columns)
    - multi-channel ECG data
    - significant speed up of read-in
    - generalized framework for later changes to format
    - interpolation of different sampling rates RESP/CARDIAC
- updated README about documentation, new support policy and [TAPAS on GitHub](https://translationalneuromodeling.github.io/tapas)
- extended FAQ

Minor Release Notes (R2017.2)
-----------------------------

- Included Markdown-based documentation via Wiki (also CITATION, LICENSE, CHANGELOG.md)
- Included FAQ in Wiki
- Split git repositories into public, dev, examples, and added wiki, to disentangle development from deployed toolbox code and data
- Bugfix and Typo correction
- Philips SCANPYHSLOG for their software release 5.1.7.

Minor Release Notes (R2017.1)
-----------------------------

- Substantially improved Siemens interface, both for VB/VD and 3T/7T releases
    - several bugfixes
    - based on extensive user feedback from Berlin and Brisbane
- New functionality tapas_physio_overlay_contrasts.m to display non-physio 
  contrasts automatically as well

Major Release Notes (r904 / R2016.1)
------------------------------------

- Software version for accepted PhysIO Toolbox Paper: doi:10.1016/j.jneumeth.2016.10.019
- Tested and expanded versions of examples
- Improved stability by bugfixes and compatibility to Matlab R2016
- Slice-wise regressor creation
- Detection of constant physiological time series (detachment, clipping)
- Refactoring of report_contrasts and compute_tsnr_gains as standalone functionality
- Improved Read-in capabilities (Siemens respiration data, BioPac .mat)
- Migration from svn (r904) to git (tnurepository) for version control

Major Release Notes (r835)
--------------------------

- Software version for Toolbox Paper submission
- Noise ROIs modeling
- Extended motion models (24 parameters, Volterra expansion)
- HRV/RVT models with optional multiple delay regressors
- Report_contrasts with automatic contrast generation for all regressor groups
- compute_tsnr_gains for individual physiological regressor groups
- consistent module naming (scan_timing, preproc)
- Visualisation improvement (color schemes, legends)

Minor Release Notes (r666)
--------------------------

- Compatibility tested for SPM12, small bugfixes Batch Dependencies
- Cleaner Batch Interface with grouped sub-menus (cfg_choice)
- new model: 'none' to just read out physiological raw data and preprocess,
  without noise modelling 
- Philips: Scan-timing via gradient log now automatized (gradient_log_auto)
- Siemens: Tics-Logfile read-in (proprietary, needs Siemens-agreement)
- All peak detections (cardiac/respiratory) now via auto_matched algorithm
- Adapt plots/saving for Matlab R2014b

Major Release Notes (r534)
--------------------------

- Read-in of Siemens plain text log files; new example dataset for Siemens
- Speed up and debugging of auto-detection method for noisy cardiac data => new method thresh.cardiac.initial_cpulse_select.method = ???auto_matched???
- Error handling for temporary breathing belt failures (Eduardo Aponte, TNU Zurich)
- slice-wise regressors can be created by setting sqpar.onset_slice to a index vector of slices

Major Release Notes (r497)
--------------------------

- SPM matlabbatch GUI implemented (Call via Batch -> SPM -> Tools -> TAPAS PhysIO Toolbox)
- improved, automatic heartbeat detection for noisy ECG now standard for ECG and Pulse oximetry (courtesy of Steffen Bollmann)
- QuickStart-Manual and PhysIO-Background presentation expanded/updated
- job .m/.mat-files created for all example datasets
- bugfixes cpulse-initial-select method-handling (auto/manual/load)

Major Release Notes (r429)
--------------------------

- Cardiac and Respiratory response function regressors integrated in workflow (heart rate and breathing volume computation)
- Handling of Cardiac and Respiratory Logfiles only
- expanded documentation (Quickstart.pdf and Handbook.pdf)
- read-in of custom log files, e.g. for BrainVoyager peripheral data
- more informative plots and commenting (especially in tapas_physio_new).

Minor Release Notes (r354)
--------------------------

- computation of heart and breathing rate in Philips/PPU/main_PPU.m
- prefix of functions with tapas_*

Major Release Notes (r241)
--------------------------

- complete modularization of reading/preprocessing/regressor creation for peripheral physiological data
- manual selection of missed heartbeats in ECG/pulse oximetry (courtesy of Jakob Heinzle)
- support for logfiles from GE scanners (courtesy of Steffen Bollmann, KiSpi Zuerich)
- improved detection of pulse oximetry peaks (courtesy of Steffen Bollmann)
- improved documentation
- consistent function names (prefixed by "physio_")

NOTE: Your main_ECG/PPU.m etc. scripts from previous versions (<=r159) will not work with this one any more. Please adapt one of the example scripts for your needs (~5 min of work). The main benefit of this version is a complete new variable structure that is more sustainable and makes the code more readable.

