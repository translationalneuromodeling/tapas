RELEASE INFORMATION
===================

Current Release
---------------

*Current version: PhysIO Toolbox Release R2024a, v9.0.0*

March 25th, 2025


Bugfix Release Notes (v9.0.1)
----------------------------

### Fixed
- Discrepancy Linux/Windows in detected cardiac pulses (`ons_secs.raw.cpulse`) for near-constant traces (e.g., in `tapas_physio_example_siemens_vb_ppu3t_sync_first_spm_job`)
due to `tapas_physio_corrcoef2` for 
- Inconsistent Handling of `bids_dir`; now consistent with `save_dir` etc.: as cell in SPM job, but char in `physio` struct

Major Release Notes (v9.0.0)
----------------------------

### Added
- BIDS writer: write out BIDS-compatbile physiological logfiles (`.tsv.gz` and `.json`)
  from any vendor format
- Greatly expanded Review visualization (`tapas_physio_review`): Allow more detailed re-creation of figures from online execution, control visual verbosity retrospectively (e.g., for debugging)
- Read-In of field `AcquisitionTime` from BIDS `.json` side-car file for converted Siemens DICOMs after [dcm2niix](https://www.nitrc.org/plugins/mwiki/index.php/dcm2nii:MainPage) conversion for synchronization (Gitlab issue 109)

### Changed
- Expanded usage of scan trigger trace (continuous, binary, on/off vs alternating levels)

### Fixed
- Bugfix processing of short Siemens IdeaCmdTool logfiles (stop before scan ending)
- DICOM Read-in for files without extension

Minor Release Notes (v8.2.0)
----------------------------

### Added
- Interface `tapas_physio_test` to TAPAS-generic `tapas_test` function
- Added suport for logfile version 3 of Siemens physio recordings 
    - multi ECG/Resp channels and interleaved status messages
    - new integration test for Siemens VB Logversion 3
- Added support for ADInstruments/LabChart Txt-export format (see 
  [CUBRIC Seminar Example](https://github.com/BRAIN-TO/cubric-physio) and
  gitlab branch #107)

### Fixed
- Removed dependence on `nanmean` (Statistics Toolbox)
    - See [GitHub issue #205](https://github.com/translationalneuromodeling/tapas/issues/205) 
- Compatibility with multiple SPM toolbox locations for `lmod` ([GitHub issue #211](https://github.com/translationalneuromodeling/tapas/issues/211))
    - as listed in `spm_get_defaults('tbx')`
- Refactoring of Philips read-in to support novel 12-column logfile version, see [GitHub issue #207](https://github.com/translationalneuromodeling/tapas/issues/207#issuecomment-1246078600)
- Unit/Integration tests for filtered traces (cardiac and respiratory) switched to absolute tolerances (relative problematic for traces close to zero)

Minor Release Notes (v8.1.0)
----------------------------

### Added
- Compatibility of whole code base with Matlab compiler in order to run `spm_make_standalone`
    - provides oppurtunity to run SPM Batch Editor GUI version of PhysIO without Matlab license requirement 
    - compiled version readily available within [Neurodesk](https://neurodesk.github.io/tutorials/functional_imaging/physio/) 
- BIDS Read-in for separate cardiac/respiratory trace files (e.g., due to different sampling frequencies)
    - see GitHub Issue #164 and pull request #167 by @alexsayal
    - Additional unit tests for new read-in and example data

### Changed
- Switch for certain toolbox functions (e.g., `imtool`) to only run in 
  non-compiled code

### Fixed
- Documentation (function headers, see Github issue #149)
- Typos in unused function (spotted in compilation)
- Synchronization SIEMENS AcquisitionLog / Physiological files (see Github issue #172)
    - better visualization of sync, clearer error messages if dummy scans not found

Bugfix Release Notes (v8.0.1)
-----------------------------

### Changed
- Citation of novel [TAPAS paper](https://doi.org/10.3389/fpsyt.2021.680811) in README and CITATION


Major Release Notes (v8.0.0)
----------------------------

### Added
- New method for computing respiratory volume per unit time (RVT) via the
  Hilbert transform.
    - Publication: Harrison et al., "A Hilbert-based method for processing
      respiratory timeseries", NeuroImage, 2021.
      <https://doi.org/10.1016/j.neuroimage.2021.117787>
    - This is now the default option, but the old method is available by
      setting `physio.model.rvt.method = 'peaks'` (or the equivalent within
      the SPM batch editor).

- Respiratory preprocessing now includes an optional de-spiking step based on
  median filtering.

### Changed
- Now possible to change the frequencies of the respiratory filtering during
  preprocessing via `physio.preproc.respiratory.filter` (or the equivalent
  within the SPM batch editor).

- More robust detrending of raw respiratory timeseries via windowed padding
  before filtering.

### Fixed


Bugfix Release Notes (v7.3.2)
-----------------------------

### Added
- version number `physio.version` in physio-struct (Gitlab issue #101)

### Fixed
- Subfolder of SPM in path (fieldtrip) created ambiguous function calls 
  to overloaded functions (Gitlab issue #102 and Gihub issue #110)
    - e.g., `filtfilt` in `tapas_physio_filter_respiratory` created flat 
      regressors for v7.3 for Siemens VD example dataset
    - now subfolders of SPM will be removed when calling
     `tapas_physio_main_create_regressors`


Bugfix Release Notes (v7.3.1)
-----------------------------

### Fixed
- PPU read-in works for BioPac mat-file now 
    - correct column labels and cardiac modality (Github Issue #103)
    - thank you to Manon Durand-Ruel @mdurandruel for reporting  


Minor Release Notes (R2020a, v7.3.0)
------------------------------------

### Added
- Added descriptive names for the multiple regressors matrix.
    - Closes GitLab issue #82.
    - Now possible to straightforwardly inspect `physio.model.R` and the
      contents of `physio.model.output_multiple_regressors` using
      `physio.model.R_column_names`.
    - Added `tapas_physio_guess_regressor_names()` to maintain backwards
      compatibility.
- New example datasets Siemens VB PPU3T with DICOM Sync (courtesy of Alexander Ritter, Jena, Germany)
- More versatile control on figure visibility, saving and closing during `main` and `review` runs of PhysIO
    - feature provided by Stephan Heunis, TU Eindhoven, The Netherlands (github issue #89)
    - figures can now be plotted in the background without disturbing interactive Matlab sessions, and can be (more) selectively saved and closed during execution of `tapas_physio_review`
    - more comprehensive support within `tapas_physio_main_create_regressors` to follow


Bugfix Release Notes (v7.2.8)
-----------------------------

### Fixed
- Bug(s) when checking SPM and PhysIO paths in `tapas_physio_init` under
  certain particular Matlab path environment settings (Gitlab merge request !37)
    - e.g., when add `physio-public/code` manually without subfolder
    - or if spm existed as a folder name without being added to path


Bugfix Release Notes (v7.2.7)
-----------------------------

### Changed
- Reimplemented `tapas_physio_filter_respiratory.m`.
    - Closes GitLab issue #98.
    - Reduces the cut-off frequency of the high-pass filter, so as to behave
      more like a detrending step. Previous value of 0.1 Hz could distort e.g.
      sigh breaths, which can easily last upwards of 10 s.
    - Reimplements the filters themselves, with a higher filter order, extra
      padding, and a two step high-pass and low-pass procedure (i.e. rather
      than band pass) to improve the overall stability.


Bugfix Release Notes (v7.2.6)
-----------------------------

### Fixed
- Meaningful error message for `auto_matched` peak detection, if time series is too short
    - at least 20 peaks (and pulse templates) are required to create a pulse template
    - this is now stated explicitly, if time series is too short
    - reported by Joy Schuurmans Stekhoven, TNU, as gitlab issue #92


Bugfix Release Notes (v7.2.5)
-----------------------------

### Fixed
- Corrected documentation for ` preproc.cardiac.initial_cpulse_select.min` parameter
    - threshold for peak relative to z-scored time series, *not* correlation
    - reported by Sam Harrison, TNU, as gitlab issue #95


Bugfix Release Notes (v7.2.4)
-----------------------------

### Fixed
- Stop docked figure default throwing error with `-nodisplay`
    - allows generating saved figure without a display, e.g., on remote server
    - bugfix provided by Sam Harrison, TNU


Bugfix Release Notes (v7.2.3)
-----------------------------

### Fixed
- Bugfix manual peak selection (Github issue #85, Gitlab #90)
    - did not work because of figure handling


Bugfix Release Notes (v7.2.2)
-----------------------------

### Fixed
- Bugfix Siemens VB (`*.resp, *.puls, *.ecg`)
    - Synchronization to DICOM time stamp did not work for extended physiological recordings (not starting/ending with functional run) due to ignored absolute start time stamp
    - reported by Alexander Ritter, Jena, Germany (Github issue #55, Gitlab #87)
    - probably fixes Github issue #63 (Gitlab #86) as well


Bugfix Release Notes (v7.2.1)
-----------------------------

### Changed
- PhysIO: removed Matlab statistics toolbox dependency for PCA by SVD implementation (thanks to Benoît Beranger, [pull request 64](https://github.com/translationalneuromodeling/tapas/pull/64))
    - new function `tapas_physio_pca` allows for switch between stats and native SVD implementation of PCA
    - comes with unit tests checking equivalency


Minor Release Notes (R2019b, v7.2.0)
------------------------------------

### Added
- `requirements.txt` making dependencies on Matlab and specific toolboxes 
  explicit
- `max_heart_rate_bpm` now a user parameter to adjust prior on max allowed 
  heart rate for cardiac pulse detection (`method = 'auto_matched'`)
- bandpass-filtering of cardiac data during preprocessing now possible 
  (`preproc.cardiac.filter`)
- Added integration tests for all examples in `tests/integration` for both SPM
Batch Editor and Matlab-only configuration scripts. Reference data provided in
`examples/TestReferenceResults/examples`

### Changed
- Toned down and replaced irrelevant peak height and missing cardiac pulse 
  warnings (github issue #51)
- Updated README to include external contributors and recent findings about
  impact of physiological noise for serial correlations (Bollmann2018)
- Added unit tests for convolution and moved all to `tests/unit`

### Fixed
- Corrected half-width shift of response functions for HRV and RVT regressors by
  erroneous use of Matlab `conv` 
    - For details on the bug, its impact and fix, see our specific [Release
    Info on RVT/HRV Bugfix](https://github.com/translationalneuromodeling/tapas/issues/65)
    - other references: TNU gitlab issue #83, found and fixed by Sam Harrison,
    TNU, see `tapas_physio_conv`)
- Bugfix `tapas_physio_init()` not working, because dependent on functions 
  in `utils` subfolder not in path; `utils` added to path
- `tapas_physio_review` for motion parameters (found and fixed by 
  Sam Harrison, TNU)
- visualization error for regressor orthogonalization (github issue #57), 
  when only `'RETROICOR'` set was chosen


Minor Release Notes (R2019a, v7.1.0)
------------------------------------

### Added
- Brain Imaging Data Structure ([BIDS](https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/06-physiological-and-other-continuous-recordings.html)) reader and example for `*_physio.tsv[.gz]/.json` files
- Added BioPac txt-File read-in and example
- Template example with all physio-fields for matlab script and settings as in default SPM batch
- Started unit testing framework in folder `tests`
    - example functions for findpeaks and BIDS readin
    - reference data saved with example data in subfolder `TestReferenceResults`
    - reference data reflects physio structure after running example scripts
    with PhysIO R2019a

### Changed
- put all functions in `code` into subfolders relating to different modules:
  `readin`, `sync`, `preproc`, `model`, `assess`, `utils` (gitlab-issue #58)
    - updated deployment `tapas_physio_init` because of that
    - updated figure names to reflect respective code module
- matlab-script examples now contain more comments
    - fixed internal bug that prepended absolute paths to input logfiles in automatic example generation
- `tapas_physio_create_noise_rois_regressors` with more flexible ROI reslicing 
  options (speed-up) and uses `spm_erode` (no Matlab image processing toolbox needed),
  thanks to a [contribution by Benoît Béranger](https://github.com/translationalneuromodeling/tapas/pull/34)
- introduced semantic version numbers for all previous releases, and changed
Release numbering to R<YEAR><LETTER> style
- extended documentation (FAQ, new read-in BIDS)
- several bugfixes (Sep 18 - Mar 19), see 
  [GitHub Issues](https://github.com/translationalneuromodeling/tapas/issues?q=label:physio)

### Removed
- `tapas_physio_findpeaks` now refers to current Matlab signal processing
toolbox implementation, instead of copy of older version
- some Matlab toolbox dependencies by custom simplified functions (e.g.,
`suptitle`)


Bugfix Release Notes (R2018.1.3, v7.0.3)
----------------------------------------

### Changed
- fixed bug for matching of Philips SCANPHYSLOG-files (Gitlab #62), if 
  physlogs were acquired on different days, but similar times


Bugfix Release Notes (R2018.1.2, v7.0.2)
----------------------------------------

### Added
- BioPac txt-file reader (for single file, resp/cardiac/trigger data in different columns)

### Changed
- fixed bug for 3D nifti array read-in in tapas_physio_create_noise_rois_regressors (github issue #24, gitlab #52)


Bugfix Release Notes (R2018.1.1, v7.0.1)
----------------------------------------

### Changed
- documentation.{html,pdf} export nicer with different FAQ numbering


Major Release Notes (R2018.1, v7.0.0)
-------------------------------------

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


Minor Release Notes (R2017.3, v6.3.0)
-------------------------------------

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


Minor Release Notes (R2017.2, v6.2.0)
-------------------------------------

- Included Markdown-based documentation via Wiki (also CITATION, LICENSE, CHANGELOG.md)
- Included FAQ in Wiki
- Split git repositories into public, dev, examples, and added wiki, to disentangle development from deployed toolbox code and data
- Bugfix and Typo correction
- Philips SCANPYHSLOG for their software release 5.1.7.


Minor Release Notes (R2017.1, v6.1.0)
-------------------------------------

- Substantially improved Siemens interface, both for VB/VD and 3T/7T releases
    - several bugfixes
    - based on extensive user feedback from Berlin and Brisbane
- New functionality tapas_physio_overlay_contrasts.m to display non-physio 
  contrasts automatically as well


Major Release Notes (r904 / R2016.1, v6.0.0)
--------------------------------------------

- Software version for accepted PhysIO Toolbox Paper: doi:10.1016/j.jneumeth.2016.10.019
- Tested and expanded versions of examples
- Improved stability by bugfixes and compatibility to Matlab R2016
- Slice-wise regressor creation
- Detection of constant physiological time series (detachment, clipping)
- Refactoring of report_contrasts and compute_tsnr_gains as standalone functionality
- Improved Read-in capabilities (Siemens respiration data, BioPac .mat)
- Migration from svn (r904) to git (tnurepository) for version control


Major Release Notes (r835, v5.0.0)
----------------------------------

- Software version for Toolbox Paper submission
- Noise ROIs modeling
- Extended motion models (24 parameters, Volterra expansion)
- HRV/RVT models with optional multiple delay regressors
- Report_contrasts with automatic contrast generation for all regressor groups
- compute_tsnr_gains for individual physiological regressor groups
- consistent module naming (scan_timing, preproc)
- Visualisation improvement (color schemes, legends)


Minor Release Notes (r666, v4.1.0)
----------------------------------

- Compatibility tested for SPM12, small bugfixes Batch Dependencies
- Cleaner Batch Interface with grouped sub-menus (cfg_choice)
- new model: 'none' to just read out physiological raw data and preprocess,
  without noise modelling 
- Philips: Scan-timing via gradient log now automatized (gradient_log_auto)
- Siemens: Tics-Logfile read-in (proprietary, needs Siemens-agreement)
- All peak detections (cardiac/respiratory) now via auto_matched algorithm
- Adapt plots/saving for Matlab R2014b


Major Release Notes (r534, v4.0.0)
----------------------------------

- Read-in of Siemens plain text log files; new example dataset for Siemens
- Speed up and debugging of auto-detection method for noisy cardiac data => new
method thresh.cardiac.initial_cpulse_select.method = 'auto_matched'
- Error handling for temporary breathing belt failures (Eduardo Aponte, TNU Zurich)
- slice-wise regressors can be created by setting sqpar.onset_slice to a index vector of slices


Major Release Notes (r497, v3.0.0)
----------------------------------

- SPM matlabbatch GUI implemented (Call via Batch -> SPM -> Tools -> TAPAS PhysIO Toolbox)
- improved, automatic heartbeat detection for noisy ECG now standard for ECG and Pulse oximetry (courtesy of Steffen Bollmann)
- QuickStart-Manual and PhysIO-Background presentation expanded/updated
- job .m/.mat-files created for all example datasets
- bugfixes cpulse-initial-select method-handling (auto/manual/load)


Major Release Notes (r429, v2.0.0)
----------------------------------

- Cardiac and Respiratory response function regressors integrated in workflow (heart rate and breathing volume computation)
- Handling of Cardiac and Respiratory Logfiles only
- expanded documentation (Quickstart.pdf and Handbook.pdf)
- read-in of custom log files, e.g. for BrainVoyager peripheral data
- more informative plots and commenting (especially in tapas_physio_new).


Minor Release Notes (r354, v1.1.0)
----------------------------------

- computation of heart and breathing rate in Philips/PPU/main_PPU.m
- prefix of functions with tapas_*


Major Release Notes (r241, v1.0.0)
----------------------------------

- complete modularization of reading/preprocessing/regressor creation for peripheral physiological data
- manual selection of missed heartbeats in ECG/pulse oximetry (courtesy of Jakob Heinzle)
- support for logfiles from GE scanners (courtesy of Steffen Bollmann, KiSpi Zuerich)
- improved detection of pulse oximetry peaks (courtesy of Steffen Bollmann)
- improved documentation
- consistent function names (prefixed by "physio_")

NOTE: Your main_ECG/PPU.m etc. scripts from previous versions (<=r159) will not work with this one any more. Please adapt one of the example scripts for your needs (~5 min of work). The main benefit of this version is a complete new variable structure that is more sustainable and makes the code more readable.
