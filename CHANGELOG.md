# Changelog
TAPAS toolbox 

## [6.0.1] 2022-05-17

### Fixed
- HGF: [Version 7.1](HGF/README.md#release-notes): Fixed bug in `tapas_simModel`.

## [6.0.0] 2022-04-08
For this major release of TAPAS, we have decided to move currently unmaintained toolboxes
to a dedicated [TAPAS Legacy repository](https://github.com/translationalneuromodeling/tapas_legacy).

### Added
- PhysIO [Version 8.1.0](Physio/CHANGELOG.md): BIDS Read-in for separate cardiac/respiratory trace files (e.g., due to different sampling frequencies)
    - Compatibility of whole code base with Matlab compiler in order to run `spm_make_standalone`.
### Changed 
- Moved `MICP`,`VBLM`,`h2gf` and `mpdcm` toolboxes to TAPAS Legacy repository.
- HGF [Version 7.0](HGF/README.md#release-notes): Numerical stability improvements, various enhancements
- rDCM [Version 1.4](rDCM/CHANGELOG.md):  The update includes a new function that can convert the rDCM output to the SPM format. Additionally, some small bug fixes and changes have been made. No changes to the main functionality of rDCM have been made.

### Fixed


## [5.1.2] 2021-09-06

### Fixed
- TOOLS: Fixed a bug in the computation of WAIC (function [tapas_waic.m](tools/tapas_waic.m)). Added new function to compute log-exp_sum (function [tapas_logsumexp.m](tools/tapas_logsumexp.m)).

## [5.1.1] 2021-06-22

### Fixed
- Added references to TAPAS paper (in collection and toolboxes)
- HGF: [Version 6.3](HGF/README.md#release-notes): Minor improvements
- rDCM: [Version v1.3](rDCM/CHANGELOG.md): Corrected bug in construction of the full posterior covariance matrix (from the region-wise posterior covariance matrices)

## [5.1.0] 2021-04-23

### Added
- HUGE: Added Monte Carlo based inference (https://doi.org/10.1002/hbm.25431)

### Fixed
- Fixed typos


## [5.0.0] 2021-03-15

### Added
- [genbed](genbed/README.md): Python package for Generative Embedding
- [UniQC](UniQC/README.md): unified neuroimaging quality control (beta release)
- [task/BLT](task/BLT/README.md): breathing learning task 
- [PhysIO](PhysIO/CHANGELOG.md): Release 2021a-v8.0; new method for estimating respiratory volume per unit time (RVT) via the Hilbert transform ([Harrison et al., NeuroImage 230 (2021)](https://doi.org/10.1016/j.neuroimage.2021.117787))
    - also: robust respiratory preprocessing: optional de-spiking (median filter) and window-padded detrending

## [4.0.0] 2020-09-09

### Added
- [ceode](ceode/README.md): Toolbox to integrate delay differential equations (DDEs) underlying convolution based Dynamic Causal Models for ERPs. 
- task/FDT: Filter Detection Task [version 0.2.2](task/FDT/README.md)
- [citation information](CITATION.cff)

### Changed
- rDCM: [version v1.2](rDCM/CHANGELOG.md)
- HUGE: removed deprecated functions
- external toolboxes are now managed as matlab packages

### Fixed
- Fixed name collision issue due to mcmcdiag toolbox (see [issue 106](https://github.com/translationalneuromodeling/tapas/issues/106))

### Known issues
- h2gf (beta): demo script currently does not work


## [3.3.0] 2020-07-17

### Added
- h2gf: Hierarchical inference for the HGF, beta version released
- SERIA: Added the Watanabe Akaike information criterion (WAIC) as output.
- SERIA: Added computation of early and late response likelihood.
- SERIA: Added delta plots to output summary.

### Changed
- HGF: v6.0 released
- SERIA: The constraints matrix is now automatically plot by 
        tapas_sem_display_posterior.m
- SERIA: Python code is now updated to python 3.7 and the requirement file
            now uses the latest libraries (numpy, scipy, cython).
- SERIA: Posterior plots includes fits of the complete group.
- SERIA: Summaries are automatically generated when preparing the posterior.
- PhysIO: PhysIO Toolbox Release R2020a, v7.3.0
- PhysIO: removed Matlab statistics toolbox dependency for PCA by SVD implementation (thanks to Benoît Beranger, [pull request 64](https://github.com/translationalneuromodeling/tapas/pull/64))

### Fixed
- HUGE: fixed tick labels

### Known issues
- TAPAS uses the mcmcdiag toolbox for Monte Carlo convergence diagnostics, which shadows a number of matlab built-in functions. Affected functions include: Contents.m, cusum.m, join.m and score.m. Workaround: If you use one of these functions in your own code, remove the tapas folder from your matlab path when you are not using the TAPAS toolbox.

## [3.2.0] 2019-09-29

### Added
- HUGE: introduced object-oriented interface in addition to old interface
- HUGE: build-in unit tests
- HUGE: user manual
- PhysIO (details in tapas/PhysIO/CHANGELOG.md)
    - more unit testing and integration testing for examples
    - bandpass-filtering for cardiac data in preprocessing, user-defined max
    heart rate for peak detection

### Fixed
- PhysIO: Bugfix RVT/HRV convolution had erroneous half-width shift
    - For details on the bug, its impact and fix, see our specific [Release
    Info on RVT/HRV Bugfix](https://github.com/translationalneuromodeling/tapas/issues/65)

### Changed
- HUGE: demo script reflect interface changes


## [3.1.0] 2019-03-26

### Added
- Get revision info from Matlab.
- PhysIO R2018.1.2: BioPac txt-file reader (for single file, resp/cardiac/trigger data in different columns)
- SERIA: Automatic plotting of the seria model.
- SERIA: Example for single subject.

### Fixed 
- Huge: minor bugs.

### Changed
- Huge: Improved documentation.
- New version of the HGF toolbox (v5.3). Details in tapas/HGF/README.md
- New version of the rDCM toolbox (v1.1). Details in tapas/rDCM/CHANGELOG.md.
- New version of the PhysIO Toolbox (R2019a-v7.1.0)
    - BIDS and BioPac readers; code sorted in modules (`readin`, `preproc` etc.), 
      also reflected in figure names
    - Updated and extended all examples, and introduced unit testing
    - Full details in tapas/PhysIO/CHANGELOG.md
- Improved the documentation of SERIA.

## [3.0.1] 2018-10-17

### Fixed
- PhysIO R2018.1.2: fixed bug for 3D nifti array read-in in tapas_physio_create_noise_rois_regressors (github issue #24, gitlab #52)

### Added

## [3.0.0] 2018-09-09

### Added
- tapas_get_tapas_revision.m outputs the branch and hash of the repository.
- Revision is printed when initiliazing tapas.
- Contributor License Agreement (CLA) file
- CONTRIBUTING.md explaining coding and style guidelines, signing procedure for CLA file
- Include the function tapas_get_current_version.m.
- Implements download of example data from the server using 
    tapas_download_example_data.
- Now there is log file that list the versions of tapas, the download link
    and the hash of the file that is downloaded.
- Use the an external file to do the md5 check sum. See external javamd5.
- HUGE toolbox: hierarchical unsupervised generative embedding  
    https://doi.org/10.1016/j.neuroimage.2018.06.073
- rDCM toolbox: Regression dynamic causal modeling   
    https://doi.org/10.1016/j.neuroimage.2017.02.090  
    https://doi.org/10.1016/j.neuroimage.2018.05.058

### Changed
- README.md to include reference to CONTRIBUTING.md and explanation of CLA
- Dropped 4 digits versioning for 3.
- The version of tapas is now read from misc/log_tapas.txt. It is the first
    line of this file.
- Updated the documentation of SEM.
- Updated SEM to include hierarchical models for inference.
- New version of the HGF toolbox (5.2.0). Details in tapas/HGF/README.md.
- New version of the PhysIO toolbox (R2018.1). Details in tapas/PhysIO/CHANGELOG.md.
- tapas_init.m displays a message to download the data in case the examples
    directory is not present.
- Update MPDCM for supporting up to cuda 9.0 and openmp.

### Removed
- Deleted the gpo folder.

## [2.7.4.1] 2018-01-24

### Added
- includes PhysIO R2017.3 
    - new Readers: Human Connectome Project (HCP) and Siemens multiband (CMRR) WIP Physlog files
    - extended documentation and FAQ via Wiki

## [2.7.3.2] 2018-01-19

### Changed
- README.md includes references to github, new support policy, links to toolbox documentation

### Fixed
- tapas_version typo
- tapas_print_logo with T=Translational instead of TNU

## [2.7.3.1] 2017-08-18

### Changed
- CHANGELOG.md includes value for hot fixes.

### Fixed
- Repair links to the wiki in the readme file.
- Typos in SEM documentation.


## [2.7.3.0] 2017-08-17

### Added
- Added condhalluc_obs and condhalluc_obs2 models.

### Changed
- Introduced kappa1 in all binary HGF models.


## [2.7.2.0] 2017-08-17

### Added
- New PhysIO CHANGELOG.md specific file.

### Changed
- Specified in PhysIO/CHANGELOG.md.


## [2.7.1.0] 2017-08-17

### Added
- Added a CHANGELOG.md file

### Changed
- Now the README.txt file is in markdown format.
- The documentation is integrated with the wiki in github.
