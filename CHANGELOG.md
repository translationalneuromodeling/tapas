# Changelog
TAPAS toolbox 

## [3.0.0] 2018-09-09

### Added
- tapas\_get\_tapas\_revision.m outputs the branch and hash of the repository.
- Revision is printed when initiliazing tapas.
- Contributor License Agreement (CLA) file
- CONTRIBUTING.md explaining coding and style guidelines, signing procedure for CLA file
- Include the function tapas\_get\_current\_version.m.
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
- The version of tapas is now read from misc/log\_tapas.txt. It is the first
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
