# Changelog
Regression Dynamic Causal Modeling (rDCM) toolbox 


## [1.4] 2022-03-22

### Added
- Helper function that converts the model inversion output from the rDCM format to the SPM format. This will allow users to use the SPM machinery to perform additional analyses on their rDCM 
results (e.g., Bayesian model selection, Bayesian model averaging, Parametric Empirical Bayes). NOTE: Not all functionality in SPM is compatible with the rDCM conversion.

### Changed
- Improved the visualization function tapas_rdcm_visualize.m
- Corrected bug in construction of the full posterior covariance matrix (from the region-wise posterior covariance matrices)

### Removed
- Removed the misleading "empty" predicted BOLD signal time courses for resting-state models (only PSD are relevant) 


## [1.3] 2021-06-10

### Added
none

### Changed
- Adapted "Cite Me" information to include TAPAS paper and update existing rDCM papers
- Corrected typo in the comments of tapas_rdcm_estimate.m
- Corrected bug in construction of the full posterior covariance matrix (from the region-wise posterior covariance matrices) 

### Removed
none


## [1.2] 2020-09-01

### Added
- Facilitation of using rDCM for resting-state fMRI data. This includes changes to the MATLAB functions as well as information added to the [Manual](docs/Manual.pdf) of the toolbox. 
- Possibility to include multiple confounds rather than a simple constant for baseline shifts.
- Helper function for specification of whole-brain dynamic causal models (i.e., DCM structure).

### Changed
- Corrected small bug in the specification of regressors

### Removed
- none


## [1.1] 2019-03-03

### Added
- rDCM now stores the actual (measured) and predicted derivatives of the signal (in frequency domain). This is
relevant because it represents the data feature that is actually fitted. Additionally, a routine has been added to
tapas_rdcm_visualize.m to plot these fits in terms of the power spectral density in frequency domain.
- Option to inform the sparsity constraints (p0) of rDCM using external binary information (e.g., from anatatomical
or functional connectivity) 

### Changed
- Small changes to the console output of rDCM
- rDCM stores experimental inputs and the run-time in the output structure
- rDCM checks for empty input regressors and sets them to zero (if necessary) 
- Corrected small bug in the evaluation of the log-determinant
- Corrected small bug in the storage of the posterior covariance matrix

### Removed
- none


## [1.0] 2018-09-03

### Added
- Original release of the regression dynamic causal modeling (rDCM) toolbox (v1.0) 
as part of the open-source software package TAPAS [3.0].
- MATLAB functions implementing the necessary steps of an rDCM analysis have been added
- A brief tutorial tapas_rdcm_tutorial.m that demonstrates how to implement an rDCM analysis has been added
- Documentation has been added in the Manual.pdf, ReadMe.md, and throughout the code 

### Changed
- none

### Removed
- none
