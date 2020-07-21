# Changelog
Regression Dynamic Causal Modeling (rDCM) toolbox 


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
