%% List of name-value pair arguments accepted by tapas_Huge() and estimate()
% 
% 
% NAME:        Confounds
% VALUE:       double array
% DESCRIPTION: Specify confounds for group-level analysis (e.g. age or sex)
%              as double array with one row per subject and one column per
%              confound. Note: This property can only be used in
%              combination with the Dcm property.
%              WARNING: This feature is experimental.
% 
% NAME:        ConfoundsVariant
% VALUE:       'none' | 'global' | 'cluster' (default: 'global' if
%              confounds specified, 'none' otherwise)
% DESCRIPTION: Determines how confounds enter model. 'none': Confounds are
%              not used. 'global': Confounds enter global regression 
%              (variant 1). 'cluster': Confounds enter cluster-specific 
%              regression (variant 2).
% 
% NAME:        Dcm
% VALUE:       cell array of DCM structs in SPM format
% DESCRIPTION: Specify DCM structure and BOLD time series for all subjects
%              as cell array with one DCM struct in SPM format per subject.
% 
% NAME:        K
% VALUE:       positive integer (default: 1)
% DESCRIPTION: Number of clusters.
% 
% NAME:        Method  
% VALUE:       'VB'
% DESCRIPTION: Name of inversion method specified as character array. VB:
%              variational Bayes.
% 
% NAME:        NumberOfIterations
% VALUE:       positive integer (default: 999 for VB, 2e5 for MH)
% DESCRIPTION: For VB: maximum number of iterations. For Monte Carlo
%              methods: Length of Monte Carlo chain in samples.
% 
% NAME:        OmitFromClustering
% VALUE:       array of logical | struct with fields a, b, c and d
%              (default: empty struct)
% DESCRIPTION: Select DCM parameters to exclude from clustering. Parameters
%              excluded from clustering will still be estimated, but under
%              a static Gaussian prior. If input is array, it will be
%              treated as the a field in a struct. Missing fields will be
%              treated the same as arrays of false. Note: This property can
%              only be used in combination with the Dcm property.
% 
% NAME:        PriorClusterMean
% VALUE:       'default' | row vector of double
% DESCRIPTION: Prior cluster mean. Scalar input will be expanded into
%              vector. Default: [0, ... ,0]. 
% 
% NAME:        PriorClusterVariance
% VALUE:       'default' | symmetric, positive definite matrix of double
% DESCRIPTION: Prior mean of cluster covariances. Must be a symmetric, 
%              positive definite matrix. Scalar input will be expanded into
%              diagonal matrix. Default: diag(..., 0.01, ...).
% 
% NAME:        PriorDegree
% VALUE:       'default' | positive double
% DESCRIPTION: nu_0 determines the prior precision of the cluster
%              covariance. For VB, this is the degrees of freedom of the
%              inverse-Wishart. For MH, this is the prior precision of the
%              cluster log-precision. Default: 100.
% 
% NAME:        PriorVarianceRatio
% VALUE:       'default' | positive double
% DESCRIPTION: Ratio tau_0 between prior mean cluster covariance and prior
%              covariance over cluster mean. Prior covariance over cluster
%              mean equals prior cluster covariance divided tau_0. Default:
%              0.1.
% 
% NAME:        Randomize
% VALUE:       bool (default: false)
% DESCRIPTION: If true, starting values for subject level DCM parameter 
%              estimates are randomized.
% 
% NAME:        SaveTo
% VALUE:       character array
% DESCRIPTION: Location for saving results consisting of path name and
%              optional file name. Path name must end on file separator and
%              point to an existing directory. If file name is not
%              specified, it is set to 'huge' followed by date and time.
% 
% NAME:        Seed
% VALUE:       double | cell array of double and rng name | random number 
%              generator seed obtained with rng() command
% DESCRIPTION: Seed for random number generator.
% 
% NAME:        StartingValueDcm
% VALUE:       'prior' | 'spm' | double array (default: 'prior')
% DESCRIPTION: Starting values for subject-level DCM parameter estimates. 
%              'prior' uses prior cluster mean for all subjects. 'spm' uses 
%              values supplied in the 'Ep' field of the SPM DCM structs.
%              Use a double array with number of rows equal to number of
%              subjects to specify custom starting values.
% 
% NAME:        StartingValueGmm
% VALUE:       'prior' | double array (default: 'prior')
% DESCRIPTION: Starting values for cluster-level DCM parameter estimates. 
%              'prior' uses prior cluster mean for all clusters. Use a
%              double array with number of rows equal to number of clusters
%              to specify custom starting values.
% 
% NAME:        Tag
% VALUE:       character array
% DESCRIPTION: Model description
% 
% NAME:        TransformInput
% VALUE:       bool (default: false)
% DESCRIPTION: Transform input strength (C matrix) to log-domain.
% 
% NAME:        Verbose
% VALUE:       bool (default: false)
% DESCRIPTION: Activate/deactivate command line output.
% 
% 

% Author: Yu Yao (yao@biomed.ee.ethz.ch)
% Copyright (C) 2019 Translational Neuromodeling Unit
%                    Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
% 
% This file is part of TAPAS, which is released under the terms of the GNU
% General Public Licence (GPL), version 3. For further details, see
% <https://www.gnu.org/licenses/>.
% 
% This software is provided "as is", without warranty of any kind, express
% or implied, including, but not limited to the warranties of
% merchantability, fitness for a particular purpose and non-infringement.
% 
% This software is intended for research only. Do not use for clinical
% purpose. Please note that this toolbox is under active development.
% Considerable changes may occur in future releases. For support please
% refer to:
% https://github.com/translationalneuromodeling/tapas/issues
% 