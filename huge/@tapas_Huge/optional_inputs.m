function [ obj ] = optional_inputs( obj, varargin )
% Parse and check name-value pair arguments for constructor tapas_Huge and
% estimate method.
% 
% This is a protected method of the tapas_Huge class. It cannot be called
% from outside the class.
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


if nargin < 3
    return
end

defaultOptions = obj.default_options( );
obj.options.nvp.k = obj.K;
obj.options.nvp.tag = obj.tag;
nvp = tapas_huge_parse_inputs( obj.options.nvp, varargin );

%% check and process inputs
% switch confounds on/off
assert(any(strcmpi(nvp.confoundsvariant, {'default', 'none', 'global', ...
    'cluster'})), 'TAPAS:HUGE:Nvp:ConfoundsVariant',  ['Valid values for ' ...
    'ConfoundsVariant are: "default", "none", "global" or "cluster"']);

% BOLD data as list of DCMs in SPM format (will overwrite previous data and
% results)
if ~isempty(nvp.dcm) || ~isempty(nvp.confounds) || ...
        ~isempty(nvp.omitfromclustering)
    obj = obj.import(nvp.dcm, nvp.confounds, nvp.omitfromclustering);
    nvp.dcm = {};
    nvp.confounds = [];
    nvp.omitfromclustering = [];
end

  
% number of clusters
obj.K = nvp.k;

% inversion scheme
assert(any(strcmpi(nvp.method, {'vb'})), 'TAPAS:HUGE:Nvp:Method', ...
    'Valid values for Method are: VB.')
nvp.method = upper(nvp.method);

% number of iterations
val = nvp.numberofiterations;
if ~isempty(val)
    assert(isscalar(val) && isnumeric(val) && val > 0 && mod(val, 1) == 0,...
        'TAPAS:HUGE:Nvp:Iterations', ...
        'Number of iterations must be positive integer.');
end

% prior mean of cluster centers (mu_0)
val = nvp.priorclustermean;
if ischar(val) && strcmpi(val, 'default')
    nvp.priorclustermean = defaultOptions.nvp.priorclustermean;
else
    assert(isnumeric(val) && isvector(val), 'TAPAS:HUGE:Nvp:PriorMean', ...
        'PriorClusterMean must be array of double.')
end

% prior mean of cluster covariance (S_0)
val = nvp.priorclustervariance;
if ischar(val) && strcmpi(val, 'default')
    nvp.priorclustervariance = defaultOptions.nvp.priorclustervariance;
else
    [~,p] = chol(val);
    assert(issymmetric(val) && p == 0, 'TAPAS:HUGE:Nvp:PriorVariance', ...
        'PriorClusterVariance must be symmetric and positive definite.');
end

% prior degrees of freedom (nu_0)
val = nvp.priordegree;
if ischar(val) && strcmpi(val, 'default')
    nvp.priordegree = defaultOptions.nvp.priordegree;
else
    assert( isnumeric(val) && val(1) > 0, 'TAPAS:HUGE:Nvp:PriorDegree', ...
        'PriorDegree must be positive scalar.');
end

% prior ratio between cluster covariance and covariance of cluster mean
% (tau_0) 
nvp.priorvarianceratio;
if ischar(val) && strcmpi(val, 'default')
    nvp.priorvarianceratio = defaultOptions.nvp.priorvarianceratio;
else
    assert( isnumeric(val) && any(val(:) > 0), 'TAPAS:HUGE:Nvp:PriorTau', ...
        'PriorVarianceRatio must be a positive scalar.');
end

% path and filename for saving results
if ~isempty(nvp.saveto)
    pathStr = fileparts(nvp.saveto);
    assert(exist(pathStr, 'dir') == 7, 'TAPAS:HUGE:Nvp:DirName', ...
        'The directory "%s" does not exist.', pathStr);
end

% set random number generator seed
if ~isempty(nvp.seed)
    if iscell(nvp.seed)
        rng(nvp.seed{:});
    else
        rng(nvp.seed);
    end
end

% starting values for ...
% ... subject parameter means: prior (default), spm, numeric array
val = nvp.startingvaluedcm;
assert(isnumeric(val) || (ischar(val) && any(strcmpi(val, {'prior', 'spm'}))), ...
    'TAPAS:HUGE:Nvp:StartingValue', ['Valid values for StartingValueDcm are: ' ...
    '"prior", "spm" or numerical array.']);
% ... cluster parameter means: prior (default), numeric array
val = nvp.startingvaluegmm;
assert(isnumeric(val) || (ischar(val) && any(strcmpi(val, {'prior'}))), ...
    'TAPAS:HUGE:Nvp:StartingValue', ['Valid values for StartingValueGmm are: ' ...
    '"prior" or numerical array.']);

% model description
assert(ischar(nvp.tag), 'TAPAS:HUGE:Nvp:Tag', ...
    'Tag must be a character array.');
obj.tag = nvp.tag;
            
% save inputs
obj.options.nvp = nvp;


