function [ obj ] = estimate( obj, varargin )
% Estimate parameters of the HUGE model.
% 
% INPUTS:
%   obj - A tapas_Huge object containing fMRI time series.
% 
% OPTIONAL INPUTS:
%   This function accepts optional name-value pair arguments. For a list of
%   valid name-value pairs, see the user manual or type 'help
%   tapas_huge_property_names'.
% 
% OUTPUTS:
%   obj - A tapas_Huge object containing the estimation result in the
%         'posterior' property. 
%
% EXAMPLES:
%   [obj] = ESTIMATE(obj)    Invert the HUGE model stored in obj. 
% 
%   [obj] = ESTIMATE(obj, 'K', 2)    Set the number of clusters to 2 and
%       invert the HUGE model stored in obj.
% 
%   [obj] = ESTIMATE(obj, 'Verbose', true)    Print progress of estimation
%       to command line. 
% 
%   [obj] = ESTIMATE(obj, 'Dcm', dcms, 'OmitFromClustering', 1)    Import
%       data stored in 'dcms' and omit self-connections from clustering.
% 
% See also TAPAS_HUGE_DEMO
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


%% process and check input
obj = obj.optional_inputs(varargin{:});
assert(obj.N > 0, 'TAPAS:HUGE:NoData', 'Cannot estimate without data.')

tapas_huge_compile( );

% save random number generator seed
seed = rng();

% check if filename for storing result is valid
filename = obj.options.nvp.saveto;
if ~isempty(filename)
    [pathStr, fileStr, extStr] = fileparts(filename);
    if ~strcmpi(extStr,'.mat')
        filename = [filename '.mat'];
        [pathStr, fileStr, extStr] = fileparts(filename);
    end
    if isempty(fileStr)
        fileStr = ['huge' datestr(now,'-yyyymmdd-HHMMSS')];
    end
    filename = fullfile(pathStr, [fileStr, extStr]);
    save(filename, obj);
end

% decide how to treat confounds
switch lower(obj.options.nvp.confoundsvariant)
    case 'default'
        obj.options.confVar = double(obj.M > 0);
    case 'none'
        obj.options.confVar = 0;
    case 'global'
        obj.options.confVar = 1;
    case 'cluster'
        obj.options.confVar = 2;
end
if obj.options.confVar && ~obj.M
    obj.options.confVar = 0;
    warning('TAPAS:HUGE:MissingConfounds',...
        ['HUGE is configured to use group-level confounds, but no values ' ...
         'for confounds were supplied. Proceeding without confounds.']);
end

if obj.K > obj.N
    warning('TAPAS:HUGE:LargeK',...
        'Number of clusters K exceeds number of subjects N.');
end

% check priors
obj.prior = build_prior(obj);

% starting values
obj = prepare_starting_values(obj);


%% model inversion
obj.aux = struct();
fprintf('Inverting HUGE with %u cluster', obj.K);
if obj.K > 1
    fprintf('s');
end
fprintf(' using %s.\n', obj.options.nvp.method);

switch obj.options.nvp.method
    case 'VB'
        obj = obj.vb_invert( );
    case 'MH'
        obj = obj.mh_invert( );
end

%% post processing
obj.aux = [];
obj.options.start = rmfield(obj.options.start, {'subjects', 'clusters'});

% if working with simulated data, calculate balanced purity
obj.posterior.bPurity = [];
if ~isempty(obj.model)
    model = obj.model;
    [~, labels] = max(model.d, [], 2);
    obj.posterior.bPurity = tapas_huge_bpurity(labels, obj.posterior.q_nk);
end
% save additional information
obj.posterior.method = obj.options.nvp.method;
obj.posterior.version = obj.version;
obj.posterior.seed = seed;

% save result
if ~isempty(filename)
    save(filename, obj);
    fprintf('Saved result to "%s".\n', filename);
end

end


function [ prior ] = build_prior( obj )

prior = struct();

% alpha_0
prior.alpha_0 = obj.options.prior.alpha_0;
if isscalar(prior.alpha_0)
    prior.alpha_0 = repmat(prior.alpha_0, obj.K, 1);
end
assert(size(prior.alpha_0, 1) == obj.K && size(prior.alpha_0, 2) == 1 && ...
    all(prior.alpha_0 > 0), 'TAPAS:HUGE:Prior', ...
    'Prior alpha0 must be positive column vector of length %u.', obj.K);

% S_0
prior.S_0 = obj.options.nvp.priorclustervariance;
if isscalar(prior.S_0)
    prior.S_0 = eye(obj.idx.P_c)*prior.S_0;
end
assert(size(prior.S_0, 1) == obj.idx.P_c, 'TAPAS:HUGE:Prior',...
     'PriorClusterVariance must be of size %ux%u.', obj.idx.P_c, obj.idx.P_c);

% m_0
prior.m_0 = obj.options.nvp.priorclustermean;
if isscalar(prior.m_0)
    prior.m_0 = repmat(prior.m_0, 1, obj.idx.P_c);
end
assert(numel(prior.m_0) == obj.idx.P_c, 'TAPAS:HUGE:Prior', ...
       'PriorClusterMean must be row vector of length %u.', obj.idx.P_c);
prior.m_0 = prior.m_0(:)';

% nu_0
prior.nu_0 = obj.options.nvp.priordegree;
assert(all(prior.nu_0(:) > 0), 'TAPAS:HUGE:Prior', ...
    'PriorDegree must be positive scalar.');

% tau_0
prior.tau_0 = obj.options.nvp.priorvarianceratio;
assert(all(prior.tau_0(:) > 0), 'TAPAS:HUGE:Prior', ...
    'PriorCovarianceRatio must be positive scalar.');
% xxxTODO check size consistent

% mu_h
prior.mu_h = obj.options.prior.mu_h;
if isscalar(prior.mu_h)
    prior.mu_h = repmat(prior.mu_h, 1, obj.idx.P_h);
end
assert(size(prior.mu_h, 2) == obj.idx.P_h && ...
       size(prior.mu_h, 1) == 1, 'TAPAS:HUGE:Prior', ...
    'Prior homogenous mean must be row vector of length %u.', obj.idx.P_h);

% Sigma_h
prior.Sigma_h = obj.options.prior.Sigma_h;
if isscalar(prior.Sigma_h)
    prior.Sigma_h = eye(obj.idx.P_h)*prior.Sigma_h;
elseif isvector(prior.Sigma_h)
    assert(length(prior.Sigma_h)>=4, 'TAPAS:HUGE:Prior', ...
        'Prior homogenous covariance must be a vector of length 4.')
    A = double(obj.dcm.a - diag(diag(obj.dcm.a)))*prior.Sigma_h(2) + ...
        eye(obj.R)*prior.Sigma_h(1);
    tmp = double([obj.dcm.b(:);obj.dcm.c(:);obj.dcm.d(:)])*prior.Sigma_h(3);
    tmp = [A(:); tmp; ones(2*obj.R+1,1)*prior.Sigma_h(4)];
    prior.Sigma_h = diag(tmp(obj.idx.homogenous));
end
try    
    assert(size(prior.Sigma_h, 1) == obj.idx.P_h && ...
        issymmetric(prior.Sigma_h), '', '');
    chol(prior.Sigma_h);
catch
    error('TAPAS:HUGE:Prior', ['Prior hemodynamic covariance must ' ...
        'be symmetric, positive definite matrix of size %u.'], obj.idx.P_h);
end

% mean and variance of lambda
prior.mu_lambda = obj.options.prior.mu_lambda;
assert(prior.mu_lambda > 0, 'TAPAS:HUGE:Prior', ...
    'Prior mean of noise precision must be positive scalar.'); 
prior.s2_lambda = obj.options.prior.s2_lambda;
assert(prior.s2_lambda > 0, 'TAPAS:HUGE:Prior', ...
    'Prior variance of noise precision must be positive scalar.'); 

% m_beta_0
prior.m_beta_0 = obj.options.prior.m_beta_0;
if isscalar(prior.m_beta_0)
    prior.m_beta_0 = repmat(prior.m_beta_0, obj.M, 1);
end
assert(size(prior.m_beta_0, 1) == obj.M && ...
       size(prior.m_beta_0, 2) == 1, 'TAPAS:HUGE:Prior', ...
    'Prior confound mean must be column vector of length %u.', obj.M);

% S_beta_0
prior.S_beta_0 = obj.options.prior.S_beta_0;
if isscalar(prior.S_beta_0)
    prior.S_beta_0 = eye(obj.M)*prior.S_beta_0;
end
try    
    assert(size(prior.S_beta_0, 1) == obj.M && ...
        issymmetric(prior.S_beta_0), '', '');
    chol(prior.S_beta_0);
catch
    error('TAPAS:HUGE:Prior', ['Prior confound covariance must ' ...
        'be symmetric, positive definite matrix of size %u.'], obj.M);
end

end


function [ obj ] = prepare_starting_values( obj )
% subject-level
% initialize starting values to prior mean
stVal1 = repmat([obj.prior.m_0, obj.prior.mu_h], obj.N, 1);

if ~isempty(obj.options.nvp.startingvaluedcm) && ...
        isnumeric(obj.options.nvp.startingvaluedcm) % custom starting values
    stVal1 = obj.options.nvp.startingvaluedcm;    
    [stN, stP] = size(stVal1);
    assert(stN == obj.N && stP == obj.idx.P_c + obj.idx.P_h,...
        'TAPAS:HUGE:StartingValues',...
        'Size of subject-level starting values does not agree with model.')
elseif strcmpi(obj.options.nvp.startingvaluedcm, 'spm') % use SPM estimates
    for n = 1:obj.N
        try
            tmp = [obj.data(n).spm.A(:); obj.data(n).spm.B(:); ...
                obj.data(n).spm.C(:); obj.data(n).spm.D(:); ...
                obj.data(n).spm.transit(:); obj.data(n).spm.decay(:);...
                obj.data(n).spm.epsilon(:)];
            stVal1(n,:) = full(tmp([obj.idx.clustering; obj.idx.homogenous]));
        catch
            fprintf('Missing or invalid SPM estimate for subject %u.' , n);
            fprintf(' Using prior as starting value instead.\n');
        end
    end
end

% randomize starting values
if obj.options.nvp.randomize && ~(strcmpi(obj.options.nvp.method, 'MH') && ...
        ~obj.options.mh.nSteps.dcm) %%% TODO remove exception and introduce new method for clustering only
    stVal1 = stVal1 + randn(size(stVal1)).*obj.options.start.dcm;
end

obj.options.start.subjects = stVal1;

% cluster-level
if ~isempty(obj.options.nvp.startingvaluegmm) && ...
        isnumeric(obj.options.nvp.startingvaluegmm) % custom starting values
    stVal2 = obj.options.nvp.startingvaluegmm;    
    [stN, stP] = size(stVal2);
    assert(stN == obj.K && stP == obj.idx.P_c,...
        'TAPAS:HUGE:StartingValues',...
        'Size of cluster-level starting values does not agree with model.')
else
    stVal2 = repmat(obj.prior.m_0, obj.K, 1);
end

% randomize starting values
if obj.options.nvp.randomize && ~(strcmpi(obj.options.nvp.method, 'MH') && ...
        ~obj.options.mh.nSteps.clusters) %%% TODO remove exception and introduce new method for clustering only
    stVal2 = stVal2 + randn(size(stVal2)).*obj.options.start.gmm;
end

obj.options.start.clusters = stVal2;

end



