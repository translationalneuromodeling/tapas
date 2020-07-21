function [ obj ] = simulate( obj, clusters, sizes, varargin )
% Generate synthetic task-based fMRI time series data, using HUGE as a
% generative model. 
% 
% INPUTS:
%   obj      - A tapas_Huge object.
%   clusters - A cell array containing DCM structs in SPM's DCM format,
%              indicating the DCM network structure and cluster mean
%              parameters.
%   sizes    - A vector containing the number of subjects for each cluster.
% 
% OPTIONAL INPUTS:
%   This function accepts optional name-value pair arguments. For a list of
%   valid name-value pairs, see examples below.
% 
% OUTPUTS:
%   obj - A tapas_Huge object containing the simulated fMRI time series in
%         its 'data' property and the ground truth parameters in its
%         'model' property.
%
% EXAMPLES:
%   [obj] = SIMULATE(obj,clusters,sizes)    Simulate fMRI time series with
%       cluster mean parameters given in 'clusters' and number of subjects
%       given in 'sizes'.
% 
%   [obj] = SIMULATE(obj,clusters,sizes,'Snr',1)    Set signal-to-noise
%       ratio of fMRI data to 1.
% 
%   [obj] = SIMULATE(obj,clusters,sizes,'NoiseFloor',0.1)    Set minimum
%       noise variance to 0.1.
% 
%   [obj] = SIMULATE(obj,clusters,sizes,'confounds',confounds)    Introduce
%       group-level confounds (like sex or age).
% 
%   [obj] = SIMULATE(obj,clusters,sizes,'beta',beta)    Set coefficients
%       for group-level confounds.
% 
%   [obj] = SIMULATE(obj,clusters,sizes,'variant',2)    Set confound
%       variant to 2 (i.e., clusters do not share confound coefficients).
% 
%   [obj] = SIMULATE(obj,clusters,sizes,'Inputs',U)    Introduce subject-
%       specific experimental stimuli. 'U' must be an array or cell array
%       of structs with fields 'dt and 'u'.
% 
%   [obj] = SIMULATE(obj,clusters,sizes,'OmitFromClustering',omit)
%       Designate DCM parameters to be excluded from clustering model.
%       Excluded parameters still exist in the DCM network structure, but
%       are drawn from the same distribution for all subjects. 'omit'
%       should be a struct with fields a, b, c, and/or d which are bool
%       arrays with sizes matching the corresponding fields of the DCMs. If
%       omit is an array, it is interpreted as the field a. If omit is 1,
%       it is expanded to an identity matrix of suitable size.
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


%% initialize
% save random number generator seed
seed = rng();

tapas_huge_compile( );


%% check inputs
% cluster means
if isvector(clusters)&&isstruct(clusters)
    try
        clusters = {clusters(:).DCM}';
    catch
        clusters = num2cell(clusters);
    end
else
    assert(iscell(clusters),'TAPAS:HUGE:Sim:InputFormat',...
        'clusters must be a cell array of DCMs in SPM format');
end

% cluster sizes
obj.K = numel(clusters);
assert(numel(sizes) == obj.K, 'TAPAPS:HUGE:Sim:MismatchK', ...
    'Number of elements of clusters and sizes must be equal.');
N = sum(sizes);


%% process optional inputs
options = struct( ...
    'snr', 1, ...
    'noisefloor', .01, ...
    'confounds', [], ...
    'beta', [], ...
    'variant', 0, ...
    'inputs', [], ...
    'omitfromclustering', struct());

if nargin > 4
    options = tapas_huge_parse_inputs( options, varargin );
end
% check options
options.noisefloor = max(options.noisefloor, 2*eps); % minimum noise floor

if ~isempty(options.confounds) && ~isempty(options.beta)
    % normalize confounds to zero mean and unit variance
    x_n = bsxfun(@minus, options.confounds, mean(options.confounds));
    tmp = std(options.confounds);
    tmp(tmp < eps) = 1;
    x_n = bsxfun(@rdivide, x_n, tmp);
    if ~options.variant
        options.variant = 1; % default variant
    elseif options.variant == 2 && size(options.beta, 3) == 1
        options.beta = repmat(options.beta, 1, 1, obj.K); % expand beta
    end
else
    x_n = [];
end

if isvector(options.inputs) && isstruct(options.inputs)
    options.inputs = num2cell(options.inputs);
end


%% generate group-level data
% build ground truth struct
model = struct();
model.pi = sizes(:)./N;
model.d = false(N, obj.K); % cluster labels (in one-hot format)

d = cell(obj.K, 1);
for k = 1:obj.K
    d{k} = repmat(k, sizes(k), 1);
    if ~isfield(clusters{k}.Y, 'y')
        rSmp = clusters{k}.Y.dt/clusters{k}.U.dt; %%% TODO 16
        clusters{k}.Y.y = zeros(length(clusters{k}.U.u(rSmp:rSmp:end, 1)), ...
            clusters{k}.n);
    end
end
d = cell2mat(d);

obj = obj.import(clusters, [], options.omitfromclustering);

model = group_level( model, obj, clusters );
model.d(sub2ind([N, obj.K], 1:N , d')) = true;


%% generate subject-level data
listDcms = clusters(d);
% subject-specific parameters
model.theta_c = zeros(N, obj.idx.P_c);
model.theta_h = zeros(N, obj.idx.P_h);
% noise precision
model.lambda = zeros(N, clusters{1}.n);

for n = 1:N
    
    dcm = listDcms{n};
    % apply confounds
    if ~isempty(x_n)
        theta = zeros(1, obj.idx.P_f);
        theta(obj.idx.clustering) = x_n(n,:)*options.beta(:,:,d(n));
        dcm = add_theta(dcm, theta);
    end
    
    % generate DCM parameters
    [ dcm, theta_c, theta_h ] = subject_level( dcm, obj.idx, ... 
        model.Sigma_k(:,:,d(n)), model.Sigma_h);
    model.theta_c(n,:) = model.mu_k(d(n), :) + theta_c;
    model.theta_h(n,:) = model.mu_h + theta_h;
    
    % generate bold time series
    if ~isempty(options.inputs)
        dcm.U = options.inputs{n};
    end
    L = size(dcm.U.u, 2);
    rSmp = dcm.Y.dt/dcm.U.dt; %%% fix
    
    data = struct( 'bold', zeros(fix(size(dcm.U.u, 1)/rSmp), dcm.n), ...
        'tr', dcm.Y.dt, 'te', 0.04, 'res', []);
    if isfield( dcm, 'TE')
        data.te = dcm.TE;
    end
    
    bold = obj.bold_gen( dcm.Ep, data, dcm.U, obj.options.hemo, dcm.n, L );
    
    assert(~(any(isnan(bold(:))) || any(isinf(bold(:)))), ...
        'TAPAS:HUGE:Sim:Stability', ['Simulated DCM parameters for ' ...
        'subject %u lead to instable DCM.'], n);

    % add measurement noise
    sigmaNoise = max(options.noisefloor, sqrt(var(bold)./options.snr));
    model.lambda(n, :) = 1./sigmaNoise;
    dcm.Y.y = bsxfun(@times, randn(size(bold)), sigmaNoise) - bold;
    % note: minus sign is because obj.bold_gen outputs epsilon
    
    dcm.delays = zeros(obj.R, 1);
    listDcms{n} = dcm;
    
end

%% add DCMs and finalize forward model
obj = obj.import(listDcms, options.confounds, options.omitfromclustering);
     
model.x_n = x_n;
if options.variant == 1
    model.beta = model.beta(:,:,1);
end
model.beta =  options.beta;   
model.options = options;
model.seed = seed;

obj.model = model;


end


function [ model ] = group_level( model, obj, clusters )
%GROUP_LEVEL parse group-level parameters

model.mu_k = zeros(obj.K, obj.idx.P_c);
model.Sigma_k = zeros(obj.idx.P_c, obj.idx.P_c, obj.K);
for k = 1:obj.K
    % cluster mean
    Ep = clusters{k}.Ep;
    clMean = [Ep.A(:); Ep.B(:); Ep.C(:); Ep.D(:); Ep.transit(:); ...
        Ep.decay(:); Ep.epsilon(:) ];
    model.mu_k(k,:) = clMean(obj.idx.clustering)';
    
    % cluster covariance
    Cp = full(clusters{k}.Cp);
    model.Sigma_k(:,:,k) = Cp(obj.idx.clustering, obj.idx.clustering);
    
    % mean and covariance of homogenous part
    if k == 1
        model.mu_h = clMean(obj.idx.homogenous)';
        model.Sigma_h = Cp(obj.idx.homogenous, obj.idx.homogenous);
    end
end
end


function [ dcm, theta_c, theta_h ] = subject_level( dcm, idx, Sigma_k, Sigma_h )
%SUBJECT_LEVEL generate subject-level DCM parameters given cluster mean

theta = zeros(1, idx.P_f);
% clustering part
[U, S] = svd(Sigma_k);
theta_c = (randn(1, idx.P_c).*sqrt(diag(S)'))*U';
theta(idx.clustering) = theta_c;

% homogenous part
[U, S] = svd(Sigma_h);
theta_h  = (randn(1, idx.P_h).*sqrt(diag(S)'))*U';
theta(idx.homogenous) = theta_h;

dcm = add_theta(dcm, theta);

end


function [ dcm ] = add_theta( dcm, theta)
%ADD_THETA update DCM parameters

dcm.Ep.A = dcm.Ep.A + reshape(theta(1:numel(dcm.a)), size(dcm.a));
theta(1:numel(dcm.a)) = [];

dcm.Ep.B = dcm.Ep.B + reshape(theta(1:numel(dcm.b)), size(dcm.b));
theta(1:numel(dcm.b)) = [];

dcm.Ep.C = dcm.Ep.C + reshape(theta(1:numel(dcm.c)), size(dcm.c));
theta(1:numel(dcm.c)) = [];

dcm.Ep.D = dcm.Ep.D + reshape(theta(1:numel(dcm.d)), size(dcm.d));
theta(1:numel(dcm.d)) = [];

dcm.Ep.transit = dcm.Ep.transit + reshape(theta(1:numel(dcm.Ep.transit)), ...
    size(dcm.Ep.transit));
theta(1:numel(dcm.Ep.transit)) = [];

dcm.Ep.decay = dcm.Ep.decay + reshape(theta(1:numel(dcm.Ep.decay)), ...
    size(dcm.Ep.decay));
theta(1:numel(dcm.Ep.decay)) = [];

dcm.Ep.epsilon = dcm.Ep.epsilon + reshape(theta(1:numel(dcm.Ep.epsilon)), ...
    size(dcm.Ep.epsilon));

end

