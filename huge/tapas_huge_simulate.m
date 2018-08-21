%% [ DcmInfo ] = tapas_huge_simulate( options )
%
% Generates a synthetic HUGE dataset for DCM for fMRI.
%
% INPUT:
%       options - struct containing information on parameters of the
%                 generative model to be used in the simulation.
%                 (see tapas_huge_generate_examples.m for an example)
%
% OUTPUT:
%       DcmInfo - struct containing DCM model specification and BOLD time
%                 series.
%
% REFERENCE:
%
% Yao Y, Raman SS, Schiek M, Leff A, Frässle S, Stephan KE (2018).
% Variational Bayesian Inversion for Hierarchical Unsupervised Generative
% Embedding (HUGE). NeuroImage, 179: 604-619
% 
% https://doi.org/10.1016/j.neuroimage.2018.06.073
%

% Author: Yu Yao (yao@biomed.ee.ethz.ch)
% Copyright (C) 2018 Translational Neuromodeling Unit
%                    Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
% 
% This file is part of TAPAS, which is released under the terms of the GNU
% General Public Licence (GPL), version 3. For further details, see
% <http://www.gnu.org/licenses/>.
% 
% This software is intended for research only. Do not use for clinical
% purpose. Please note that this toolbox is in an early stage of
% development. Considerable changes are planned for future releases. For
% support please refer to:
% https://github.com/translationalneuromodeling/tapas/issues
%
function [ DcmInfo ] = tapas_huge_simulate( options )
%% process input
rngSeed = rng();
% compile integrator
tapas_huge_compile();

K = numel(options.N_k); % number of clusters
N = sum(options.N_k);   % number of subjects
R = options.R;          % number of regions

% inputs
u = options.input.u;
if iscell(u)
    u = u(:)';
    assert(length(u)==N,...
        'TAPAS:HUGE:NumberOfInputs',...
        'Number of input arrays inconsistent with number of subjects');
    L = unique(cellfun(@size,u,repmat({2},size(u))));
    assert(isscalar(L(:)),...
        'TAPAS:HUGE:InputSize',...
        'Size of inputs inconsistent');
else
    L = size(u,2);
    u = repmat({u},1,N);
end

% number of parameters for fully connected DCM
nParametersConnect = R^2 + R^2*L + R*L + R^3;
nParameters = nParametersConnect + 3*R;

% clusters
mu_k = zeros(K,nParametersConnect);
mu_k(:,options.mu_k.idx) = options.mu_k.value;
p_c = numel(options.mu_k.idx);
assert(size(mu_k,1)==K,...
    'TAPAS:HUGE:NumberOfClusters',...
    'Number of mean vectors inconsistent with number of clusters');
sigma_k = options.sigma_k;
if isscalar(sigma_k)
    sigma_k = repmat(sigma_k,size(options.mu_k.value));
elseif iscolumn(sigma_k)
    sigma_k = repmat(sigma_k,1,p_c);
end
assert(all(size(sigma_k)==size(options.mu_k.value)),...
    'TAPAS:HUGE:NumberOfClusters',...
    'Size of sigma_k inconsistent with mu_k');

% hemodynamics
p_h = 2*R + 1;
mu_h = options.mu_h;
assert(all(size(mu_h)==[1,p_h]),...
    'TAPAS:HUGE:HemodynamicsSize',...
    'Size of mu_h must be 1x2*R+1');
sigma_h = options.sigma_h;
if isscalar(sigma_h)
    sigma_h = repmat(sigma_h,1,p_h);
end
assert(all(size(sigma_h)==size(mu_h)),...
    'TAPAS:HUGE:HemodynamicsSize',...
    'mu_h and sigma_h must have same size');

% observation noise
snrBold = options.snr;

%% assemble DcmInfo struct
DcmInfo = struct();

DcmInfo.nSubjects = N;
DcmInfo.nStates = R;
DcmInfo.nParameters = nParameters;
DcmInfo.connectionIndicator = options.mu_k.idx;
DcmInfo.noConnectionIndicator = 1:nParametersConnect;
DcmInfo.noConnectionIndicator(DcmInfo.connectionIndicator) = [];
DcmInfo.nNoConnections = nParametersConnect - p_c;
DcmInfo.nConnections = p_c;

dummy = zeros(1,nParameters);
dummy(:,options.mu_k.idx) = 1;
dummy = tapas_huge_pack_params(dummy,[NaN NaN R L]);
DcmInfo.adjacencyA = logical(dummy{1});
DcmInfo.adjacencyB = logical(dummy{3});
DcmInfo.adjacencyC = logical(dummy{2});
DcmInfo.adjacencyD = logical(dummy{4});
DcmInfo.dcmTypeB = any(DcmInfo.adjacencyB(:));
DcmInfo.dcmTypeD = any(DcmInfo.adjacencyD(:));

DcmInfo.hemParam = options.hemParam;
DcmInfo.nInputs = L;
DcmInfo.nTime = cellfun(@size,u,repmat({1},1,length(u)));
DcmInfo.listU = u;
% ratio of sampling rate between input and BOLD
if isscalar(options.input.trSteps)
    DcmInfo.trSteps = repmat(options.input.trSteps,1,N);
else
    assert(numel(options.input.trSteps)==N,...
        'TAPAS:HUGE:NumberOfInputs',...
        'Length of trSteps inconsistent with number of subjects');
    DcmInfo.trSteps = options.input.trSteps;
end
% repetition time
if isscalar(options.input.trSeconds)
    DcmInfo.trSeconds = repmat(options.input.trSeconds,1,N);
else
    assert(numel(options.input.trSeconds)==N,...
        'TAPAS:HUGE:NumberOfInputs',...
        'Length of trSeconds inconsistent with number of subjects');
    DcmInfo.trSeconds = options.input.trSeconds;
end
% sampling time step of input
DcmInfo.timeStep = DcmInfo.trSeconds./DcmInfo.trSteps;

DcmInfo.ClusterParameters.clusterMeans = mu_k;
DcmInfo.ClusterParameters.clusterStds = sigma_k;
DcmInfo.ClusterParameters.clusterCovariances = zeros(p_c,p_c,K);
for k = 1:K
    DcmInfo.ClusterParameters.clusterCovariances(:,:,k) = ...
        diag(sigma_k(k,:).^2);
end




%% generate subject-specific BOLD responses

listN = cumsum(options.N_k);
DcmInfo.listBoldResponse = cell(1,N);
DcmInfo.listParameters = cell(N,2);
DcmInfo.listResponseTimeIndices = cell(1,N);
DcmInfo.trueLabels = zeros(1,N);

for n = 1:N
    
    k = nnz(n>listN) + 1; % current cluster index
    DcmInfo.trueLabels(n) = k;
    DcmInfo.listResponseTimeIndices{n} = ...
        DcmInfo.trSteps(n):DcmInfo.trSteps(n):DcmInfo.nTime(n);
    
    bStable = false;
    while ~bStable
        
        % draw subject-specific parameter vector
        mu_n = zeros(1,nParameters);
        mu_n(DcmInfo.connectionIndicator) = randn(1,p_c).*sigma_k(k,:);
        mu_n(nParametersConnect+1:nParametersConnect+p_h) = ...
            randn(1,p_h).*sigma_h;
        mu_n = mu_n + [mu_k(k,:),mu_h,zeros(1,R-1)];
        
        % simulate response BOLD
        response = tapas_huge_bold(mu_n,DcmInfo,n);
        
        % check if DCM stable
        dcmParameters = tapas_huge_pack_params(mu_n,[NaN NaN R L]);
        responseSum = sum(response(:));        
        bStable = ~isnan(responseSum) && ~isinf(abs(responseSum)) && ...
                  max(svd(dcmParameters{1}))<1;

    end
    
    % noise model
    lambda_nr = snrBold./var(response,0,1);
    DcmInfo.listBoldResponse{n} = response + ...
        bsxfun(@rdivide,randn(size(response)),sqrt(lambda_nr));
    DcmInfo.listBoldResponse{n} = DcmInfo.listBoldResponse{n}(:);

    DcmInfo.listParameters{n,1} =  dcmParameters;
    DcmInfo.listParameters{n,2} = lambda_nr;
    
end


% save rng seed
DcmInfo.rngSeed = rngSeed;
DcmInfo.tag = 'TAPAS:HUGE:v0';


end

