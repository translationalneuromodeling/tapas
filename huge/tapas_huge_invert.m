function [DcmResults] = tapas_huge_invert(DCM, K, priors, verbose, randomize, seed)
%   WARNING: This function is deprecated and will be removed in a future
%   version of this toolbox. Please use the new object-oriented interface
%   provided by the tapas_Huge class.
%   
% Invert hierarchical unsupervised generative embedding (HUGE) model.
%
% INPUT:
%   DCM       - cell array of DCM in SPM format
%   K         - number of clusters (set K to one for empirical Bayes)
%
% OPTIONAL INPUT:
%   priors    - model priors stored in a struct containing the
%               following fields:
%       alpha:         parameter of Dirichlet prior (alpha_0 in Fig.1 of
%                      REF [1])
%       clustersMean:  prior mean of clusters (m_0 in Fig.1 of REF [1])
%       clustersTau:   tau_0 in Fig.1 of REF [1]
%       clustersDeg:   degrees of freedom of inverse-Wishart prior (nu_0 in
%                      Fig.1 of REF [1]) 
%       clustersSigma: scale matrix of inverse-Wishart prior (S_0 in Fig.1
%                      of REF [1]) 
%       hemMean:       prior mean of hemodynamic parameters (mu_h in Fig.1
%                      of REF [1]) 
%       hemSigma:      prior covariance of hemodynamic parameters (Sigma_h
%                      in Fig.1 of REF [1]) 
%       noiseInvScale: prior inverse scale of observation noise (b_0 in
%                      Fig.1 of REF [1]) 
%       noiseShape:    prior shape parameter of observation noise (a_0 in
%                      Fig.1 of REF [1])
%   verbose   - activates command line output (prints free energy
%               difference, default: false)
%   randomize - randomize starting values (default: false). WARNING:
%               randomizing starting values can cause divergence of DCM.
%   seed      - seed for random number generator
% 
% OUTPUT:
%   DcmResults - struct used for storing the results from VB. Posterior
%                parameters are stored in DcmResults.posterior, which is a
%                struct containing the following fields:
%       alpha:               parameter of posterior over cluster weights
%                            (alpha_k in Eq.(15) of REF [1]) 
%       softAssign:          posterior assignment probability of subjects 
%                            to clusters (q_nk in Eq.(18) in REF [1])
%       clustersMean:        posterior mean of clusters (m_k in Eq.(16) of
%                            REF [1]) 
%       clustersTau:         tau_k in Eq.(16) of REF [1]
%       clustersDeg:         posterior degrees of freedom (nu_k in Eq.(16) 
%                            of REF [1])
%       clustersSigma:       posterior scale matrix (S_k in Eq.(16) of
%                            REF [1]) 
%       logDetClustersSigma: log-determinant of S_k
%       dcmMean:             posterior mean of DCM parameters (mu_n in
%                            Eq.(19) of REF [1])  
%       dcmSigma:            posterior covariance of hemodynamic
%                            parameters (Sigma_n in Eq.(19) of REF [1]) 
%       logDetPostDcmSigma:  log-determinant of Sigma_n
%       noiseInvScale:       posterior inverse scale of observation noise
%                            (b_n,r in Eq.(21) of REF [1]) 
%       noiseShape:          posterior shape parameter of observation noise
%                            (a_n,r in Eq.(21) of REF [1])
%       meanNoisePrecision:  posterior mean of precision of observation
%                            noise (lambda_n,r in Eq.(23) of REF [1])
%       modifiedSumSqrErr:   b'_n,r in Eq.(22) of REF [1]
%
% See also tapas_Huge, tapas_Huge.estimate, tapas_huge_demo
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


wnMsg = ['This function is deprecated and will be removed in a future ' ...
	     'version of this toolbox. Please use the new object-oriented ' ...
         'interface provided by the tapas_Huge class.'];
warning('tapas:huge:deprecated',wnMsg)

%% check input
if isvector(DCM)&&isstruct(DCM)
    try
        DCM = {DCM(:).DCM}';
    catch
        DCM = num2cell(DCM);
    end
else
    assert(iscell(DCM),'TAPAS:HUGE:inputFormat',...
        'DCM must be cell array of DCMs in SPM format');
end

%% settings
opts = {'K', K};
opts = [opts, {'Dcm', DCM}];

if nargin >= 6
    opts = [opts, {'Seed', seed}];
end

if nargin >= 5
    opts = [opts, {'Randomize', randomize}];
end

if nargin >= 4
    opts = [opts, {'Verbose', verbose}];
end

if nargin >= 3
    dcm = DCM{1};
    nConnections = nnz([dcm.a(:);dcm.b(:);dcm.c(:);dcm.d(:)]);
    priors.clustersSigma = priors.clustersSigma/...
                       (priors.clustersDeg - nConnections - 1);
                   
    opts = [opts, {'PriorVarianceRatio', priors.clustersTau, ...
        'PriorDegree', priors.clustersDeg, ...
        'PriorClusterVariance', priors.clustersSigma, ...
        'PriorClusterMean', priors.clustersMean}];
end

assert(K>0,'TAPAS:HUGE:clusterSize',...
    'Cluster size K must to be positive integer');

%% invert model
obj = tapas_Huge(opts{:});
obj = obj.estimate();

DcmResults.freeEnergy = obj.posterior.nfe;
DcmResults.maxClusters = K;
DcmResults.rngSeed = obj.posterior.seed;
DcmResults.posterior = struct( ...
    'alpha', obj.posterior.alpha, ...
    'softAssign', obj.posterior.q_nk, ...
    'clustersMean' , obj.posterior.m        , ... 
    'clustersTau', obj.posterior.tau  , ...
    'clustersDeg'  , obj.posterior.nu     , ... 
    'clustersSigma'   , obj.posterior.S    , ... 
    'logDetClustersSigma', [] , ...
    'dcmMean', obj.posterior.mu_n  , ...
    'dcmSigma', obj.posterior.Sigma_n  , ...
    'logDetPostDcmSigma', []  , ...
    'noiseShape', obj.posterior.a, ...
    'noiseInvScale', obj.posterior.b  , ...
    'meanNoisePrecision', obj.posterior.a./obj.posterior.b  , ...
    'modifiedSumSqrErr', []);
DcmResults.residuals = obj.trace.epsilon;

end














