%% [DcmResults] = tapas_huge_invert(DCM, K, priors, verbose, randomize, seed)
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
%       hemSigma:      prior covariance of hemodynamic parameters(Sigma_h
%                      in Fig.1 of REF [1]) 
%       noiseInvScale: prior inverse scale of observation noise (b_0 in
%                      Fig.1 of REF [1]) 
%       noiseShape:    prior shape parameter of observation noise (a_0 in
%                      Fig.1 of REF [1])
%               (you may use tapas_huge_build_prior(DCM) to generate this
%               struct)
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
% REFERENCE:
% [1] Yao Y, Raman SS, Schiek M, Leff A, Frässle S, Stephan KE (2018).
%     Variational Bayesian Inversion for Hierarchical Unsupervised
%     Generative Embedding (HUGE). NeuroImage, 179: 604-619
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
function [DcmResults] = tapas_huge_invert(DCM, K, priors, verbose, randomize, seed)
%% check input
if nargin >= 6
    rng(seed);
else
    seed = rng();
end

if ~isfield(DCM,'listBoldResponse')
    try
        DcmInfo = tapas_huge_import_spm(DCM);
    catch err
        disp('tapas_huge_invert: Unsupported format.');
        disp('Use cell array of DCM in SPM format as first input.');
        rethrow(err);
    end
else
    DcmInfo = DCM;
end

if nargin < 5
    randomize = false;
end

if nargin < 4
    verbose = false;
end

if nargin < 3
    priors = tapas_huge_build_prior(DcmInfo);
end

assert(K>0,'TAPAS:HUGE:clusterSize',...
    'Cluster size K must to be positive integer');

% compile integrator
tapas_huge_compile();


%% settings
DcmResults = struct();
DcmResults.maxClusters = K;
DcmResults.priors = priors;

% variational parameters
% stopping criterion: minimum increase in free energy
DcmResults.epsEnergy = 1e-5;
% stopping cirterion: maximum number of iterations
DcmResults.nIterations = 1e3;

% computational and technical parameters
% method for calculating jacobian matrix
DcmResults.fnJacobian = @tapas_huge_jacobian;
% small constant to be added to the diagonal of inv(postDcmSigma) for
% numerical stability
DcmResults.diagConst = 1e-10;
% keep history of parameters and important auxiliary variables
DcmResults.bKeepTrace = false;
% keep history of response related auxiliary variables
% has no effect if bKeepTrace is false
DcmResults.bKeepResp = false;
DcmResults.bVerbose = verbose;

% set update schedule
DcmResults.schedule.dfDcm = 50;
DcmResults.schedule.dfClusters = 10;
DcmResults.schedule.itAssignment = 1;
DcmResults.schedule.itCluster = 1;
DcmResults.schedule.itReturn = 5;
DcmResults.schedule.itKmeans = 1;


%% randomize starting values
if randomize
    init = struct();
    init.dcmMean = repmat([DcmResults.priors.clustersMean,...
        DcmResults.priors.hemMean], DcmInfo.nSubjects, 1);
    init.dcmMean = init.dcmMean + randn(size(init.dcmMean))*.05;

    init.clustersMean = repmat(...
        DcmResults.priors.clustersMean, DcmResults.maxClusters,1);
    init.clustersMean = init.clustersMean + ...
        randn(size(init.clustersMean))*.05;
    DcmResults.init = init;
end


%% call VB inversion
DcmResults = tapas_huge_inv_vb(DcmInfo, DcmResults);

DcmResults.seed = seed;
DcmResults.ver = '201809';

end














