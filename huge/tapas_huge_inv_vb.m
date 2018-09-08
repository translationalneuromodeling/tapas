%% [ DcmResults ] = tapas_huge_inv_vb( DcmInfo, DcmResults )
%
% Inversion of hierarchical unsupervised generative embedding (HUGE) for
% DCM for fMRI using variational Bayes.
%
% INPUT:
%       DcmInfo    - struct containing DCM model specification and BOLD
%                    time series in DcmInfo format
%                    (see tapas_huge_simulate.m for an example)
%       DcmResults - struct containing prior specification and settings.
%
% OUTPUT:
%       DcmResults - struct used for storing the results from VB
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
function [ DcmResults ] = tapas_huge_inv_vb( DcmInfo, DcmResults )
%
KMEANS_REP = 10;


%% extract DCM information
nStates = DcmInfo.nStates; %%% R
nSubjects = DcmInfo.nSubjects; %%% N

listBoldResponse = DcmInfo.listBoldResponse; %%% y

nMeasurements = zeros(nSubjects,1);
nMeasurementsPerState = zeros(nSubjects,1);

for iSubject = 1:nSubjects
    % %%% dim(y)
    nMeasurements(iSubject) = length(listBoldResponse{iSubject});
    nMeasurementsPerState(iSubject) = nMeasurements(iSubject)/nStates; %%% q_r

end

[nParameters,idxParamsInf,idxSelfCon] = tapas_huge_count_params(DcmInfo);
% % % % % nDcmParametersAll = nParameters(1,3);
nDcmParamsInfCon = nParameters(2,1); %%% d_c
% % % % % nDcmParamsInfHem = nParameters(2,2); %%% d_h
nDcmParamsInfAll = nParameters(2,3); %%% d

% measurement function %%% g(theta)
fnGenerateResponse = @tapas_huge_predict;

counts.nSubjects = nSubjects;
counts.nParameters = nParameters;
counts.nMeasurements = nMeasurements;
counts.nMeasurementsPerState = nMeasurementsPerState;


%% parameters for variational Bayes:
% maximum number of iterations
nIterations = DcmResults.nIterations;

% free energy threshold
epsEnergy = DcmResults.epsEnergy;

% parameters for jacobian calculation
paramsJacobian.dimOut = nMeasurements;

% method for calculating the jacobian matrix
fnJacobian = DcmResults.fnJacobian;

% diagonal constant for numerical stability
diagConst = DcmResults.diagConst;
diagConst = diagConst*eye(nDcmParamsInfAll);

% keep history of important variables
bKeepTrace = DcmResults.bKeepTrace;

% keep history of response related variables
% note: has no effect if bKeepTrace is false
bKeepResp = DcmResults.bKeepResp;

% switch off command line output
bVerbose = DcmResults.bVerbose;

% update schedule
schedule = DcmResults.schedule;

% number of clusters
nClusters = DcmResults.maxClusters; %%% K    
if nClusters > nSubjects
    nClusters = nSubjects;
    DcmResults.maxClusters = nClusters;
    warning('TAPAS:HUGE:NumberOfCluster',...
        ['The number of clusters exceeds the number of subjects and '...
         'has been decreased for efficiency.']);
end
counts.nClusters = nClusters;

% set default parameters to zero
dcmParametersDefault = zeros(1,DcmInfo.nParameters);


%% priors
priors = DcmResults.priors;
% component weights
if isscalar(priors.alpha)
    priors.alpha = repmat(priors.alpha,nClusters,1);
end

priors.hemPrecision = inv(priors.hemSigma);

% prior noise level on fmri data
if isscalar(priors.noiseShape) %%% a_r: shape
    priors.noiseShape = repmat(priors.noiseShape,1,nStates);
end
if isscalar(priors.noiseInvScale) %%% b_r: inverse scale
    priors.noiseInvScale = repmat(priors.noiseInvScale,1,nStates);
end


%% initial values
% initialize posterior covariance of DCM parameters
bInit = isfield(DcmResults,'init');

if bInit&&isfield(DcmResults.init,'covFactor')
    priors.clustersSigma = DcmResults.init.covFactor*priors.clustersSigma;
    bCovFactor = true;
else
    bCovFactor = false;
end


if bInit && isfield(DcmResults.init,'dcmSigma')
    posterior.dcmSigma = DcmResults.init.dcmSigma;
else
    posterior.dcmSigma(1:nDcmParamsInfCon,1:nDcmParamsInfCon) = ...
        priors.clustersSigma;
    posterior.dcmSigma(nDcmParamsInfCon+1:nDcmParamsInfAll,...
                 nDcmParamsInfCon+1:nDcmParamsInfAll) = priors.hemSigma;
    posterior.dcmSigma = repmat(posterior.dcmSigma,[1,1,nSubjects]);
end
posterior.logDetPostDcmSigma = zeros(nSubjects,1);
for iSubject = 1:nSubjects
    posterior.logDetPostDcmSigma(iSubject) = ...
        tapas_util_logdet(posterior.dcmSigma(:,:,iSubject));
end

% initialize posterior mean of DCM parameters
if bInit && isfield(DcmResults.init,'dcmMean')
    posterior.dcmMean = DcmResults.init.dcmMean;
else
    posterior.dcmMean = repmat([priors.clustersMean,priors.hemMean],...
        nSubjects,1);
end
% precalculate Jacobian, error term and sum of squared error
respError = cell(nSubjects,1); %%% epsilon
respJacobian = cell(nSubjects,1); %%% G
posterior.modifiedSumSqrErr = zeros(nSubjects,nStates); %%% hat(b)_nr

for iSubject = 1:nSubjects
    respCurrent = fnGenerateResponse(posterior.dcmMean(iSubject,:).',...
                                                 dcmParametersDefault,...
                                                 idxParamsInf,...
                                                 idxSelfCon,...
                                                 DcmInfo, iSubject);
    respError{iSubject} = listBoldResponse{iSubject} - respCurrent;
    paramsJacobian.dimOut = nMeasurements(iSubject);
    respJacobian{iSubject} = fnJacobian(...
                      fnGenerateResponse,...
                      posterior.dcmMean(iSubject,:).',... %%% mu_n and current m_n
                      paramsJacobian, respCurrent,...
                      dcmParametersDefault, idxParamsInf, ...
                      idxSelfCon, DcmInfo, iSubject);

    % check if initial values cause divergence in response generating
    % function
    if fnc_check_response(respError{iSubject},...
                               respJacobian{iSubject})
        error('TAPAS:HUGE:badInit',...
              'Initial values of DCM Parameters cause divergence.');
    end
    %%% epsilon^T*Q_r*epsilon + tr(Q_r*G_n*Sigma_n*G_n^T)
    posterior.modifiedSumSqrErr(iSubject,:) = ... 
        sum(reshape(...
              respError{iSubject}.^2 + ...
                  sum(...
                      (respJacobian{iSubject}*...
                          posterior.dcmSigma(:,:,iSubject)).*...
                      respJacobian{iSubject},2),...
              nMeasurementsPerState(iSubject),nStates ));
end


% initialize posterior mean and covariance of clusters
if bInit && isfield(DcmResults.init,'clustersMean')
    posterior.clustersMean = DcmResults.init.clustersMean;
else
    posterior.clustersMean = repmat(priors.clustersMean,nClusters,1);
end

if bInit && isfield(DcmResults.init,'clustersSigma')
    posterior.clustersSigma = DcmResults.init.clustersSigma;
else
    posterior.clustersSigma = repmat(priors.clustersSigma,[1,1,nClusters]);
end
posterior.logDetClustersSigma = zeros(nClusters,1); %%% log(det(bar(Sigma)_k))
invClustersSigma = zeros(nDcmParamsInfCon,nDcmParamsInfCon,nClusters); %%% bar(Sigma)_k^-1
for iCluster = 1:nClusters
    posterior.logDetClustersSigma(iCluster) = ...
        tapas_util_logdet(posterior.clustersSigma(:,:,iCluster));
    invClustersSigma(:,:,iCluster) = ...
        inv(posterior.clustersSigma(:,:,iCluster));
end

posterior.noiseInvScale = repmat(priors.noiseInvScale,nSubjects,1);
% note: with the current parameterization 'posterior.noiseShape' is precomputable
posterior.noiseShape = ... %%% a_nr
    repmat(priors.noiseShape + nMeasurementsPerState(iSubject)/2,...
            nSubjects,1);
% mean precision
posterior.meanNoisePrecision = ...
    posterior.noiseShape./posterior.noiseInvScale;


%% prelocating memory and initializing auxiliary variables
weightedSumDcmMean = zeros(nClusters,nDcmParamsInfCon); %%% mu_ck

% initialize soft assign to one cluster only (empirical Bayes)
if bInit && isfield(DcmResults.init,'logSoftAssign')
    logSoftAssign = DcmResults.init.logSoftAssign;
else
    logSoftAssign = ones(nSubjects,nClusters); %%% log(q_nk)
    logSoftAssign(:,1) = 100;
    DcmResults.init.logSoftAssign = logSoftAssign;
end
posterior.softAssign = fnc_exp_norm(logSoftAssign);
nAssign = sum(posterior.softAssign,1).' + realmin;
posterior.alpha = priors.alpha + nAssign; %%% alpha
posterior.clustersTau = priors.clustersTau + nAssign;
posterior.clustersDeg = priors.clustersDeg + nAssign;

partialDcmPrec = zeros(nDcmParamsInfAll); %%% Lambda'_n
partialDcmPrec(nDcmParamsInfCon+1:end,nDcmParamsInfCon+1:end) = ...
    priors.hemPrecision; % note: the lower submatrix is constant

partialDcmMean = zeros(1,nDcmParamsInfAll); %%% mu'_n
partialDcmMean(nDcmParamsInfCon+1:end) = priors.hemMean/priors.hemSigma;

[freeEnergy,feAux] = tapas_huge_nfe(counts,priors,posterior);
feCurrent = freeEnergy;
dF = -1;



%% history

nItSubject = zeros(nSubjects,1);

% keep history of important variables
if bKeepTrace
    histClustersMean = cell(nIterations,1);
    histClustersSigma = cell(nIterations,1);
    histClustersTau = cell(nIterations,1);
    histClustersDeg = cell(nIterations,1);

    histDcmMean = cell(nIterations,1);
    histDcmSigma = cell(nIterations,1);

    histNoiseInvScale = cell(nIterations,1);

    histposterior.softAssign = cell(nIterations,1);
    histPartialDcmMean = cell(nIterations,1);
    histPartialDcmPrec = cell(nIterations,1);
    if bKeepResp
        histRespError = cell(nIterations,1);
        histRespJacobian = cell(nIterations,1);
    end
    
    histFreeEnergy = cell(nIterations,1);
    histFeParts = cell(nIterations,1);

    timeSinceStart = cell(nIterations,1);
    tic;
end

histFe = zeros(nIterations,1);
itSatDcm = nIterations + 1;
itSatClusters = nIterations + 1;


% -------------------------------------------
%% Variational Updates - Main loop
% -------------------------------------------

for iIteration = 1:nIterations



  
    if schedule.itKmeans&&(iIteration == itSatClusters + schedule.itKmeans)

        [kmeansIdx,posterior.clustersMean] = ...
            kmeans(posterior.dcmMean(:,1:nDcmParamsInfCon),...
                   nClusters,'Replicates',KMEANS_REP);
        posterior.softAssign = zeros(nSubjects,nClusters);
        posterior.softAssign(sub2ind([nSubjects,nClusters],...
            (1:nSubjects)',kmeansIdx)) = 1;
        nAssign = sum(posterior.softAssign,1).' + realmin;
        posterior = fnc_set_cluster_cov(priors,posterior,kmeansIdx,nAssign);
        

        % debug info
        DcmResults.kmeans.kmIdx = kmeansIdx;
        DcmResults.kmeans.dcmMean = posterior.dcmMean;
        DcmResults.kmeans.iRelease = iIteration;
        DcmResults.kmeans.dF = dF;
%         DcmResults.kmeans.kmProb = KMEANS_PROB;
        DcmResults.kmeans.kmRep = KMEANS_REP;
    end

    

    if iIteration >= itSatClusters + schedule.itAssignment
    %%% q_nk
    % update soft assignments
    postAlphaDigamma = psi(0,posterior.alpha); % Psi(sum(alpha)) is a constant
    for iCluster = 1:nClusters
        diffMeanDcmCluster = bsxfun(@minus,... %%% mu_cn - mu_k
                         posterior.dcmMean(:,1:nDcmParamsInfCon),...
                         posterior.clustersMean(iCluster,:));
        % note: logSoftAssign is only correct up to a constant 
        logSoftAssign(:,iCluster) = ... %%% log(q_nk)
            -posterior.logDetClustersSigma(iCluster)/2 ...
            +sum(psi(0,...
                (posterior.clustersDeg(iCluster)-...
                    (0:nDcmParamsInfCon-1))/2))/2 ...
            -nDcmParamsInfCon/2/posterior.clustersTau(iCluster)...
            -posterior.clustersDeg(iCluster)/2*...
                reshape( sum( sum( ...                        
                    bsxfun( @times, ...
                            invClustersSigma(:,:,iCluster),...
                            posterior.dcmSigma(1:nDcmParamsInfCon,...
                                         1:nDcmParamsInfCon,:) ), ...
                    1), 2), nSubjects,1) ... %%% -nu/2tr(bar(Sigma)_k^-1*Sigma_cn)
            -posterior.clustersDeg(iCluster)/2*...
                sum((diffMeanDcmCluster/...
                        posterior.clustersSigma(:,:,iCluster)).*...
                     diffMeanDcmCluster,2) ...
            +postAlphaDigamma(iCluster);
    end
    posterior.softAssign = fnc_exp_norm(logSoftAssign); %%% q_nk
    % effective number of samples assigned to each cluster
    nAssign = sum(posterior.softAssign,1).' + realmin; %%% q_k
    %%% pi: alpha
    % update parameters for mixture weights
    posterior.alpha = priors.alpha + nAssign;
    
    end

    
    
    if iIteration >= itSatDcm + schedule.itCluster
    
    
    %%% bar(mu)_k, bar(Sigma)_k
    % update parameters of inverse Wishart
    posterior.clustersTau = priors.clustersTau + nAssign;
    posterior.clustersDeg = priors.clustersDeg + nAssign;
    
    for iCluster = 1:nClusters
        weightedSumDcmMean(iCluster,:) = ...
            sum(bsxfun(@times,...
                       posterior.dcmMean(:,1:nDcmParamsInfCon),...
                       posterior.softAssign(:,iCluster)),...
                1);
        posterior.clustersMean(iCluster,:) = ... %%% (q_k*mu_ck + tau_0*bar(mu)_0)/tau_k
            (weightedSumDcmMean(iCluster,:) + ...
                priors.clustersTau*priors.clustersMean)/...
            posterior.clustersTau(iCluster);
        weightedSumDcmMean(iCluster,:) = ...
            weightedSumDcmMean(iCluster,:)/nAssign(iCluster);
        diffMeanDcmWeighted = bsxfun(@minus,... %%% mu_cn - mu_ck
                          posterior.dcmMean(:,1:nDcmParamsInfCon),...
                          weightedSumDcmMean(iCluster,:));
        posterior.clustersSigma(:,:,iCluster) = ... %%% bar(Sigma)_k = ..
            +priors.clustersSigma ... %%% bar(Sigma)_0 + ..
            +nAssign(iCluster)*priors.clustersTau/...
                    posterior.clustersTau(iCluster)*... %%% q_k*tau_0/tau_k*(mu_ck - bar(mu)_0)*(mu_ck - bar(mu)_0)^T + ..
                (weightedSumDcmMean(iCluster,:) - priors.clustersMean).'* ...
                (weightedSumDcmMean(iCluster,:) - priors.clustersMean) ...
            +diffMeanDcmWeighted.'*...
                bsxfun(@times,diffMeanDcmWeighted,...
                    posterior.softAssign(:,iCluster)) ... %%% sum(q_nk*(mu_cn - mu_ck)*(mu_cn - mu_ck)^T) + ...
            +sum(... %%% Sigma_ck
                bsxfun(...
                    @times,...
                    posterior.dcmSigma(1:nDcmParamsInfCon,1:...
                        nDcmParamsInfCon,:),...
                    reshape(posterior.softAssign(:,iCluster),1,1,nSubjects)...
                       ),...
                3); %%% Sigma_ck
        posterior.logDetClustersSigma(iCluster) = ...
            tapas_util_logdet(posterior.clustersSigma(:,:,iCluster));
        invClustersSigma(:,:,iCluster) = ...
            inv(posterior.clustersSigma(:,:,iCluster)); %%% inv(bar(Sigma)_k)      %%% TODO check rcond
    end    
    end

    
    % cache DCM parameters
    dcmMeanBkp = posterior.dcmMean;
    dcmSigmaBkp = posterior.dcmSigma;
    logDetDcmSigmaBkp = posterior.logDetPostDcmSigma;

    respErrorBkp = respError;
    respJacBkp = respJacobian;
    sumSqErrBkp = posterior.modifiedSumSqrErr;
    

    % DCM update    
    for iSubject = 1:nSubjects

        %%% mu_n, Sigma_n
        % update parameters for distribution over DCM parameters
        weightedClustersPrecision = bsxfun(...
                       @times,... %%% q_nk*nu_k*bar(Sigma)_k^-1
                       invClustersSigma,...
                       reshape(transpose(...
                                posterior.softAssign(iSubject,:)).*...
                                    posterior.clustersDeg,...
                               1,1,nClusters));
        partialDcmPrec(1:nDcmParamsInfCon,1:nDcmParamsInfCon) = ...
            sum(weightedClustersPrecision,3); %%% sum(q_nk*nu_k*bar(Sigma)_k^-1)
        partialDcmMean(1:nDcmParamsInfCon) = zeros(1,nDcmParamsInfCon);
        for iCluster = 1:nClusters
            partialDcmMean(1:nDcmParamsInfCon) = ...
                partialDcmMean(1:nDcmParamsInfCon) + ...
                posterior.clustersMean(iCluster,:)*...
                    weightedClustersPrecision(:,:,iCluster);
        end
        expandedMeanNoisePrecision = ...
            kron(posterior.meanNoisePrecision(iSubject,:),...
                 ones(1,nMeasurementsPerState(iSubject)));

        posterior.dcmSigma(:,:,iSubject) = ... %%% Sigma_n = ...
            inv(... %%% (G^T*bar(Lambda)_n*G_n + ...
                transpose(respJacobian{iSubject})*... %%% TODO check rcond
                    bsxfun(...
                        @times,...
                        respJacobian{iSubject},...
                        expandedMeanNoisePrecision.') + ...
                partialDcmPrec + ... %%% Lambda'_n)^-1
                diagConst); % add a small constant to the diagonal elements
        
        posterior.logDetPostDcmSigma(iSubject) = ...
            tapas_util_logdet(posterior.dcmSigma(:,:,iSubject));

        posterior.dcmMean(iSubject,:) = ...
            (((respError{iSubject}.' + ...
                 posterior.dcmMean(iSubject,:)*respJacobian{iSubject}.').*...
               expandedMeanNoisePrecision)*respJacobian{iSubject} + ...
               partialDcmMean)*...
            posterior.dcmSigma(:,:,iSubject);

        % calculate auxiliary variables for next iteration
        respCurrent = fnGenerateResponse(posterior.dcmMean(iSubject,:).',...
                                         dcmParametersDefault,...
                                         idxParamsInf,idxSelfCon, ...
                                         DcmInfo, iSubject);
        respError{iSubject} = listBoldResponse{iSubject} - respCurrent;
        paramsJacobian.dimOut = nMeasurements(iSubject);
        respJacobian{iSubject} = fnJacobian(...
                              fnGenerateResponse,...
                              posterior.dcmMean(iSubject,:).',... %%% mu_n and current m_n
                              paramsJacobian,...
                              respCurrent,... % current function value
                              dcmParametersDefault, idxParamsInf, ...
                              idxSelfCon, DcmInfo, iSubject);

        posterior.modifiedSumSqrErr(iSubject,:) = ... %%% epsilon^T*Q_r*epsilon + tr(Q_r*G_n*Sigma_n*G_n^T)
            sum(reshape(...
                  respError{iSubject}.^2 + ...
                      sum(...
                          (respJacobian{iSubject}*...
                              posterior.dcmSigma(:,:,iSubject)).*...
                          respJacobian{iSubject},2),...
                  nMeasurementsPerState(iSubject),nStates ));

        % check free energy
        feBkp = feCurrent;
        feAuxBkp = feAux;
        [feCurrent,feAux] = tapas_huge_nfe(counts,priors,posterior,...
                                                     feAux,iSubject);

        % if free energy decreases or response/jacobian contains NaNs
        % cancel update and restore old values
        if feCurrent < feBkp || ...
           fnc_check_response(respError{iSubject},...
                                   respJacobian{iSubject})
       
            posterior.dcmMean(iSubject,:) = dcmMeanBkp(iSubject,:);
            posterior.dcmSigma(:,:,iSubject) = dcmSigmaBkp(:,:,iSubject);
            posterior.logDetPostDcmSigma(iSubject) = ...
                logDetDcmSigmaBkp(iSubject);

            respError{iSubject} = respErrorBkp{iSubject};
            respJacobian{iSubject} = respJacBkp{iSubject};
            posterior.modifiedSumSqrErr(iSubject,:) = ...
                sumSqErrBkp(iSubject,:);
            feCurrent = feBkp;
            feAux = feAuxBkp;

        else
            nItSubject(iSubject) = nItSubject(iSubject) + 1;
        end
    end
    
    % noise precision update
    noiseInvScaleBkp = posterior.noiseInvScale;
    meanNoisePrecisionBkp = posterior.meanNoisePrecision;
    
    posterior.noiseInvScale = bsxfun(@plus,... %%% b_nr
                               posterior.modifiedSumSqrErr/2,...
                               priors.noiseInvScale);
    posterior.meanNoisePrecision = ...
        posterior.noiseShape./posterior.noiseInvScale; %%% bar(lambda)_nr

    % check free energy
    feBkp = feCurrent;
    feCurrent = tapas_huge_nfe(counts,priors,posterior);    
    if feCurrent < feBkp 
        
        posterior.noiseInvScale = noiseInvScaleBkp;
        posterior.meanNoisePrecision = meanNoisePrecisionBkp;
        feCurrent = feBkp; %#ok<NASGU>
    
    end
    
    
    
    
    


%--------------------
    % save complete history of parameters and important auxiliary variables
    if bKeepTrace
        histClustersMean{iIteration} = posterior.clustersMean;
        histClustersSigma{iIteration} = posterior.clustersSigma;
        histClustersTau{iIteration} = posterior.clustersTau;
        histClustersDeg{iIteration} = posterior.clustersDeg;

        histDcmMean{iIteration} = posterior.dcmMean;
        histDcmSigma{iIteration} = posterior.dcmSigma;

        histNoiseInvScale{iIteration} = posterior.noiseInvScale;

        histposterior.softAssign{iIteration} = posterior.softAssign;
        histPartialDcmMean{iIteration} = partialDcmMean;
        histPartialDcmPrec{iIteration} = partialDcmPrec;
        if bKeepResp
            histRespError{iIteration} = respError;
            histRespJacobian{iIteration} = respJacobian;
        end

        histFreeEnergy{iIteration} = freeEnergy;
        histFeParts{iIteration} = feParts;
        timeSinceStart{iIteration} = toc;
    end
    


%-------------------
    
    [feCurrent,feAux] = tapas_huge_nfe(counts,priors,posterior); 
    % check stopping condition
    dF = feCurrent - freeEnergy;
    freeEnergy = feCurrent;
    histFe(iIteration) = freeEnergy;
    if bVerbose
        display(['iteration ' num2str(iIteration) ...
                 ', dF: ' num2str(dF)]);
    end

    if itSatDcm > nIterations
        if dF < schedule.dfDcm
            itSatDcm = iIteration;
        end
    elseif itSatClusters > nIterations
        if dF < schedule.dfClusters
            itSatClusters = iIteration;
            if bCovFactor
                [priors, posterior] = fnc_reset_cov(priors,posterior,...
                    DcmResults,DcmInfo,nDcmParamsInfCon,nDcmParamsInfAll);
            end
        end
    else
        if (iIteration>=itSatClusters+schedule.itReturn) && (dF < epsEnergy)
            break;
        end
    end


end



%------------------------ End: Main loop ------------------------------
%% save results
DcmResults.posterior = posterior;

% save other info: number of actual iterations etc
DcmResults.freeEnergy = freeEnergy;
DcmResults.nIterationsActual = iIteration;
DcmResults.nIterationsSubject = nItSubject;
DcmResults.inversionScheme = 'VB';

% save prediction and residual
predictedResponse = cell(nSubjects,1);
for iSubject = 1:nSubjects
    predictedResponse{iSubject} = ...
        fnGenerateResponse(posterior.dcmMean(iSubject,:).',...
                                                 dcmParametersDefault,...
                                                 idxParamsInf,...
                                                 idxSelfCon,...
                                                 DcmInfo, iSubject);
end
DcmResults.predictedResponse = predictedResponse;
DcmResults.residuals = respError;

DcmResults.histFe = histFe(1:iIteration);
DcmResults.itSatDcm = itSatDcm;
DcmResults.itSatClusters = itSatClusters;



% save history of important variables
if bKeepTrace
    % note: 'DcmResults.debug.trace' will be an array of structs not a 
    %       struct of arrays
    DcmResults.debug.trace = struct(...
        'clustersMean',histClustersMean(1:iIteration), ...
        'clustersSigma',histClustersSigma(1:iIteration), ...
        'clustersTau',histClustersTau(1:iIteration), ...
        'clustersDeg',histClustersDeg(1:iIteration), ...
        'dcmMean',histDcmMean(1:iIteration), ...
        'dcmSigma',histDcmSigma(1:iIteration), ...
        'noiseInvScale',histNoiseInvScale(1:iIteration), ...
        'posterior.softAssign',histposterior.softAssign(1:iIteration), ...
        'partialDcmMean',histPartialDcmMean(1:iIteration), ...
        'partialDcmPrec',histPartialDcmPrec(1:iIteration), ...
        'freeEnergy',histFreeEnergy(1:iIteration), ...
        'feParts',histFeParts(1:iIteration), ...
        'timeSinceStart',timeSinceStart(1:iIteration));
	DcmResults.debug.idxDcmParamsInfer = idxParamsInf;
    DcmResults.debug.idxSelfCon = idxSelfCon;
    DcmResults.debug.nParameters = nParameters;
    
    if bKeepResp
        DcmResults.debug.resp = struct(...
            'responseError',histRespError(1:iIteration), ...
            'responseJacobian',histRespJacobian(1:iIteration));
    end
end

end




function [ bError ] = fnc_check_response( response, jacobian )
% function [ bError ] = dcm_fmri_check_response( response, jacobian )
% Checks for NaNs and Infs in DCM response and jacobian matrix.

bError = any(isnan(response(:))) || any(isinf(response(:))) || ...
         any(isnan(jacobian(:))) || any(isnan(jacobian(:)));

end


function [ normWeights ] = fnc_exp_norm(logWeights)
% exponentiate and normalize log weights
% subtract max of each row for numerical stability

normWeights = exp(bsxfun(@minus,logWeights,max(logWeights,[],2)));
normWeights = bsxfun(@rdivide,normWeights,sum(normWeights,2));

end


function [priors, posterior] = fnc_reset_cov(priors,posterior,DcmResults,DcmInfo,nDcmParamsInfCon,nDcmParamsInfAll)

    priors.clustersSigma = DcmResults.priors.clustersSigma;
    posterior.clustersSigma = repmat(priors.clustersSigma,...
        [1,1,DcmResults.maxClusters]);
    posterior.dcmSigma = ...
        priors.clustersSigma/(priors.clustersDeg-DcmInfo.nConnections-1);
    posterior.dcmSigma(nDcmParamsInfCon+1:nDcmParamsInfAll,...
                 nDcmParamsInfCon+1:nDcmParamsInfAll) = priors.hemSigma;
    posterior.dcmSigma = repmat(posterior.dcmSigma,[1,1,DcmInfo.nSubjects]);
    for iSubject = 1:DcmInfo.nSubjects
        posterior.logDetPostDcmSigma(iSubject) = ...
            tapas_util_logdet(posterior.dcmSigma(:,:,iSubject));
    end

end


function [posterior] = fnc_set_cluster_cov(priors,posterior,kmeansIdx,nAssign)

    [nClusters,dimClusters] = size(posterior.clustersMean);
    nSubjects = size(posterior.dcmMean,1);
    
    posterior.alpha = priors.alpha + nAssign;
    posterior.clustersTau = priors.clustersTau + nAssign;
    posterior.clustersDeg = priors.clustersDeg + nAssign;
        
    for iCluster = 1:nClusters

        S = zeros(size(priors.clustersSigma));
        for iSubject = 1:nSubjects
            if kmeansIdx(iSubject) == iCluster
                n = posterior.dcmMean(iSubject,1:dimClusters) - ...
                    posterior.clustersMean(iCluster,:);
                S = S + n'*n;
            end
        end

        m = posterior.clustersMean(iCluster,:) - priors.clustersMean;
        posterior.clustersSigma(:,:,iCluster) = priors.clustersSigma + ...
            S + nAssign(iCluster)*priors.clustersTau/...
            (nAssign(iCluster) + priors.clustersTau)*(m'*m);

    end

end