%% [ DcmInfo, DcmResults ] = tapas_huge_demo( )
%
% An example for using the variational Bayes inversion on HUGE for sythetic
% data.
%
% OUTPUT:
%       DcmInfo    - struct containing the synthetic dataset
%       DcmResults - struct containing the results of VB inference
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
function [ DcmInfo, DcmResults ] = tapas_huge_demo( )
%% generate synthetic DCM fMRI dataset
rng(8032,'twister')

optionsGen = struct();
optionsGen.snr = 1; % signal-to-noise-ratio
optionsGen.N_k = [40 20 20]; % number of subjects by cluster
optionsGen.R = 3; % number of regions

% cluster
optionsGen.mu_k.idx = [1,2,3,4,5,6,9,10,27];
optionsGen.mu_k.value = [...
    -0.7, 0.2,-0.1,-0.2 ,-0.6, 0.3,-0.4,0.3, 0.3;...
    -0.7, 0.1, 0.3, 0.1 ,-0.4,-0.1,-0.6,0.6, 0.1;...
    -0.7,-0.2, 0.3, 0.25,-0.4,-0.1,-0.6,0.6,-0.2];
optionsGen.sigma_k = 0.1;

% heamodynamics
optionsGen.mu_h = zeros(1,optionsGen.R*2+1);
optionsGen.sigma_h = 0.01;

% hemodynamic parameters
optionsGen.hemParam.listHem = [0.6400 2 1];
optionsGen.hemParam.scaleC = 16;
optionsGen.hemParam.echoTime = 0.0400;
optionsGen.hemParam.restingVenousVolume = 4;
optionsGen.hemParam.relaxationRateSlope = 25;
optionsGen.hemParam.frequencyOffset = 40.3000;
optionsGen.hemParam.oxygenExtractionFraction = 0.4000;
optionsGen.hemParam.rho = 4.3000;
optionsGen.hemParam.gamma = 0.3200;
optionsGen.hemParam.alphainv = 3.1250;
optionsGen.hemParam.oxygenExtractionFraction2 = 0.3200;

% input
optionsGen.input.u = double([...
    reshape(repmat((1:2^9<2^8)'&(mod(1:2^9,2^5)==0)',1,2^3),2^12,1),...
    reshape(repmat((1:2^10<2^8)',1,2^2),2^12,1)]);
optionsGen.input.u = circshift(optionsGen.input.u,117,1);           
optionsGen.input.trSteps = 16;
optionsGen.input.trSeconds = 2;


DcmInfo = tapas_huge_simulate(optionsGen);




%% set options for inversion
DcmResults = struct;
DcmResults.jacobianStepSize = 1e-3;
% DcmResults.fnJacobian = @dcm_jacobian_fd;
DcmResults.nIterations = 2^10;
DcmResults.epsEnergy = 1e-6;
DcmResults.bKeepTrace = false;
DcmResults.maxClusters = 3;
DcmResults.bVerbose = false;



%% set priors
priors.alpha = 1;
priors.clustersMean = [-.5,1/64/DcmInfo.nStates,1/64/DcmInfo.nStates,...
    1/64/DcmInfo.nStates,-.5,1/64/DcmInfo.nStates,-.5,0,0]; % SPM prior
p_c = length(priors.clustersMean);
priors.clustersTau = 0.1;
priors.clustersDeg = 100;
priors.clustersSigma = 0.01*eye(p_c)*(priors.clustersDeg - p_c - 1);
priors.hemMean = zeros(1,DcmInfo.nStates*2 + 1); % SPM prior
priors.hemSigma = diag(zeros(1,DcmInfo.nStates*2 + 1)+exp(-6)); % SPM prior

priors.noiseInvScale = .025;
priors.noiseShape = 1.28;

DcmResults.priors = priors;


%% set update schedule
schedule.dfDcm = 50;
schedule.dfClusters = 10;
schedule.itAssignment = 1;
schedule.itCluster = 1;
schedule.itReturn = 25;
schedule.itKmeans = 1;

DcmResults.schedule = schedule;


%% set initial values
init.dcmMean = repmat([DcmResults.priors.clustersMean,...
    DcmResults.priors.hemMean],DcmInfo.nSubjects,1);
init.dcmMean = init.dcmMean + randn(size(init.dcmMean))*.1;

init.clustersMean = repmat(...
    DcmResults.priors.clustersMean,DcmResults.maxClusters,1);
init.clustersMean = init.clustersMean + ...
    randn(size(init.clustersMean))*.1;

DcmResults.init = init;


%% invert HUGE
DcmResults.bVerbose = true;

currentTimer = tic;
DcmResults = tapas_huge_inv_vb(DcmInfo,DcmResults);
toc(currentTimer)

disp(['Negative free energy: ' num2str(DcmResults.freeEnergy)])

%% plot result


figure
% subject assignment
subplot(2,2,1)
bar(DcmResults.posterior.softAssign,'stacked');
axis([0 DcmInfo.nSubjects+1 0 1])
title('assignments')
ylabel('q_{nk}')
xlabel('subject index')

% cluster
subplot(2,2,2)
hold on
line([0 DcmInfo.nConnections+1],[0 0],'color','k')
for k = 1:DcmResults.maxClusters
    xOffset = ((k-1)/max(1,(DcmResults.maxClusters-1)) - .5)/2;
    clMean = DcmResults.posterior.clustersMean(k,:);
    clStd = sqrt(diag(DcmResults.posterior.clustersSigma(:,:,k))'/...
        (DcmResults.posterior.clustersTau(k)*...
        (DcmResults.posterior.clustersDeg(k) - DcmInfo.nConnections + 1)));
    s = tinv(1-0.025,DcmResults.posterior.clustersDeg(k));
    errorbar((1:DcmInfo.nConnections) + xOffset,clMean,s*clStd,'d')
end
xlim([0 DcmInfo.nConnections+1])
title('clusters')
ylabel('\mu_k')
xlabel('connection index');

% DCM
subplot(2,1,2)
hold on
n = 1;
postMean = DcmResults.posterior.dcmMean(n,:);
postCovM = DcmResults.posterior.dcmSigma(:,:,n);
for iSmp = 1:25
    postSmp = mvnrnd(postMean,postCovM);
    tmp = zeros(1,DcmInfo.nParameters);
    tmp(DcmInfo.connectionIndicator) = postSmp(1:DcmInfo.nConnections);
    tmp(end-3*DcmInfo.nStates+1:end-2) = ...
        postSmp(DcmInfo.nConnections+1:end);
    pred = tapas_huge_bold(tmp,DcmInfo,n);
    plot(pred(:),'b')
end
plot(DcmInfo.listBoldResponse{n}(:),'k')
tmp = DcmInfo.listParameters{n};
pred = tapas_huge_bold(...
    [tmp{1}(:);tmp{2}(:);tmp{3}(:);tmp{4}(:);tmp{5}(:);]',DcmInfo,n);
plot(pred(:),'r')
title('black: measurement - blue: posterior samples');
ylabel('BOLD')
xlabel('sample index')





end
