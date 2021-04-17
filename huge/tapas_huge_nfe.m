%% [freeEnergyFull, feAux] = tapas_huge_nfe( counts, priors, posterior, feAux, listSubjects )
% 
% Calculates or updates negative free energy (NFE) for HUGE.
% 
% INPUT:
%       counts       - number of parameters and subjects
%       priors       - parameters of prior distribution
%       posterior    - parameters of approximate posterior distribution.
%       feAux        - struct containing intermediate values of the NFE.
%       listSubjects - indices of subjects for which to update the NFE.
% 
% OUTPUT:
%       freeEnergyFull - value of the negative free energy.
%       feAux          - struct containing intermediate values of the NFE.
% 
% REFERENCE:
%
% Yao Y, Raman SS, Schiek M, Leff A, Frässle S, Stephan KE (2018).
% Variational Bayesian Inversion for Hierarchical Unsupervised Generative
% Embedding (HUGE). NeuroImage, 179: 604-619
% 
% https://doi.org/10.1016/j.neuroimage.2018.06.073
%

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
function [freeEnergyFull, feAux] = tapas_huge_nfe(counts,priors,posterior,feAux,listSubjects)
%
fepIdx = 0;
freeEnergyFullPart = zeros(100,1);


nParameters = counts.nParameters;
nDcmParamsInfCon = nParameters(2,1); %%% d_c
nDcmParamsInfHem = nParameters(2,2); %%% d_h
nDcmParamsInfAll = nParameters(2,3); %%% d

if nargin < 4
    feAux.line2 = zeros(counts.nSubjects,counts.nClusters,2);
    feAux.line4 = zeros(counts.nSubjects,2);
end
if nargin < 5
    listSubjects = 1:counts.nSubjects;
end



%% line 1
fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    -sum(sum(bsxfun(@times,posterior.softAssign,...
        posterior.logDetClustersSigma.')/2));

fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    -sum(posterior.softAssign(:))*nDcmParamsInfCon*log(pi)/2;

fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    +sum(sum(...
        bsxfun(@times,...
            posterior.softAssign,...
            sum(psi(bsxfun(@plus,...
                posterior.clustersDeg,...
                1-(1:nDcmParamsInfCon))/2),2).'...
            )...
        ))/2;

fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    -sum(sum(bsxfun(@times,...
        posterior.softAssign,...
        nDcmParamsInfCon./posterior.clustersTau.')...
        ))/2;



%% line 2
fepIdx = fepIdx + 1;
for iSub = listSubjects
    for iCluster = 1:counts.nClusters
        feAux.line2(iSub,iCluster,1) = ...
            -posterior.clustersDeg(iCluster)*...
                    posterior.softAssign(iSub,iCluster)*(...
                trace(...
                    posterior.clustersSigma(:,:,iCluster)\...
                    posterior.dcmSigma(1:nDcmParamsInfCon,...
                        1:nDcmParamsInfCon,...
                        iSub...
                        )...
                    )...
                )/2;
    end
end
freeEnergyFullPart(fepIdx) = sum(sum(feAux.line2(:,:,1)));
%%%

fepIdx = fepIdx + 1;
for iSub = listSubjects
    for iCluster = 1:counts.nClusters
        feAux.line2(iSub,iCluster,2) = ...
            -posterior.clustersDeg(iCluster)*...
                    posterior.softAssign(iSub,iCluster)*(...
                (posterior.dcmMean(iSub,1:nDcmParamsInfCon)...
                    -posterior.clustersMean(iCluster,:))*...
                (posterior.clustersSigma(:,:,iCluster)\...
                    (posterior.dcmMean(iSub,1:nDcmParamsInfCon)...
                        -posterior.clustersMean(iCluster,:)).' ...
                    )...
                )/2;
    end
end
freeEnergyFullPart(fepIdx) = sum(sum(feAux.line2(:,:,2)));
%%%


%% line 3
fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    -sum(counts.nMeasurements)*log(2*pi)/2;

fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    +sum(sum(...
        bsxfun(@times,...
            counts.nMeasurementsPerState,...
            psi(posterior.noiseShape) - log(posterior.noiseInvScale))...
        ))/2;    

fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    -posterior.meanNoisePrecision(:).'*posterior.modifiedSumSqrErr(:)/2;


%% line 4
fepIdx = fepIdx + 1;
for iSub = listSubjects
    feAux.line4(iSub,1) = ...
        -trace(priors.hemSigma\posterior.dcmSigma(nDcmParamsInfCon+1:end,...
                            nDcmParamsInfCon+1:end,...
                            iSub...
                            ) ...
                )/2;
end
freeEnergyFullPart(fepIdx) = sum(feAux.line4(:,1));


fepIdx = fepIdx + 1;
for iSub = listSubjects
    feAux.line4(iSub,2) = ...
        -(posterior.dcmMean(iSub,nDcmParamsInfCon+1:end)-priors.hemMean)*...
            (priors.hemSigma\...
                (posterior.dcmMean(iSub,nDcmParamsInfCon+1:end)...
                    -priors.hemMean).'...
                )/2;
end
freeEnergyFullPart(fepIdx) = sum(feAux.line4(:,2));


%% line 5
fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    -counts.nSubjects*tapas_huge_logdet(priors.hemSigma)/2;

fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    -counts.nSubjects*nDcmParamsInfHem*log(2*pi)/2;


%% line 6
fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    -counts.nSubjects*sum(gammaln(priors.noiseShape));

fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    +counts.nSubjects*sum(priors.noiseShape.*log(priors.noiseInvScale));


%% line 7
fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    +sum(sum(bsxfun(@times, ...
        priors.noiseShape-1, ...
        (psi(posterior.noiseShape)-log(posterior.noiseInvScale)) ...
        )));

fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    -sum(sum(bsxfun(@times, ...
        priors.noiseInvScale, ...
        posterior.noiseShape./posterior.noiseInvScale ...
        )));


%% line 8
fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    -nDcmParamsInfCon*sum(priors.clustersTau./posterior.clustersTau)/2;



fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = 0;
for iCluster = 1:counts.nClusters
    freeEnergyFullPart(fepIdx) = freeEnergyFullPart(fepIdx) + ...
        -priors.clustersTau*posterior.clustersDeg(iCluster)/2 ...
            *(posterior.clustersMean(iCluster,:)-priors.clustersMean) ...
            *(posterior.clustersSigma(:,:,iCluster)\...
                (posterior.clustersMean(iCluster,:)-priors.clustersMean).');
end
%%%


%% line 9
fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = 0;
for iCluster = 1:counts.nClusters
    freeEnergyFullPart(fepIdx) = freeEnergyFullPart(fepIdx) + ...
        -posterior.clustersDeg(iCluster)/2 ...
            *trace(posterior.clustersSigma(:,:,iCluster)\...
                priors.clustersSigma);
end
%%%


fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    -(priors.clustersDeg+nDcmParamsInfCon+2)/2*(...
        sum(posterior.logDetClustersSigma) ...
        -sum(sum(psi(bsxfun(@plus,...
            posterior.clustersDeg,...
            1-(1:nDcmParamsInfCon)...
            )/2))) ...
        );



%% line 10
fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    -counts.nClusters*nDcmParamsInfCon*(nDcmParamsInfCon-1)/4*log(pi);
fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    -counts.nClusters*sum(gammaln((priors.clustersDeg+1-...
        (1:nDcmParamsInfCon))/2));


fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    +counts.nClusters*priors.clustersDeg*tapas_huge_logdet(...
        priors.clustersSigma)/2;

fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    +counts.nClusters*nDcmParamsInfCon*(nDcmParamsInfCon+2)/2*log(2);

fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    -counts.nClusters*nDcmParamsInfCon*log(2*pi)/2;
fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    +counts.nClusters*nDcmParamsInfCon*log(priors.clustersTau)/2;



%% line 11
fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    +sum(sum(bsxfun(@times,...
        posterior.softAssign,...
        psi(posterior.alpha.')-psi(sum(posterior.alpha))...
        )));


%% line 12
fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    +gammaln(sum(priors.alpha))-sum(gammaln(priors.alpha));


%% line 13
fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    +(priors.alpha.'-1)*(psi(posterior.alpha)-psi(sum(posterior.alpha)));


%% line 14
fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    -sum(posterior.softAssign(:).*log(posterior.softAssign(:)+realmin));


%% line 15
fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    +sum(posterior.logDetPostDcmSigma)/2;
%%% neg dF

fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    +counts.nSubjects*nDcmParamsInfAll/2;


fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    +counts.nSubjects*nDcmParamsInfAll*log(2*pi)/2;



%% line 16
fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    +sum(gammaln(posterior.noiseShape(:)));


fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    -posterior.noiseShape(:).'*log(posterior.noiseInvScale(:));


%% line 17
fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    -(posterior.noiseShape(:).'-1)*(psi(posterior.noiseShape(:))-...
        log(posterior.noiseInvScale(:)));
    

fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    +sum(posterior.noiseShape(:));


%% line 18
fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    +counts.nClusters*nDcmParamsInfCon/2;


fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    +nDcmParamsInfCon/2*sum(posterior.clustersDeg);


fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    -counts.nClusters*nDcmParamsInfCon*(nDcmParamsInfCon+2)/2*log(2);


fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    +counts.nClusters*nDcmParamsInfCon/2*log(2*pi);
fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    +nDcmParamsInfCon/2*sum(log(1./posterior.clustersTau));


%% line 19
fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    +(posterior.clustersDeg.'+nDcmParamsInfCon+2)/2*(...
        posterior.logDetClustersSigma...
        -sum(psi(bsxfun(@plus,...
            posterior.clustersDeg,...
            1-(1:nDcmParamsInfCon)...
             )/2),2)...
        );


%% line 20
fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    +counts.nClusters*nDcmParamsInfCon*(nDcmParamsInfCon-1)/4*log(pi);
fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) =  ...
    +sum(sum(gammaln(bsxfun(@plus,...
        posterior.clustersDeg,...
        1-(1:nDcmParamsInfCon)...
        )/2)));


fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    -posterior.clustersDeg.'*posterior.logDetClustersSigma/2;
    


%% line 21
fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    +sum(gammaln(posterior.alpha));

fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    -gammaln(sum(posterior.alpha));

%% line 22
fepIdx = fepIdx + 1;
freeEnergyFullPart(fepIdx) = ...
    -(posterior.alpha.'-1)*(psi(posterior.alpha)-psi(sum(posterior.alpha)));
    


%% sum
freeEnergyFull = sum(freeEnergyFullPart);


%% aux
freeEnergyFullPart(61) = freeEnergyFullPart(2)...
                        +freeEnergyFullPart(13)...
                        +freeEnergyFullPart(34);



