%% [  ] = tapas_huge_plot( DCM, DcmResults )
%
% Generate simple graphical summary of inversion result.
%
% INPUT:
%   DCM        - cell array of DCM in SPM format
%   DcmResults - struct used for storing the results from VB. Output of
%                tapas_huge_invert.m
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
function [  ] = tapas_huge_plot( DCM, DcmResults )
%% clustering
if ~isfield(DCM,'listBoldResponse')
    try
        DcmInfo = tapas_huge_import_spm(DCM);
    catch err
        disp('tapas_huge_plot: Unsupported format.');
        disp('Use cell array of DCM in SPM format as first input.');
        rethrow(err);
    end
else
    DcmInfo = DCM;
end


figure
if DcmResults.maxClusters > 1
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
    % plot posterior cluster mean and 95% marginal credible intervals
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
    % plot 25 samples from posterior over (noise free) BOLD response 
    subplot(2,1,2)
    hold on
    n = 1; % for first subject
    postMean = DcmResults.posterior.dcmMean(n,:);
    postStd = chol(DcmResults.posterior.dcmSigma(:,:,n));
    for iSmp = 1:25
        % draw a sample from posterior over DCM parameters
        postSmp = postMean + randn(size(postMean))*postStd;
        tmp = zeros(1,DcmInfo.nParameters);
        tmp(DcmInfo.connectionIndicator) = postSmp(1:DcmInfo.nConnections);
        tmp(end-3*DcmInfo.nStates+1:end-DcmInfo.nStates+1) = ...
            postSmp(DcmInfo.nConnections+1:end);
        % predict BOLD response for current sample
        pred = tapas_huge_bold(tmp,DcmInfo,n);
        plot(pred(:),'b')
    end
    plot(DcmInfo.listBoldResponse{n}(:),'k')
    % plot ground truth if available
    try
        tmp = DcmInfo.listParameters{n};
        pred = tapas_huge_bold([tmp{1}(:);tmp{2}(:);tmp{3}(:);tmp{4}(:);...
            tmp{5}(:);]',DcmInfo,n);
        plot(pred(:),'r')
    catch
        % omit ground truth
    end
    title('black: measurement - blue: posterior samples');
    ylabel('BOLD')
    xlabel('sample index')
    
else
    %% empirical Bayes
    subplot(2,1,1)
    hold on
    line([0 DcmInfo.nConnections+1],[0 0],'color','k')
    k = 1;
%     plot posterior cluster mean and 95% marginal credible intervals
    clMean = DcmResults.posterior.clustersMean(k,:);
    clStd = sqrt(diag(DcmResults.posterior.clustersSigma(:,:,k))'/...
        (DcmResults.posterior.clustersTau(k)*...
        (DcmResults.posterior.clustersDeg(k) - DcmInfo.nConnections + 1)));
    s = tinv(1-0.025,DcmResults.posterior.clustersDeg(k));
    errorbar(1:DcmInfo.nConnections,clMean,s*clStd,'d')
    ylabel('\mu_k')
    xlabel('connection index');
    title('Empirical Bayes')
    
    % boxplot of MAP estimates of DCM parameters
    subplot(2,1,2)
    hold on
    line([0 DcmInfo.nConnections+1],[0 0],'color','k')
    boxplot(DcmResults.posterior.dcmMean)
    ylabel('\mu_n')
    xlabel('connection index');
    
end

end

