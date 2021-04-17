function [ fHdl ] = plot( obj, subjects )
% Plot cluster and subject-level estimation result from HUGE model.
% 
% INPUTS:
%   obj - A tapas_Huge object containing estimation results.
% 
% OPTIONAL INPUTS:
%   subjects - A vector containing indices of subjects for which to plot
%              detailed results.
% 
% OUTPUTS:
%   fHdl - Handle of first figure.
% 
% See also tapas_Huge.ESTIMATE
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


if nargout > 0
    fHdl = figure( );
else
    figure( );
end
if nargin < 2
    subjects = [];
end

tickLabels = obj.parse_labels( obj.dcm, obj.labels, obj.idx );

%% assignments/boxplot
% figure;
if obj.K > 1
    % subject assignment
    bar(obj.posterior.q_nk,'stacked');
    axis([0 obj.N+1 0 1])
    title('assignments')
    ylabel('q_{nk}')
    xlabel('subject index')
    title('Assignment');
else
    % boxplot of MAP estimates of DCM parameters
    hold on
    line([0 obj.idx.P_c + obj.idx.P_h], [0 0], 'color', 'k')
    boxplot(obj.posterior.mu_n)
    ylabel('\mu_n')
    title('Empirical Bayes');
    set(gca,'XTick',1:obj.idx.P_c + obj.idx.P_h, 'XTickLabelRotation', 60, ...
        'XTickLabel', tickLabels, 'TickLabelInterpreter', 'none');
end

%% cluster
figure
hold on
% plot posterior cluster mean and 95% marginal credible intervals
legends = cell(obj.K,1);
for k = 1:obj.K
    xOffset = ((k-1)/max(1,(obj.K-1)) - .5)/4;
    switch obj.posterior.method
        case 'VB'
            clMean = obj.posterior.m(k,:);
            clStd = sqrt(diag(obj.posterior.S(:,:,k))'/...
                (obj.posterior.tau(k)*...
                (obj.posterior.nu(k) - obj.idx.P_c + 1)));
            s = tinv(1-0.025, obj.posterior.nu(k));
            errorbar((1:obj.idx.P_c) + xOffset, clMean, s*clStd, 'd');
        case 'MH'
            clMean = obj.posterior.mean.mu(k,:);
            [~,i1] = min(abs(obj.posterior.quantile.levels - .025));
            [~,i2] = min(abs(obj.posterior.quantile.levels - .975));
            neg = clMean - obj.posterior.quantile.mu(k,:,i1);
            pos = obj.posterior.quantile.mu(k,:,i2) - clMean;
            errorbar((1:obj.idx.P_c) + xOffset, clMean, neg, pos, 'd');
    end
    legends{k} = ['cluster ' num2str(k)];
end
line([0 obj.idx.P_c+1], [0 0], 'color', 'k')
xlim([0 obj.idx.P_c+1])
ylabel('\mu_k')
set(gca,'XTick',1:obj.idx.P_c, 'XTickLabelRotation', 60, ...
    'XTickLabel', tickLabels(1:obj.idx.P_c), 'TickLabelInterpreter', 'none');
legend(legends)
title('Clusters')

%% DCM
% plot 25 samples from posterior over (noise free) BOLD response 
nSmp = 25;
for n = subjects(:)'
    figure
    hold on
    % draw samples from posterior over DCM parameters
    switch obj.posterior.method
        case 'VB'
            postMean = obj.posterior.mu_n(n,:);
            postStd = chol(obj.posterior.Sigma_n(:,:,n));
            postSmp = randn(nSmp,obj.idx.P_c + obj.idx.P_h);
            postSmp = bsxfun(@plus, postSmp*postStd, postMean);
        case 'MH'
            nTrace = length(obj.trace.smp);
            nSmp = min(nSmp, nTrace);
            idx = randsample(nTrace, nSmp);
            tmp = [reshape([obj.trace.smp(idx).theta_c], obj.N, obj.idx.P_c, []), ...
                reshape([obj.trace.smp(idx).theta_h], obj.N, obj.idx.P_h, [])];
            postSmp = permute(tmp(n,:,:), [3 2 1]);
    end
    legends = {'measured'};
    plot(obj.data(n).bold(:),'k')
    % plot ground truth if available
    if ~isempty(obj.model)
        [ ~, epsilon ] = obj.gen_bold( n, obj.model.theta(n,:) );
        plot(obj.data(n).bold(:) - epsilon(:),'r')
        legends = [legends, {'ground truth'}]; %#ok<AGROW>
    end
    legends = [legends ,{'posterior samples'}]; %#ok<AGROW>
    for iSmp = 1:nSmp
        [ ~, epsilon ] = obj.gen_bold( n, postSmp(iSmp,:) );
        plot(obj.data(n).bold(:) - epsilon(:),'b')
    end
    legend(legends);
    title(['Subject ' num2str(n)]);
    ylabel('BOLD')
    xlabel('sample index')
end

end

