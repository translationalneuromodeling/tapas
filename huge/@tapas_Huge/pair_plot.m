function [ fHdls ] = pair_plot( obj, clusters, subjects, pdx, maxSmp )
% Generate pair plots for cluster and subject-level parameters from MCMC
% trace.
% 
% INPUTS:
%   obj - A tapas_Huge object containing estimation results.
% 
% OPTIONAL INPUTS:
%   clusters - A vector containing indices of clusters for which to make
%              pair plots. If empty, plot all clusters.
%   subjects - A vector containing indices of subjects for which to make
%              pair plots. If empty, plot all subjects.
%   pdx      - A vector containing indices of parameters to plot. If empty,
%              plot all parameters.
%   maxSmp   - Maximum number of samples to use for plots (default: 1000).
% 
% OUTPUTS:
%   fHdls - Cell array of figure handles.
%
% EXAMPLES:
%   [fHdls] = PAIRS_PLOT(obj, [], 1, 1:3)    Generate pairs plot for the
%       first three parameters for all clusters and the first subject.
% 
% See also tapas_Huge.PLOT
% 

% Author: Yu Yao (yao@biomed.ee.ethz.ch)
% Copyright (C) 2020 Translational Neuromodeling Unit
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

assert(strcmpi(obj.options.nvp.method,'mh') && ~isempty(obj.trace), ...
               'TAPAS:HUGE:plot',...
               'Pair plot is available only for MH inversion method.');

if nargin<2 || isempty(clusters)
    clusters = 1:obj.K;
end
if nargin<3 || isempty(subjects)
    subjects = 1:obj.N;
end
if nargin<4 || isempty(pdx)
    pdx = 1:obj.idx.P_c+obj.idx.P_h;
end
if nargin < 5
    maxSmp = 1e3;
end


fHdls = cell(0,1);
nSmp = length(obj.trace.smp);
if maxSmp > nSmp
    rdx = randsample(nSmp,maxSmp);
else
    rdx = 1:nSmp;
end

tickLabels = obj.parse_labels( obj.dcm, obj.labels, obj.idx );

%% cluster parameter
mdx = pdx(pdx<=obj.idx.P_c);
X = reshape([obj.trace.smp(rdx).mu],[size(obj.trace.smp(1).mu),numel(rdx)]);
for k = clusters
    fHdls{end+1,1} = plot_pairs(squeeze(X(k,mdx,:))',tickLabels(mdx));
    title(fHdls{end}.Children(end), ['cluster ' num2str(k)]);
end

%% subject parameter
X = cat(2, reshape([obj.trace.smp(rdx).theta_c], ...
    [size(obj.trace.smp(1).theta_c),numel(rdx)]), ...
    reshape([obj.trace.smp(rdx).theta_h], ...
    [size(obj.trace.smp(1).theta_h),numel(rdx)]));
for n = subjects
    fHdls{end+1,1} = plot_pairs(squeeze(X(n,pdx,:))',tickLabels(pdx));
    title(fHdls{end}.Children(end), ['subject ' num2str(n)]);
end

end


function [ fHdl ] = plot_pairs(X, labels)
fHdl = figure();
P = size(X,2);
ax = cell(P);
for p1 = 1:P
    ax{p1,p1} = subplot(P,P,(p1-1)*P+p1);
    histogram(ax{p1,p1}, X(:,p1), 'normalization', 'pdf');
    for p2 = p1+1:P
        ax{p1,p2} = subplot(P,P,(p1-1)*P+p2);
        tmp = scatter(ax{p1,p2}, X(:,p2), X(:,p1), 5, 'filled');        
        ax{p2,p1} = subplot(P,P,(p2-1)*P+p1);
        copyobj(tmp,ax{p2,p1});
    end
end
for p = 1:P
    ylabel(ax{p,1},labels{p});
    xlabel(ax{P,p},labels{p});
end
end
