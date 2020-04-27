function fh = tapas_physio_plot_retroicor_regressors(R, order, ...
    hasCardiacData, hasRespData, verbose)
% Plots RETROICOR regressors split into cardiac/resp and interaction
%
%   fh = tapas_physio_plot_retroicor_regressors(R, order)
%
% IN
%   R           [nVolumes, nRegressors] physiological nuisance regressor
%               matrix
%   order       physio.model.retroicor.order (i.e. holding parameters
%               .c, .r and .cr)
% OUT
%   fh          figure handle
%
% EXAMPLE
%   tapas_physio_plot_retroicor_regressors
%
%   See also

% Author: Lars Kasper
% Created: 2015-08-03
% Copyright (C) 2015 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% License (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


if nargin < 3
    hasCardiacData = 1;
end

if nargin < 4
    hasRespData = 1;
end


if hasCardiacData
    cardiac_sess    = R(:,1:(2*order.c));
else
    cardiac_sess = [];
end


if hasRespData
    if hasCardiacData
        respire_sess    = R(:,(2*order.c) + (1:(2*order.r)));
    else
        respire_sess    = R(:, 1:(2*order.r));
    end
else
    respire_sess = [];
end


if hasCardiacData && hasRespData
    mult_sess       = R(:,2*(order.c+order.r) + (1:4*order.cr));
else
    mult_sess = [];
end


if nargin == 5
    % If verbose is passed as argument (from updated tapas_physio_review):
    fh = tapas_physio_get_default_fig_params(verbose);
else
    % Backwards compatibility:
    fh = tapas_physio_get_default_fig_params();
end

set(gcf,'Name','Model: RETROICOR timecourse physiological regressors');

orders = {order.c, order.r, order.cr};
yData = {cardiac_sess, respire_sess, mult_sess};
titles = {
    'RETROICOR cardiac regressors (cos = full, sin = dashed), vertical shift of orders for visibility'
    'RETROICOR respiratory regressors (cos = full, sin = dashed), vertical shift of orders for visibility'
    'RETROICOR multiplicative cardiac x respiratory regressors, vertical shift for visibility'
    };
funColors = {@autumn, @winter, @summer};
nPerOrder = {2 2 4};
lineStyles = {'-', '-.', ':', '--'};

Nsubs = (order.cr>0) + (order.r>0) + (order.c>0);

iSub = 0;
for s = 1:3
    if orders{s}
        iSub = iSub+1;
        ax(iSub) = subplot(Nsubs, 1, iSub);
        y = yData{s};
        
        % color map for line plots
        colors = colormap(funColors{s}(orders{s}+1));
        colors(end,:) = [];
        set(gcf, 'DefaultAxesColorOrder', colors);
        
        
        plotY = [y + ...
            repmat(-1+2*ceil((1:size(y,2))/nPerOrder{s}),length(y),1)];
        
        for iPerOrder = 1:nPerOrder{s}
            plot(plotY(:,iPerOrder:nPerOrder{s}:end), ...
                'LineStyle', lineStyles{iPerOrder}); hold on;
            xlabel('scan volumes');
            title(titles{s});
        end
    end
end



if ~(isempty(cardiac_sess) || isempty(respire_sess)), linkaxes(ax,'x'); end
