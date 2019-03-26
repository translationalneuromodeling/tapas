function fh = plot_gradient(G, t)
% plots gradient time-course in 3 linked sub-plots, plus 2-norm of G
%
%   output = plot_gradient(input)
%
% IN
%   G   [nSamples, 3] matrix of [G_x, G_y, G_z] vectors
%   t   [nSamples, 1] time vector corresponding to G
% OUT
%   fh  figure handle of created plot
%
% EXAMPLE
%   plot_gradient
%
%   See also

% Author: Lars Kasper
% Created: 2015-01-10
% Copyright (C) 2015 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


stringTitle = 'Sync: Gradient subplots';


nSamples = size(G,1);

if nargin < 2
    t = 1:nSamples;
end

colorOrder = get(0, 'DefaultAxesColorOrder');


fh = tapas_physio_get_default_fig_params();
set(gcf, 'Name', stringTitle);

G(:,4) = sqrt(sum(G.*G,2));

stringSubplotArray = {
    'G_x'
    'G_y'
    'G_z'
    '|G|^2'
    };

for m = 1:4
    currLabel = stringSubplotArray{m};
    hs(m) = subplot(4,1,m);
    plot(t, G(:,4), '--k');
    hold all; 
    plot(t,G(:,m), 'color', colorOrder(m, :));
    title(currLabel);
    xlabel('t');
    ylabel(currLabel);
end

linkaxes(hs, 'x');
tapas_physio_suptitle(stringTitle);