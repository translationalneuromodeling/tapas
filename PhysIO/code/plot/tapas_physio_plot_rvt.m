function fh = tapas_physio_plot_rvt(ons_secs, sqpar)
% Plots respiratory volume per time at sampling points, and filtered time
% series built from
%
%   fh = tapas_physio_plot_rvt(ons_secs, sqpar)
%
% IN
%
% OUT
%   fh figure handle
% EXAMPLE
%   tapas_physio_plot_rvt
%
%   See also

% Author:   Lars Kasper
% Created:  2021-01-15
% Copyright (C) 2021 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under
% the terms of the GNU General Public License (GPL), version 3. You can
% redistribute it and/or modify it under the terms of the GPL (either
% version 3 or, at your option, any later version). For further details,
% see the file COPYING or <http://www.gnu.org/licenses/>.
fh = tapas_physio_get_default_fig_params();
set(gcf, 'Name', 'Model: Respiratory Volume per Time (RVT)');

% Calculate RVT
sample_points  = tapas_physio_get_sample_points(ons_secs, sqpar, sqpar.onset_slice);

plot(ons_secs.t, ons_secs.fr, 'Color', [0.5, 1 0.5], 'LineWidth', 1);
hold all;
  
plot(sample_points,ons_secs.rvt, 'g');xlabel('time (seconds)');
title('Respiratory volume per time');
ylabel('a.u.');

legend('filtered respiratory time series', 'RVT (sampled at onset slice time)')