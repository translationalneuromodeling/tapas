function fh = tapas_physio_plot_results_comparison(actPhysio, expPhysio)
%Compares two physio output structures in terms of their
%processing/modeling results
%
%   fh = tapas_physio_plot_results_comparison(actPhysio, expPhysio)
%
% IN
%
% OUT
%
% EXAMPLE
%   tapas_physio_plot_results_comparison
%
%   See also

% Author:   Lars Kasper
% Created:  2024-12-21
% Copyright (C) 2024 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under
% the terms of the GNU General Public License (GPL), version 3. You can
% redistribute it and/or modify it under the terms of the GPL (either
% version 3 or, at your option, any later version). For further details,
% see the file COPYING or <http://www.gnu.org/licenses/>.

fh = tapas_physio_get_default_fig_params();

traceArray = {'c', 'r', 'acq_codes'};

nTraces = numel(traceArray);

for iTrace = 1:nTraces
    ax(iTrace) = subplot(nTraces,1, iTrace);

    actY = getfield(actPhysio, 'ons_secs', traceArray{iTrace});
    expY = getfield(expPhysio, 'ons_secs', traceArray{iTrace});
    try
        plot(actPhysio.ons_secs.t, actY);
        hold all;
        plot(expPhysio.ons_secs.t, expY,'--')
        title(traceArray{iTrace},'Interpreter', 'none');

    catch % take raw data for time axis instead
        plot(actPhysio.ons_secs.raw.t, actY);
        hold all;
        plot(expPhysio.ons_secs.raw.t, expY,'--')
        title(traceArray{iTrace},'Interpreter', 'none');
    end
end
xlabel('time (seconds)')
legend('actual', 'expected')
linkaxes(ax, 'x');
set(fh, 'Name', 'Plot: Comparison of Results Actual/Expected Physio struct')