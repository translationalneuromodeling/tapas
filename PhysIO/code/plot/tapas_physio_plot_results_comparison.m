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

titleArray = {
    'Plot: Comparison of Results Actual/Expected Physio struct (ons_secs.raw)'
    'Plot: Comparison of Results Actual/Expected Physio struct (ons_secs)'
   };



for iFigure = 1:2

    switch iFigure
        case 1
            stringRaw = 'raw';
        case 2
            stringRaw = {};
    end
    fh(iFigure) = tapas_physio_get_default_fig_params();

    traceArray = {'c', 'r', 'acq_codes'};

    nTraces = numel(traceArray);

    for iTrace = 1:nTraces
        ax(iTrace) = subplot(nTraces,1, iTrace);
        actY = getfield(actPhysio, 'ons_secs', stringRaw, traceArray{iTrace});
        expY = getfield(expPhysio, 'ons_secs', stringRaw, traceArray{iTrace});
        t = getfield(expPhysio, 'ons_secs', stringRaw,'t');

         if ~(isempty(actY) || isempty(expY)) && ...
                 (numel(actY) == numel(t)) && ...
                 (numel(expY) == numel(t))
            plot(t, actY);
            hold all;
            plot(t, expY,'--')
            title(traceArray{iTrace},'Interpreter', 'none');
        end


        % plot cpulse for c-trace
        if strcmpi(traceArray{iTrace}, 'c')
            actYStem = getfield(actPhysio, 'ons_secs', stringRaw, 'cpulse');
            expYStem = getfield(expPhysio, 'ons_secs', stringRaw, 'cpulse');
      
            stem(actYStem, ones(size(actYStem)), 'LineWidth', 2);
            stem(expYStem, ones(size(expYStem)), '--', 'LineWidth', 2);
        end

        xlabel('time (seconds)')
        legend('actual', 'expected')
        linkaxes(ax, 'x');
        set(fh(iFigure), 'Name', titleArray{iFigure})
        if exist('sgtitle')
            sgtitle(titleArray{iFigure}, 'Interpreter', 'none');
        end
    end

end