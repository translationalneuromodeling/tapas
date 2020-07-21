function fh = tapas_physio_plot_raw_physdata_siemens_hcp(t, c, r, acq_codes, ...
    stringTitle)
% plots cardiac data as extracted from Human Connectome Phys log file
% (=preprocessed Siemens log file)
%
%   fh = tapas_physio_plot_raw_physdata_siemens_hcp(t, c, r, acq_codes, ...
%           stringTitle)
%
% IN
%
% OUT
%
% EXAMPLE
%   tapas_physio_plot_raw_physdata_siemens_hcp
%
%   See also tapas_physio_read_physlogfiles_siemens_hcp

% Author: Lars Kasper
% Created: 2018-01-23
% Copyright (C) 2018 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% License (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
if nargin < 5
    stringTitle = 'Read-In: Raw Human Connectome Project physlog data (preprocessed Siemens Data)';
end

volpulse_on = find(acq_codes == 8);
volpulse_off = find(acq_codes == 16);

% check what needs to be plotted
doPlotItem = ~[isempty(volpulse_on) isempty(volpulse_off) isempty(c) isempty(r)];

fh = tapas_physio_get_default_fig_params();
set(gcf, 'Name', stringTitle);

ampl = max(max(c), max(r));

if doPlotItem(1)
    stem(t(volpulse_on), ampl*ones(size(volpulse_on)), 'c'); hold all;
end

if doPlotItem(2)
    stem(t(volpulse_off), ampl*ones(size(volpulse_off)), 'c--'); hold all;
end

if doPlotItem(3)
    plot(t, c, 'r'); hold all;
end

if doPlotItem(4)
    plot(t, r, 'g'); hold all;
end

stringLegend = {'volpulse on', 'volpulse off', 'cardiac trace', 'respiratory trace'};

legend(stringLegend(doPlotItem));
title(stringTitle);
xlabel('t (seconds)');
