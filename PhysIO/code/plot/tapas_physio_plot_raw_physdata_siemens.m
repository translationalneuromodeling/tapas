function fh = tapas_physio_plot_raw_physdata_siemens(dataCardiac)
% plots cardiac data as extracted from Siemens log file
%
%   output = tapas_physio_plot_raw_physdata_siemens(input)
%
% IN
%
% OUT
%
% EXAMPLE
%   tapas_physio_plot_raw_physdata_siemens
%
%   See also tapas_physio_read_physlogfiles_siemens
%   See also tapas_physio_siemens_table2cardiac

% Author: Lars Kasper
% Created: 2016-02-29
% Copyright (C) 2016 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% License (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

tapas_physio_strip_fields(dataCardiac);

stringTitle = 'Read-In: Raw Siemens physlog data';
fh = tapas_physio_get_default_fig_params();
set(gcf, 'Name', stringTitle);
stem(cpulse_on, ampl*ones(size(cpulse_on)), 'g'); hold all;
stem(cpulse_off, ampl*ones(size(cpulse_off)), 'r');
stem(t(stopSample), ampl , 'm');
plot(t, channel_1);
plot(t, channel_AVF);
plot(t, meanChannel);

stringLegend = { ...
    'cpulse on', 'cpulse off', 'assumed last sample of last scan volume', ...
    'channel_1', 'channel_{AVF}', 'mean of channels'};

if ~isempty(recording_on)
    stem(recording_on, ampl*ones(size(recording_on)), 'k');
    stringLegend{end+1} = 'phys recording on';
end

if ~isempty(recording_off)
    stem(recording_off, ampl*ones(size(recording_off)), 'k');
    stringLegend{end+1} = 'phys recording off';
end
legend(stringLegend);
title(stringTitle);
xlabel('t (seconds)');
