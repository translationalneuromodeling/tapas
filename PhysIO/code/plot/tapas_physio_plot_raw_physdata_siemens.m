function fh = tapas_physio_plot_raw_physdata_siemens(dataPhysio)
% plots cardiac data as extracted from Siemens log file
%
%   fh = tapas_physio_plot_raw_physdata_siemens(dataCardiac)
%
% IN
%   dataPhysio     output struct from tapas_physio_siemens_table2cardiac
% OUT
%
% EXAMPLE
%   tapas_physio_plot_raw_physdata_siemens
%
%   See also tapas_physio_read_physlogfiles_siemens tapas_physio_siemens_table2cardiac

% Author: Lars Kasper
% Created: 2016-02-29
% Copyright (C) 2016 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% License (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

tapas_physio_strip_fields(dataPhysio);

stringTitle = 'Read-In: Raw Siemens physlog data';
fh = tapas_physio_get_default_fig_params();
set(gcf, 'Name', stringTitle);
stem(cpulse_on, ampl*ones(size(cpulse_on)), 'g'); hold all;
stem(cpulse_off, ampl*ones(size(cpulse_off)), 'r');
stem(t(stopSample), ampl , 'm');

stringLegend = {'physio trigger on', 'physio trigger off', ...
    'assumed last sample of last scan volume'};

nChannels = size(recordingChannels, 2);

for iChannel = 1:nChannels
    plot(t, recordingChannels(:,iChannel));
    stringLegend{end+1} = sprintf('channel %d', iChannel);
end

plot(t, meanChannel);
stringLegend{end+1}  = 'mean of channels';

if ~isempty(recording_on)
    stem(recording_on, ampl*ones(size(recording_on)), 'k');
    stringLegend{end+1} = 'physio recording on';
end

if ~isempty(recording_off)
    stem(recording_off, ampl*ones(size(recording_off)), 'k');
    stringLegend{end+1} = 'physio recording off';
end
legend(stringLegend);
title(stringTitle);
xlabel('t (seconds)');
