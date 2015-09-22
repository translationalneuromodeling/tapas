function [VOLLOCS, LOCS, verbose] = ...
    tapas_physio_create_scan_timing_from_tics_siemens(t, log_files, verbose)
% Creates locations of scan volume and slice events in physiological time series vector from Siemens Tics-file
% (<date>_<time>_AcquisitionInfo*.log)
%
%   [VOLLOCS, LOCS] = tapas_physio_create_nominal_scan_timing(t, log_files);
%
% IN
%   t           - timing vector of physiological logfiles
%   log_files   - structure holding log-files, here:
%                 .scan_timing - <date>_<time>_AcquisitionInfo*.log
%                                Siemens Acquisition info logfile, e.g.
%                                 Volume_ID Slice_ID AcqTime_Tics
%                                 0 0 22953292
%                                 0 1 22954087
%                                 0 2 22953690
%                                 1 0 22954700
%                                 1 1 22955495
%                                 1 2 22955097
%                                 2 0 22956136
% OUT
%           VOLLOCS         - locations in time vector, when volume scan
%                             events started
%           LOCS            - locations in time vector, when slice or volume scan
%                             events started
% EXAMPLE
%   [VOLLOCS, LOCS] = tapas_physio_create_nominal_scan_timing(t, sqpar);
%
%   See also
%
% Author: Lars Kasper
% Created: 2014-09-08
% Copyright (C) 2014 Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_create_scan_timing_from_tics_siemens.m 763 2015-07-14 11:28:57Z kasperla $
DEBUG = verbose.level >=3;

fid = fopen(log_files.scan_timing);

C = textscan(fid, '%d %d %d', 'HeaderLines', 1);

dtTicSeconds = 2.5e-3;

idVolumes       = C{1};
idSlices        = C{2};

% HACK: take care of interleaved acquisition:
ticsAcq         = sort(double(C{3}));
tAcqSeconds     = ticsAcq*dtTicSeconds;

% find time in physiological log time closest to time stamp of acquisition
% for each time
LOCS = zeros(size(idSlices));
for iLoc = 1:numel(LOCS)
    [tmp, LOCS(iLoc)] = min(abs(t - tAcqSeconds(iLoc)));
end

% extract start times of volume by detecting index change volume id
indVolStarts = [1; find(diff(idVolumes) > 0) + 1]; 
VOLLOCS = LOCS(indVolStarts);

if DEBUG
   verbose.fig_handles(end+1) = tapas_physio_get_default_fig_params();
   stringTitle = 'Extracted Sequence Timing Siemens';
   set(gcf, 'Name', stringTitle);
   stem(t(LOCS), ones(size(LOCS))); hold all;
   stem(t(VOLLOCS), 1.5*ones(size(VOLLOCS)));
   legend('slice start events', 'volume start events');
   xlabel('t (seconds)');
   title(stringTitle);
end