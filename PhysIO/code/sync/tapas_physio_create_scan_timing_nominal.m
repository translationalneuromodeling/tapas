function [VOLLOCS, LOCS] = tapas_physio_create_scan_timing_nominal(t, ...
    sqpar, align_scan, durationPhyslogAfterEndOfLastScan)
% Creates locations of scan volume and slice events in time vector of SCANPHYSLOG-files
%
%   [VOLLOCS, LOCS] = tapas_physio_create_scan_timing_nominal(t, sqpar);
%
% In cases where the SCANPHYSLOG-file has no gradient entries (column
% 7-9), the actual time-course of the sequence has to be inferred from the
% nominal sequence parameters in sqpar. Here, the corresponding slice scan
% events are generated for the resampling of regressors in the GLM under
% the assumption that the SCANPHYSLOG-file ended exactly when the scan ended
% ...which is usually the case if a scan is not stopped manually
% 
% Additionally, one can set a buffer end time, i.e., the duration of the
% phys logging lasting longer after the end of the last scan
%
% IN
%   t       - timing vector of SCANPHYSLOG-file, usually sampled with 500
%             Hz (Philips)
%   sqpar                   - sequence timing parameters
%           .Nslices        - number of slices per volume in fMRI scan
%           .NslicesPerBeat - usually equals Nslices, unless you trigger with the heart beat
%           .TR             - repetition time in seconds
%           .Ndummies       - number of dummy volumes
%           .Nscans         - number of full volumes saved (volumes in nifti file,
%                             usually rows in your design matrix)
%           .Nprep          - number of non-dummy, volume like preparation pulses
%                             before 1st dummy scan. If set, logfile is read from beginning,
%                             otherwise volumes are counted from last detected volume in the logfile
%           .time_slice_to_slice - time between the acquisition of 2 subsequent
%                             slices; typically TR/Nslices or
%                             minTR/Nslices, if minimal temporal slice
%                             spacing was chosen
%           .onset_slice    - slice whose scan onset determines the adjustment of the
%                             regressor timing to a particular slice for the whole volume
%   align_scan              'first' or 'last' (default)
%                           'first' t == 0 will be aligned to first scan
%                                   volume, first slice
%                           'last'  t(end) will be aligned to last scan
%                                   volume, last slice
%   durationPhyslogAfterEndOfLastScan
%                            duration (in seconds) of physiological logfile
%                            after end of last scan volume in the run
%                           default: 0
% OUT
%           VOLLOCS         - locations in time vector, when volume scan
%                             events started
%           LOCS            - locations in time vector, when slice or volume scan
%                             events started
% EXAMPLE
%   [VOLLOCS, LOCS] = tapas_physio_create_scan_timing_nominal(t, sqpar);
%
%   See also

% Author: Lars Kasper
% Created: 2013-02-07
% Copyright (C) 2013 Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


if nargin < 3
    align_scan = 'last';
end

if nargin < 4
    durationPhyslogAfterEndOfLastScan = 0;
end

Nscans          = sqpar.Nscans;
Ndummies        = sqpar.Ndummies;

NallVols = (Ndummies+Nscans);
VOLLOCS = NaN(NallVols,1);
TR = sqpar.TR;


%% First, find volume starts either forward or backward through time series
do_count_from_start = strcmpi(align_scan, 'first');
if do_count_from_start % t = 0 is assumed to be the start of the scan
    for n = 1:NallVols
        [tmp, VOLLOCS(n)] = min(abs(t - TR*(n-1)));
    end
else
    tEndLastScan = t(end)-durationPhyslogAfterEndOfLastScan;
    
    tStartPhys = t(1);
    for n = 1:NallVols
        
        tStartVol = (tEndLastScan-TR*(NallVols-n+1));
        
        if tStartPhys > tStartVol
            VOLLOCS(n) = NaN;
        else
            [tmp, VOLLOCS(n)] = min(abs(t - tStartVol));
        end
    end
end

%% Then, find slice starts between determined volume starts

LOCS = tapas_physio_create_LOCS_from_VOLLOCS(VOLLOCS, t, sqpar);


end