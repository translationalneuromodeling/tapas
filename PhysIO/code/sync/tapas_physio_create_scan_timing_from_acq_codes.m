function [VOLLOCS, LOCS, verbose] = ...
    tapas_physio_create_scan_timing_from_acq_codes(t, acq_codes, sqpar, align_scan, verbose)
% Creates scan volume and slice onsets location in time series vector from acq_codes (Philips, Biopac),
% i.e. trigger signals on same time scale as physiological recordings
%
% [VOLLOCS, LOCS, verbose] = ...
%    tapas_physio_create_scan_timing_from_acq_codes(t, acq_codes, verbose)
% IN
%   t               - [nSamples,1] timing vector of physiological logfiles
%   acq_codes       - [nSamples,1] event vector of acquisition codes (e.g.
%                     1 for slice trigger, 10 for volume trigger), as
%                     provided by tapas_physio_read_physlogfiles*
%   sqpar                   - sequence timing parameters
%           .Nslices        - number of slices per volume in fMRI scan
%           .Ndummies       - number of dummy volumes
%           .Nscans         - number of full volumes saved (volumes in nifti file,
%   align_scan      - 'first' or 'last', for abundant triggers read from
%                     acq_codes, defines whether the scan-related ones are counted from the
%                     start ('first') or end ('last') of the sequence
%   verbose         - structure defining which output figures to plot and
%                     storing their handles
%
% OUT
%   VOLLOCS         - locations in time vector, when volume scan
%                     events started
%   LOCS            - locations in time vector, when slice or volume scan
%                      events started
%
%   See also tapas_physio_create_scan_timing tapas_physio_read_physlogfiles

% Author: Lars Kasper
% Created: 2016-08-20
% Copyright (C) 2014 Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

DEBUG = verbose.level >=3;

nVolumes        = sqpar.Nscans;
nDummies        = sqpar.Ndummies;
nSlices         = sqpar.Nslices;

nTotalVolumes = nVolumes + nDummies;
nTotalSlices = nTotalVolumes*nSlices;


doCountFromStart = strcmpi(align_scan, 'first');

LOCS = find(acq_codes == 1);
VOLLOCS = find(acq_codes == 10);

if isempty(VOLLOCS) % try Philips scan trigger onset code instead
    VOLLOCS = find(acq_codes == 8);
end

isValidVOLLOCS = numel(VOLLOCS) >= nTotalVolumes;
isValidLOCS = numel(LOCS) >= nTotalSlices;

%% crop valid location vectors to needed number of volumes or slices*volumes
if isValidLOCS
    if doCountFromStart
        LOCS = LOCS(1:nTotalSlices);
    else
        LOCS = LOCS((end-(nTotalSlices+1)):end);
    end
end

if isValidVOLLOCS
    if doCountFromStart
        VOLLOCS = VOLLOCS(1:nTotalVolumes);
    else
        VOLLOCS = VOLLOCS((end-nTotalVolumes+1):end);
    end
end


%% Something invalid? try calculating from other acq_codes or use nominal
% timing!
if ~isValidLOCS || ~isValidVOLLOCS
    
    if isValidLOCS
        % no vol events, therefore take every nSlices-th slice event
        VOLLOCS = LOCS(1:nSlices:nTotalSlices);
    elseif isValidVOLLOCS
        LOCS = tapas_physio_create_LOCS_from_VOLLOCS(VOLLOCS, t, sqpar);
    else % resort to nominal timing
        [VOLLOCS, LOCS] = tapas_physio_create_scan_timing_nominal(t, sqpar);
        verbose = tapas_physio_log('No gradient timecourse was logged in the logfile. Using nominal timing from sqpar instead', ...
            verbose, 1);
    end
    
end


if DEBUG
    verbose.fig_handles(end+1) = tapas_physio_get_default_fig_params();
    stringTitle = 'Sync: Extracted Scan Slice/Volume Onsets from Acq Codes';
    set(gcf, 'Name', stringTitle);
    stem(t(LOCS), ones(size(LOCS))); hold all;
    stem(t(VOLLOCS), 1.5*ones(size(VOLLOCS)));
    legend('slice start events', 'volume start events');
    xlabel('t (seconds)');
    title(stringTitle);
end