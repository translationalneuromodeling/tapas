function [svolpulse, spulse, spulse_per_vol, verbose] = tapas_physio_get_onsets_from_locs(t, VOLLOCS, LOCS, sqpar, verbose)
% creates timing vectors of found
%
%   [svolpulse, spulse, spulse_per_vol] = tapas_physio_get_onsets_from_locs(t, VOLLOCS,
%                                           LOCS, sqpar, verbose);
%
% IN
%   t           time vector of logfile (in seconds)
%   VOLLOCS     volume scan events (in samples of logfile)
%   LOCS        all slice (including volume) scan events (in samples of logfile)
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
%            onset_slice    - slice whose scan onset determines the adjustment of the
%                             regressor timing to a particular slice for the whole volume
%
%   verbose             physio.verbose, See also tapas_physio_new
% OUT
%   svolpulse   vector of volume scan pulse events (in seconds from logfile start)
%   spulse      vector of scan pulse events (in seconds from logfile start)
%   spulse_per_vol  cell of slice scan events; one cell element per volume
% EXAMPLE
%   [ons_secs.svolpulse, ons_secs.spulse] = tapas_physio_get_onsets_from_locs(ons_secs.t, VOLLOCS, LOCS);
%
%   See also tapas_physio_main_create_regressors tapas_physio_read_physlogfiles_philips

% Author: Lars Kasper
% Created: 2013-02-16
%
% Copyright (C) 2013, Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


Nscans          = sqpar.Nscans;
Ndummies        = sqpar.Ndummies;
NslicesPerBeat  = sqpar.NslicesPerBeat;
Nslices         = sqpar.Nslices;
do_count_from_start = isfield(sqpar, 'Nprep') && ~isempty(sqpar.Nprep);
if do_count_from_start
    Nprep = sqpar.Nprep;
end

%% remove slice scan events for last, incompletely acquired volume
if VOLLOCS(end) ~= LOCS(end-NslicesPerBeat+1)
    LOCS(LOCS>=VOLLOCS(end))=[];
    VOLLOCS(end) = [];
end

Nallvols = length(VOLLOCS);

hasWarned = false;

for v = 1:Nallvols-1
    SLICELOCS{v} = LOCS(intersect(find(LOCS>=VOLLOCS(v), Nslices, 'first'), ...
        find(LOCS<VOLLOCS(v+1))));
    if length(SLICELOCS{v})~=Nslices && ~hasWarned
        verbose = tapas_physio_log(sprintf(...
            'First volume with missing slice events (%d: %d instead of %d found)\n', ...
            Nallvols, length(SLICELOCS{v}), Nslices), ...
            verbose, 1);
        hasWarned = true;
    end
    
end

SLICELOCS{Nallvols} = LOCS(find(LOCS>=VOLLOCS(Nallvols), Nslices, 'first'));

if length(SLICELOCS{Nallvols})~=Nslices && ~hasWarned
    verbose = tapas_physio_log(sprintf(...
        'First volume with missing slice events (%d: %d instead of %d found)\n', ...
        Nallvols, length(SLICELOCS{Nallvols}), Nslices), ...
        verbose, 1);
end

if verbose.level >= 3
    stringTitle =  'Sync: Slice bundles belonging to 1 volume';
    verbose.fig_handles(end+1) = tapas_physio_get_default_fig_params();
    set(gcf, 'Name', stringTitle);
    for v=1:Nallvols-1, stem(t(SLICELOCS{v}),ones(size(SLICELOCS{v})));hold all;end
    title(stringTitle);
    xlabel('t (seconds since SCANPHYSLOG-start)');
end

try
    if do_count_from_start
        REALLOCS = cell2mat(SLICELOCS(Nprep+Ndummies+(1:Nscans))');
        DUMMYLOCS = cell2mat(SLICELOCS(Nprep+(1:Ndummies))');
    else
        REALLOCS = cell2mat(SLICELOCS(end-Nscans+1:end)');
        DUMMYLOCS = cell2mat(SLICELOCS(end-Nscans-Ndummies+1:end-Nscans)');
    end
catch err
    if exist('REALLOCS', 'var')
        verbose = tapas_physio_log(...
            sprintf('Number of dummy slice event bundles does not match Ndummies. Check parameter sqpar.Ndummies or set to 0.\n\t (Matlab error: %s)', ...
            err.message), verbose, 2);
    else
        verbose = tapas_physio_log(...
            sprintf('Number of slice event bundles does not match Nscans. Check parameters sqpar.Nscans, Ndummies and Nprep.\n\t (Matlab error: %s)', ...
            err.message), verbose, 2);
    end
end

DUMMYLOCS = reshape(DUMMYLOCS,length(DUMMYLOCS),1);
REALLOCS = reshape(REALLOCS,length(REALLOCS),1);

ons.acq_slice_all = [DUMMYLOCS; REALLOCS];

if do_count_from_start
    ons.acq_vol_all   = VOLLOCS(Nprep+(1:Nscans+Ndummies));
    ons.acq_slice_per_vol_all = SLICELOCS(Nprep+(1:Nscans+Ndummies));
else
    ons.acq_vol_all   = VOLLOCS(end-Ndummies-Nscans+1:end);
    ons.acq_slice_per_vol_all = SLICELOCS(end-Nscans-Ndummies+1:end);
end


%% set scan time (in phys time scale) corresponding to index, where valid index exists 
% (i.e. for scan events occuring during phys logfile

spulse = NaN(size(ons.acq_slice_all));
svolpulse = NaN(size(ons.acq_vol_all));

iValidSlice = find(~isnan(ons.acq_slice_all));
spulse(iValidSlice) = t(ons.acq_slice_all(iValidSlice));

iValidVol = find(~isnan(ons.acq_vol_all));
svolpulse(iValidVol) = t(ons.acq_vol_all(iValidVol));

spulse_per_vol = cellfun(@(x) t(x), ons.acq_slice_per_vol_all, 'UniformOutput', false);
