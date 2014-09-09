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
%
% Author: Lars Kasper
% Created: 2013-02-16
%
% Copyright (C) 2013, Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_get_onsets_from_locs.m 466 2014-04-27 13:10:48Z kasperla $

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
for v = 1:Nallvols-1
    SLICELOCS{v} = LOCS(intersect(find(LOCS>=VOLLOCS(v), Nslices, 'first'), ...
        find(LOCS<VOLLOCS(v+1))));
    if length(SLICELOCS{v})~=Nslices
        warning(sprintf('Volume event %d: %d instead of %d slice events found\n', v, length(SLICELOCS{v}), Nslices));
    end
end
SLICELOCS{Nallvols} = LOCS(find(LOCS>=VOLLOCS(Nallvols), Nslices, 'first'));
if length(SLICELOCS{Nallvols})~=Nslices
    warning(sprintf('Volume event %d: %d instead of %d slice events found\n', Nallvols, length(SLICELOCS{Nallvols}), Nslices));
end

if verbose.level >= 3
    titstr =  'Slice bundles belonging to 1 volume';
    verbose.fig_handles(end+1) = tapas_physio_get_default_fig_params();
    set(gcf, 'Name', titstr);
    for v=1:Nallvols-1, stem(t(SLICELOCS{v}),ones(size(SLICELOCS{v})));hold all;end
    title(titstr);
    xlabel('t (seconds since SCANPHYSLOG-start)');
end

if do_count_from_start
    REALLOCS = cell2mat(SLICELOCS(Nprep+Ndummies+(1:Nscans))');
    DUMMYLOCS = cell2mat(SLICELOCS(Nprep+(1:Ndummies))');
else
    REALLOCS = cell2mat(SLICELOCS(end-Nscans+1:end)');
    DUMMYLOCS = cell2mat(SLICELOCS(end-Nscans-Ndummies+1:end-Nscans)');
end

DUMMYLOCS = reshape(DUMMYLOCS,length(DUMMYLOCS),1);
REALLOCS = reshape(REALLOCS,length(REALLOCS),1);
% 
% if verbose.level>=2
%     figure('Name', 'Found scan events through gradient thresholding');
%     %subplot(3,1,3);
%     hp3 = stem(t(LOCS),repmat(max(gradient_choice),1,length(t(LOCS)))); hold all
%     hp3(end+1) = plot(t,gradient_choice );
%     
%     hp3(end+1) = stem(t(REALLOCS),repmat(max(gradient_choice),1,length(t(REALLOCS))));
%     if Ndummies
%         hp3(end+1) = stem(t(DUMMYLOCS),repmat(max(gradient_choice),1,length(t(DUMMYLOCS))));
%         lg = {'found scan events','chosen gradient for thresholding', ...
%             'Saved slice events', 'Dummy slice events', 'vol scan events'};
%     else
%         lg = {'found scan events','chosen gradient for thresholding', ...
%             'Saved slice events', 'vol scan events'};
%     end
%     hp3(end+1) = stem(t(VOLLOCS),repmat(max(gradient_choice),1,length(t(VOLLOCS))));title('chosen scan events');
%     legend(hp3, lg{:});
%     linkaxes(fs,'x');
% end

ons.acq_slice_all = [DUMMYLOCS; REALLOCS];

if do_count_from_start
    ons.acq_vol_all   = VOLLOCS(Nprep+(1:Nscans+Ndummies));
    ons.acq_slice_per_vol_all = SLICELOCS(Nprep+(1:Nscans+Ndummies));
else
    ons.acq_vol_all   = VOLLOCS(end-Ndummies-Nscans+1:end);
    ons.acq_slice_per_vol_all = SLICELOCS(end-Nscans-Ndummies+1:end);
end

spulse     = t(ons.acq_slice_all);
svolpulse  = t(ons.acq_vol_all);
spulse_per_vol = cellfun(@(x) t(x), ons.acq_slice_per_vol_all, 'UniformOutput', false);
