function [any_scanevent_repaired, ons, dur, index] ...
    = tapas_physio_repair_scan_events_PHILIPS(ons, sqpar, verbose)
% repairs the temporal structure of the sequence of scan events
% (time-stamps when slice was acquired) by filling up holes of missing
% slice acquisition or volume acquisition starts
% 
% USAGE
%   [any_scanevent_repaired, ons, dur, index] ...
%       = tapas_physio_repair_scan_events_PHILIPS(ons, sqpar, verbose)
%
% -------------------------------------------------------------------------
% INPUT:
%   ons         - ons_samples, as given by tapas_physio_read_physlog_PHILIPS_guess_scanevents.m
%   sqpar       - sequence timing parameters
%           .Nslices        - number of slices per volume in fMRI scan
%           .NslicesPerBeat - usually equals Nslices, unless you trigger
%                             with the heart beat
%           .TR             - repetition time in seconds
%           .Ndummies       - number of dummy volumes
%           .Nscans         - number of full volumes saved (volumes in nifti file,
%                             usually rows in your design matrix)   
%
%   verbose   - create informative plots (1= yes, 0 = no)
%
% -------------------------------------------------------------------------
% OUTPUT:
%   any_scanevent_repaired - flag, if function changed ons (1=yes, 0=no)
%   ons         - ons_samples, corrected for missing slice and volume
%                 scan events
%
% Note: The temporal unit of the output ons & dur corresponds to the input, 
% i.e. if acq_sliceevents is given in samples, the output unit will be 2 ms
% usually, NOT 1 second!
%
% -------------------------------------------------------------------------
% Lars Kasper, August 2011
% Copyright (C) 2013, Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


thrmin = 1.5; % gap between slices, if spacing >thrmin*min(slicegap)
thrmax = 0.5; % gap is large, if gap > thrmax*max(slicegap)

%% repair scan events: some scan event triggers were missed so we have to
% fill up the holes
ons.acq_slice = ons.acq_slice_all(:);
dur.acq_slice = diff(ons.acq_slice);

%% =======================================================================
%% Find and remove gaps of one missing slice trigger within one volume scan of Nslices
index.hugegaps = find(dur.acq_slice>thrmax*max(dur.acq_slice));

%double gaps, think about it later...
index.doublegaps = setdiff(find(dur.acq_slice > (thrmin+1)*min(dur.acq_slice)),index.hugegaps);
if ~isempty(index.doublegaps)
    disp('There are double gaps or worse in scan events. I will not correct that!');
end


%one slice trigger missing in volume of NslicesPerBeat slices
index.gaps                  = find(dur.acq_slice>thrmin*min(dur.acq_slice) ...
                                & dur.acq_slice<(thrmin+1)*min(dur.acq_slice));
ons.acq_slice_gaps          = floor((ons.acq_slice(index.gaps)+ons.acq_slice(index.gaps+1))/2);
ons.acq_slice_filled_gaps   = sort([ons.acq_slice; ons.acq_slice_gaps]);

%% =======================================================================
%% gaps at beginning or end of Nslices-volume block still possible
dur.acq_most_slices = diff(ons.acq_slice_filled_gaps);

%indices of ons.acq_slice which belong to start of new volume
%ons.acq_vol=[1; find(dur.acq_most_slices>thrmin*min(dur.acq_most_slices))+1];

ind_vol = 1:sqpar.Nslices:length(ons.acq_slice_filled_gaps);%(length(ons.acq_slice_filled_gaps)-sqpar.Nslices*(Ndummies+Nscans)+1):sqpar.Nslices:length(ons.acq_slice_filled_gaps);
ons.acq_vol = ind_vol;%ons.acq_slice_filled_gaps(ind_vol); 

%here we have scan volume blocks with e.g. N-1 slices, scan event trigger
%missed, e.g. for 3 slices: |||   ||  |||
index.gaps_startend_vol = find(diff(ons.acq_vol)<sqpar.NslicesPerBeat);

%OK till here
%is it |||   ||  ||| or |||  ||   ||| (larger gap before or after block
%which is too short?)
gapbefore   = ons.acq_slice_filled_gaps(ons.acq_vol(index.gaps_startend_vol))-ons.acq_slice_filled_gaps(ons.acq_vol(index.gaps_startend_vol)-1);
gapafter    = ons.acq_slice_filled_gaps(ons.acq_vol(index.gaps_startend_vol+1))-ons.acq_slice_filled_gaps(ons.acq_vol(index.gaps_startend_vol+1)-1);

ons.acq_slice_all = ons.acq_slice_filled_gaps;
ons.acq_slice_larger_gaps = [];
for i = 1:length(gapbefore)
    if (gapbefore(i) > gapafter(i))
        targetons=(ons.acq_vol(index.gaps_startend_vol(i)));
        ons.acq_slice_larger_gaps(i)=2*ons.acq_slice_filled_gaps(targetons)-ons.acq_slice_filled_gaps(targetons+1);
    else
        targetons=(ons.acq_vol(index.gaps_startend_vol(i)+1)-1);
        ons.acq_slice_larger_gaps(i)=2*ons.acq_slice_filled_gaps(targetons)-ons.acq_slice_filled_gaps(targetons-1);
    end
    ons.acq_slice_all=[ons.acq_slice_all; ons.acq_slice_larger_gaps(i)];
end

ons.acq_slice_all = sort(ons.acq_slice_all);
% END of filling up start/end slice triggers in volume
% update ons.acq_vol
dur.acq_slice_all = diff(ons.acq_slice_all);

%ons.acq_vol_all=[1; find(dur.acq_slice_all>thrmin*min(dur.acq_slice_all))+1];
ons.acq_vol_all = ons.acq_vol; 

%1st volume added manually
ons.acq_vol_all = [ons.acq_slice_all(ons.acq_vol_all)];


any_scanevent_repaired = ~(length(ons.acq_slice)==length(ons.acq_slice_all) ...
    && length(ons.acq_vol)==length(ons.acq_vol_all));

if verbose && any_scanevent_repaired
    figh{1} = plot_fix_missing_scan_events(ons, ...
        dur, index);
end

end %repair_scan_events

%%========================================================================
%% auxiliary plot functions follow;


%%=========================================================================
%% plot fixing procedure in scantrigger-time-difference view
function fh = plot_fix_missing_scan_events(ons, dur, index)

fh = tapas_physio_get_default_fig_params();
set(fh,'Name','Read-In: Time-difference view between events: Fix missing scan events');
ax(1) = subplot(3,1,1);
hold off;
plot(1:length(dur.acq_slice),dur.acq_slice);
hold on;
plot(index.hugegaps,dur.acq_slice(index.hugegaps),'bo')
plot(index.doublegaps,dur.acq_slice(index.doublegaps),'go')
plot(index.gaps,dur.acq_slice(index.gaps),'kx')

title('As extracted from the logfile');
xlabel('scan event + 1');
ylabel('\Delta t (ms) from last event')
warning('off', 'MATLAB:legend:IgnoringExtraEntries');
legend(...
    'dur.acq_slice (time diff between scan events', ...
    'index.hugegaps (dur.acq\_slice>thrmax*max(dur.acq\_slice))', ...
    'index.doublegaps (dur.acq\_slice > (thrmin+1)*min(dur.acq\_slice), but no hugegap)', ...
    'index.gaps (roughly size of 2*sliceTR)'...
    );

ax(2) = subplot(3,1,2);
hold off;
plot(1:length(dur.acq_most_slices),dur.acq_most_slices);
hold on;
plot(ons.acq_vol,dur.acq_most_slices(ons.acq_vol),'bx')

title('After fixing the slice-trigger gaps within a volume');
xlabel('scan event + 1');
ylabel('\Delta t (ms) from last event')
legend(...
    'dur.acq\_most\_slices', ...
    'ons.acq\_vol');


ax(3) = subplot(3,1,3);
hold off;
plot(1:length(dur.acq_slice_all),dur.acq_slice_all);
hold on;
plot(ons.acq_vol_all,dur.acq_slice_all(ons.acq_vol),'bx')

legend(...
    'dur.acq\_slice\_all', ...
    'ons.acq\_vol\_all');

title('After inserting the slice triggers at the beginning or end of a volume');
xlabel('scan event + 1');
ylabel('\Delta t (ms) from last event')

linkaxes(ax,'x');
end
