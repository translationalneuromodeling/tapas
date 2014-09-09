function [ons_secs, ons] = tapas_physio_read_physlog_PHILIPS(logfile, sqpar, verbose)
% reads a Philips SCANPHYSLOG-file, extracts ECG and breathing belt time
% series plus slice and volume scan events in seconds for later usage with RETROICOR-scripts; 
% 
% USAGE
%   [ons_secs, ons] = tapas_physio_read_physlog_PHILIPS(logfile, sqpar, verbose)
%------------------------------------------------------------------------
% INPUT:
%   logfile   - SCANPHYSLOG<DATE&TIME>.log from Philips G:/log/scanphyslog-
%             directory, one file is created per scan, make sure to take
%             the one with the time stamp corresponding to your PAR/REC
%             files
%   thresh
%           .zero    - lower gradient thresholds throw away gradients which are
%                      unrelated to slice acquisition start
% 
%           .slice   - minimum gradient amplitude to be exceeded for a slice
%                      scan to start
%           .vol     - minimum gradient amplitude to be exceeded for a new
%                      volume scan to start
%           .grad_direction 
%                    - which gradient direction timecourse is used to
%                    identify scan volume/slice start events ('x', 'y', 'z')
%           .vol_spacing 
%                   -  minimum distance(in samples) from last slice acq to
%                      first slice of next volume; leave [], if
%                      .vol-threshold shall be used
%           .ECG_min - [% peak height of sample QRS wave] if set, ECG heartbeat event is calculated from ECG
%                      timeseries by detecting local maxima of
%                      cross-correlation to a sample QRS-wave
%                      - if empty, Philips log of heartbeat event is read out
%
%   sqpar     - sequence timing parameters
%           .Nslices        - number of slices per volume in fMRI scan
%           .NslicesPerBeat - usually equals Nslices, unless you trigger
%                             with the heart beat
%           .TR             - repetition time in seconds
%           .Ndummies       - number of dummy volumes
%           .Nscans         - number of full volumes saved (volumes in nifti file,
%                             usually rows in your design matrix)   
%
%   logfile_w_scanevents 
%             - filename for modified scanphyslog-file where slice
%               scan events are included in last column 
%               (default: test_phys.log)
%   verbose   - create informative plots (1= yes, 0 = no)
%
%------------------------------------------------------------------------
% OUTPUT:
%   ons_secs    - onsets of all physlog events in seconds
%               .spulse     = onsets of slice scan acquisition
%               .cpulse     = onsets of cardiac R-wave peaks
%               .rpulse     = time series of respiration
%               .svolpulse  = onsets of volume scan acquisition
%               .t          = time vector of logfile rows
%
% -------------------------------------------------------------------------
% Lars Kasper, August 2011
% Copyright (C) 2013, Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_read_physlog_PHILIPS.m 354 2013-12-02 22:21:41Z kasperla $

%% =======================================================================
%% Set adjustable parameters
subj='';
figh = {};

if nargin < 3
    Nscans          = 342;
    Ndummies        = 5;
    NslicesPerBeat  = 30;
    Nslices         = 30;
    TR              = 1.75;
else
    Nscans          = sqpar.Nscans;
    Ndummies        = sqpar.Ndummies;
    NslicesPerBeat  = sqpar.NslicesPerBeat;
    Nslices         = sqpar.Nslices;
end

if nargin < 3, verbose=1; end;

sampling=1; %only for plotting

%% Correct errors in logfile created when written by Philips
if ~strfind(computer('arch'),'win')
    unix(['sed ''/^#[^ #]/d;s/000a/0005/g'' ' logfile '> tmp.log; mv tmp.log ' logfile]);
end;

%% Read out logfile
[z{1:10}]=textread(logfile,'%d %d %d %d %d %d %d %d %d %d', 'commentstyle', 'shell');
y = cell2mat(z);

Nsamples=size(y,1);
t = ((1:Nsamples)/500.0)'; %500 Hz sampling frequency

% column 3 = ECG, 5 = PPU, 6 = resp, 10 = scanner signal
% columns 7,8,9: Grad-strengh x,y,z
acq_sliceevents=find(z{10}==8 | z{10}==5); %indices where event (trigger, slice acquired) happens or accidentally 'a' was written

if isempty(thresh.ECG_min) %ECG, R-peak, read as indicated by the scanner logfile
    ecgevents=find(z{10}==1);
else % calculate R-peak next to local minimum
    ECG_min = thresh.ECG_min;
    while ECG_min
        ecgevents = tapas_physio_find_ecg_r_peaks(t,y, ECG_min);
        ECG_min = input('Press 0, then return, if right ECG peaks were found, otherwise type next numerical choice for ECG_min and continue the selection: ');
    end
end

ppuevents=find(z{10}==2); %Pulse maximum, finger pulsemeter

[ons, dur, index, any_scanevent_repaired] = ...
    repair_scan_events(acq_sliceevents, sqpar, verbose);

%% timing of scanner, cardiac and respiratory signal for Chloe's RETROICOR
ons_secs.spulse     = t(ons.acq_slice_all);
ons_secs.cpulse     = t(ecgevents);
ons_secs.rpulse     = z{6};
ons_secs.svolpulse  = t(ons.acq_vol_all);
ons_secs.t          = t;

%% HACK for subject 019_SarahM There is a trigger event missing though it was
%found AND acquisition took place
%6 time bins before scan start trigger happens usually
%cpulse=sort([cpulse; t(find(t==spulse(spulse>131 & spulse<131.05))-6)]);

%% Plot fixing prodedure in scantrigger-time-difference view
%% Plot events originally indicated by scanner and corrected spulse
%% Plot physiological data, might take some time
if verbose
    figh{4} = plot_corrected_scan_events(t, y, ons.acq_slice, spulse, svolpulse, t(ons.acq_slice_gaps), ons.acq_slice_larger_gaps, subj);
    figh{2} = plot_physiological_data(t, y, ecgevents, ppuevents, spulse, svolpulse, sampling, subj);
end

end % read_physlog


%%========================================================================
%% auxiliary plot functions follow;


%%=========================================================================
%% Plots scan events after gaps of missing indicators were filled
function fh = plot_corrected_scan_events(t, y, acq_sliceevents, spulse, svolpulse, gapfillpulse, inclscanevents, subj)
fh = tapas_physio_get_default_fig_params(1, 0.5);

set(fh,'Name','Overview scan+phys events');

%plot raw events as from logfile
stem(t(acq_sliceevents), 300*y(acq_sliceevents,10), 'c') ;
hold on;

%plot repaired events
stem(svolpulse, 1000*ones(size(svolpulse)), 'k', 'fill');
stem(spulse, 750*ones(length(spulse),1), 'g','fill');

%plot included values
stem(gapfillpulse, 1500*ones(size(gapfillpulse)), 'k');
stem(t(inclscanevents), 1500*ones(size(inclscanevents)), 'r');

title([subj, ', scan pulse check and filling']);
xlabel('t (s)'); ylabel('Amplitude (a. u.)');
legend('recorded scan events', 'corrected volume start', 'corrected scan events', 'slice scan gap', 'volume start/end gap');
end

%%=========================================================================
%% Plots cardiac and respiratory time series
function fh = plot_physiological_data(t, y, ecgevents, ppuevents, spulse, svolpulse, sampling, subj)

yshow = [3 5 6];

[fh, MyColors] = tapas_physio_get_default_fig_params(1, 0.5);

set(fh,'Name','Cardiac and respiratory time series');

x = y(1:sampling:end, yshow);
Nviewsamples=size(x,1);
hold off;


stem(spulse, 2500*ones(length(spulse),1), 'c') ;
hold on;
stem(t(ecgevents), 1000*y(ecgevents,10), 'm') ;
stem(t(ppuevents), 1000*y(ppuevents,10), 'g') ;
%set color order for plot: ECG should be red, because blood :-)
set(0,'DefaultAxesColorOrder',MyColors);
plot(t(1:sampling:end), x, '-');
stem(svolpulse, 1000*ones(length(svolpulse),1), 'k', 'fill');
xlabel('t (s)'); ylabel('Amplitude (a. u.)');
title([subj,' , cardiac+resp data and corrected events']);
legend( 'scan event marker', 'ECG event marker', 'PPU event marker', 'filtered ECG', 'finger PPU',  'resp signal');
ylim([-2500 2500]);
end

