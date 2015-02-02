function [ons_secs, sqpar] = tapas_physio_crop_scanphysevents_to_acq_window(ons_secs, sqpar)
% cropping of ons_secs into acquired scan/presentation session, augmenting
% sqpar by scan-timing parameters from SCANPHYSLOG-file
%
% USAGE
%   function [ons_secs, sqpar] = ...
%       tapas_physio_crop_scanphysevents_to_acq_window(ons_secs, sqpar)
%-------------------------------------------------------------------------
% INPUT:
%   ons_secs    - onsets of all physlog events in seconds
%               .spulse     = onsets of slice scan acquisition
%               .cpulse     = onsets of cardiac R-wave peaks
%               .r          = time series of respiration
%               .svolpulse  = onsets of volume scan acquisition
%               .t          = time vector of logfile rows
%
%   sqpar     - sequence timing parameters
%           .Nslices        - number of slices per volume in fMRI scan
%           .NslicesPerBeat - usually equals Nslices, unless you trigger
%                             with the heart beat
%           .TR             - repetition time in seconds
%           .Ndummies       - number of dummy volumes
%           .Nscans         - number of full volumes saved (volumes in nifti file,
%                             usually rows in your design matrix)
%            onset_slice    - slice whose scan onset determines the adjustment of the 
%                             regressor timing to a particular slice for the whole volume
%
%-------------------------------------------------------------------------
% OUTPUT:
%   ons_secs    - input ons_secs cropped to acquisition window
%                 .raw - uncropped ons_secs-structure as input into this
%                 function
%   sqpar       - augmented input, also contains
%           .maxscan        - acquired volumes during session running
%           .Nvols_paradigm - acquired volumes during paradigm running
%           .meanTR         - mean repetition time (secs) over session
%
%-------------------------------------------------------------------------
% Lars Kasper, August 2011
% Copyright (C) 2013 Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_crop_scanphysevents_to_acq_window.m 664 2015-01-30 10:59:26Z kasperla $

%% parameter settings
    Nscans          = sqpar.Nscans;
    Ndummies        = sqpar.Ndummies;
    NslicesPerBeat  = sqpar.NslicesPerBeat;
    Nslices         = sqpar.Nslices;

    spulse          = ons_secs.spulse;
    svolpulse       = ons_secs.svolpulse;
    cpulse          = ons_secs.cpulse;
    c               = ons_secs.c;
    r               = ons_secs.r;
    t               = ons_secs.t;
    ons_secs.raw    = ons_secs;
    
%% cut after end of paradigm window    
maxscan = Nscans + Ndummies;
tmax    = ons_secs.spulse_per_vol{maxscan}(end);

tstart  = ons_secs.spulse_per_vol{1}(1);
tend    = ons_secs.spulse_per_vol{maxscan}(end);

spulse((maxscan*Nslices+1):end) = [];

% 1st heartbeat should be right before first scan, others cut
% last heartbeat should be right after last scan; rest cut

acqwindow   = sort([find(cpulse<=tend & cpulse>=tstart); ...
                find(cpulse<tstart,1,'last'); find(cpulse>tend,1,'first')]);

if ~isempty(cpulse), cpulse      = cpulse(acqwindow); end;

% same for respiratory signal
acqwindow   = sort([find(t<=tend & t>=tstart); ...
                find(t<tstart,1,'last'); find(t>tend,1,'first')]);

            
if ~isempty(r), r      = r(acqwindow); end;
if ~isempty(c), c      = c(acqwindow); end;

ons_secs.t  = t(acqwindow);

% necessary vector for t1correction, all volume excitations needed
sqpar.t        = intersect(spulse,svolpulse);
if maxscan<max(size(sqpar.t))
    sqpar.t    = sqpar.t(1:maxscan*Nslices/NslicesPerBeat); %cut what's not needed
end
sqpar.maxscan=length(sqpar.t); %counts each excitation

% Counts only real repetitions of a slice
sqpar.Nvols_paradigm   = (length(sqpar.t)-Ndummies*Nslices/NslicesPerBeat)/Nslices*NslicesPerBeat;


% Mean TR of time paradigm is running, excluding dummies
lastmin1    = min([length(svolpulse)-1,sqpar.maxscan]);
sqpar.meanTR   = mean( diff(svolpulse((sqpar.Ndummies:Nslices/NslicesPerBeat:lastmin1)+1)));

formatstr = ['    maxscan (incl. dummies) = %d \n    ', ...
    'tmin (1st scan start (1st dummy))= %6.2f s\n    ',...
    'tmin (1st scan start (after dummies))= %6.2f s\n    ', ...
    'tmax = %6.2f s \n    ', ...
    'mean TR = %6.2f s\n'];

fprintf(1,formatstr, sqpar.maxscan, spulse(1), ...
    spulse(1+Ndummies*Nslices), tmax, sqpar.meanTR);


%% prepare output variable

ons_secs.c        = c;
ons_secs.r        = r;
ons_secs.spulse   = spulse;
ons_secs.cpulse   = cpulse;
ons_secs.svolpulse= svolpulse;
end

