function [VOLLOCS, LOCS, verbose] = tapas_physio_create_scan_timing_from_gradients_philips(log_files, thresh, sqpar, verbose)
%extracts slice and volume scan events from gradients timecourse of Philips
% SCANPHYSLOG file
%
%   [VOLLOCS, LOCS] = tapas_physio_create_scan_timing_from_gradients_philips(logfile,
%   thresh);
%
% IN
%   log_files   is a structure containing the following filenames (with full
%           path)
%       .log_cardiac        contains ECG or pulse oximeter time course
%                           for Philips: 'SCANPHYSLOG<DATE&TIME>.log';
%                           can be found on scanner in G:/log/scanphyslog-
%                           directory, one file is created per scan, make sure to take
%                           the one with the time stamp corresponding to your PAR/REC
%                           files
%       .log_respiration    contains breathing belt amplitude time course
%                           for Philips: same as .log_cardiac
%
%   thresh             gradient amplitude thresholds to detect slice and volume events            
%           
%           thresh is a structure with the following elements
%           .zero    - gradient values below this value are set to zero;
%                      should be those which are unrelated to slice acquisition start 
%           .slice   - minimum gradient amplitude to be exceeded when a slice
%                      scan starts
%           .vol     - minimum gradient amplitude to be exceeded when a new
%                      volume scan starts;
%                      leave [], if volume events shall be determined as 
%                      every Nslices-th scan event
%           .grad_direction
%                    - leave empty to use nominal timing; 
%                      if set, sequence timing is calculated from logged gradient timecourse;
%                    - value determines which gradient direction timecourse is used to
%                      identify scan volume/slice start events ('x', 'y', 'z')
%           .vol_spacing
%                   -  duration (in seconds) from last slice acq to
%                      first slice of next volume; 
%                      leave [], if .vol-threshold shall be used
%
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
%           .TimeSliceToSlice - time between the acquisition of 2 subsequent
%                             slices; typically TR/Nslices or
%                             minTR/Nslices, if minimal temporal slice
%                             spacing was chosen
%            onset_slice    - slice whose scan onset determines the adjustment of the 
%                             regressor timing to a particular slice for the whole volume
%
%                             NOTE: only necessary, if thresh.grad_direction is empty
%   verbose                
%
% OUT
%
% EXAMPLE
%   [VOLLOCS, LOCS] = tapas_physio_create_scan_timing_from_gradients_philips(logfile,
%   thresh.scan_timing);
%
%   See also
%
% Author: Lars Kasper
% Created: 2013-02-16
% Copyright (C) 2013 Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_create_scan_timing_from_gradients_philips.m 235 2013-08-19 16:28:07Z kasperla $
    
% everything stored in 1 logfile
if ~isfield(log_files, 'cardiac') || isempty(log_files.cardiac)
    logfile = log_files.respiration;
else
    logfile = log_files.cardiac;
end

do_detect_vol_events_by_count = ~isfield(thresh, 'vol') || isempty(thresh.vol);
do_detect_vol_events_by_grad_height = ~do_detect_vol_events_by_count && (~isfield(thresh, 'vol_spacing') || isempty(thresh.vol_spacing));

% check consistency of thresh-values

if thresh.slice <= thresh.zero
    error('Please set thresh.scan_timing.slice > thresh.scan_timing.zero');
end

if do_detect_vol_events_by_grad_height && (thresh.slice > thresh.vol)
    error('Please set thresh.scan_timing.vol > thresh.scan_timing.slice');
end
    


Nscans          = sqpar.Nscans;
Ndummies        = sqpar.Ndummies;
NslicesPerBeat  = sqpar.NslicesPerBeat;
Nslices         = sqpar.Nslices;
do_count_from_start = isfield(sqpar, 'Nprep') && ~isempty(sqpar.Nprep);
if do_count_from_start
    Nprep = sqpar.Nprep;
end


[z{1:10}]=textread(logfile,'%d %d %d %d %d %d %d %d %d %d','commentstyle', 'shell');
y = cell2mat(z);

Nsamples=size(y,1);

dt = 2e-3; %500 Hz sampling frequency
t=((0:(Nsamples-1))*dt)';



    % finding scan volume starts (svolpulse)
    switch lower(thresh.grad_direction)
        case 'x'
            gradient_choice = y(:,7);
        case 'y'
            gradient_choice = y(:,8);
        case 'z'
            gradient_choice = y(:,9);
    end
    gradient_choice = reshape(gradient_choice, length(gradient_choice),1);
    
    % if no gradient timecourse was recorded in the logfile (due to insufficient
    % Philips software keys), return nominal timing instead
    if ~any(gradient_choice) % all values zero
            [VOLLOCS, LOCS] = tapas_physio_create_nominal_scan_timing(t, sqpar);
            warning('No gradient timecourse was logged in the logfile. Using nominal timing from sqpar instead');
        return
    end
    
    z2 = gradient_choice; z2(z2<thresh.zero)=0;
    z2 = z2 + rand(size(z2)); % to find double-peaks/plateaus, make them a bit different
    
    [tmp,LOCS]    = tapas_physio_findpeaks(z2,'minpeakheight',thresh.slice,'minpeakdistance',2);
        
    try
    if do_detect_vol_events_by_count
        if do_count_from_start
            VOLLOCS = LOCS(Nprep*Nslices + (1:Nslices:(Ndummies+Nscans)*Nslices));
        else % count from end
            VOLLOCS = LOCS((end-(Ndummies+Nscans)*Nslices+1):Nslices:end);
        end
        
    else
        if do_detect_vol_events_by_grad_height
            [tmp,VOLLOCS] = tapas_physio_findpeaks(z2,'minpeakheight',thresh.vol,'minpeakdistance',2*(Nslices-1));
        else %detection via bigger spacing from last to first slice of next volume
            VOLLOCS = LOCS(find((diff(LOCS) > thresh.vol_spacing/dt)) + 1);
        end
    end
    LOCS    = reshape(LOCS,length(LOCS),1);
    VOLLOCS = reshape(VOLLOCS,length(VOLLOCS),1);
    catch
        VOLLOCS = [];
    end
    
    if verbose.level>=2
        verbose.fig_handles(end+1) = tapas_physio_get_default_fig_params();
        set(gcf,'Name', 'Thresholding Gradient for slice acq start detection');
        fs(1) = subplot(2,1,1);
        hp = plot(t,[gradient_choice z2]); hold all;
        hp(end+1) = plot(t, repmat(thresh.zero, length(t),1));
        hp(end+1) = plot(t, repmat(thresh.slice, length(t),1));
        lg = {'chosen gradient for thresholding', ...
            'gradient with values<thresh.zero set to 0', ...
            'thresh.zero', 'thresh.slice'};
        if do_detect_vol_events_by_grad_height
            hp(end+1) = plot(t, repmat(thresh.vol, length(t),1));
            lg = {lg{:}, 'thresh.vol'};
        end
        title({'Thresholding Gradient for slice acq start detection', '- found scan events -'});
        legend(hp, lg);
        xlabel('t(s)');
    end
    
    % VOLLOCS = find(abs(diff(z2))>thresh.vol);
    if isempty(VOLLOCS) || isempty(LOCS)
        error('No volume start events found, Decrease thresh.vol or thresh.slice after considering the Thresholding figure');
    elseif length(LOCS) < NslicesPerBeat
        error('Too few slice start events found. Decrease thresh.slice after considering the Thresholding figure');
    end
    
    if do_count_from_start
        if length(VOLLOCS)< (Nprep+Nscans+Ndummies)
            error('Not enough volume events found. Please lower thresh.vol');
        end
    else
        if length(VOLLOCS)< (Nscans+Ndummies)
            error('Not enough volume events found. Please lower thresh.vol');
        end
    end
    
    %% Plot gradient thresholding for slice timing determination
    if verbose.level >= 2 % continue figure, if sth was found!
        hp(end+1) = stem(t(VOLLOCS),1.25*max(gradient_choice)*ones(size(VOLLOCS))); hold all
        hp(end+1) = stem(t(LOCS),max(gradient_choice)*ones(size(LOCS))); hold all
        lg = {lg{:}, 'found volume events', 'found slice events'};
        legend(hp, lg);
        ymin = tapas_physio_prctile(diff(LOCS), 25);
        ymax = tapas_physio_prctile(diff(LOCS), 99);
        
        fs(2) = subplot(2,1,2);
        plot(t(LOCS(1:end-1)), diff(LOCS)); title('duration betwenn scan events - search for bad peaks here!');
        xlabel('t(s)');
        ylabel('t(ms)');
        ylim([0.9*ymin, 1.1*ymax]);
        linkaxes(fs,'x');
    end
