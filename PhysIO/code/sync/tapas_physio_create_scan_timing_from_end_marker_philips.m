function [VOLLOCS, LOCS, verbose] = tapas_physio_create_scan_timing_from_end_marker_philips(log_files, thresh, sqpar, verbose)
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

% Author: Lars Kasper
% Created: 2013-02-16
% Copyright (C) 2013 Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

    
% everything stored in 1 logfile
if ~isfield(log_files, 'cardiac') || isempty(log_files.cardiac)
    logfile = log_files.respiration;
else
    logfile = log_files.cardiac;
end

do_detect_vol_events_by_count = (~isfield(thresh, 'vol') || isempty(thresh.vol)) && (~isfield(thresh, 'vol_spacing') || isempty(thresh.vol_spacing));
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

[z{1:10}]=textread(logfile,'%d %d %d %d %d %d %d %d %d %d','commentstyle', 'shell');
y = cell2mat(z);

Nsamples=size(y,1);

dt = log_files.sampling_interval; 

%default: 500 Hz sampling frequency
if isempty(dt)
    dt = 2e-3;
end

t = -log_files.startScanSeconds + ((0:(Nsamples-1))*dt)';

LOC_END_MARKER = find( y(:,end) == 20 );

TA = 1/dt;

NallVols = (Ndummies+Nscans);
VOLLOCS = zeros(NallVols,1);
LOCS = zeros(NallVols*Nslices,1);
TR = sqpar.TR;

VOLLOCS = [LOC_END_MARKER-TR*TA : -TR*TA : LOC_END_MARKER-TR*TA*NallVols];
VOLLOCS = VOLLOCS(end:-1:1);
LOCS = [];
for V=1:NallVols
    
    LOCS_PER_SLICE = sqpar.PtsSliceToSlice;
    
    for SL=1:Nslices
        LOCS(end+1) = [VOLLOCS(V) + (SL-1)*LOCS_PER_SLICE ];
    end
    
end

    LOCS    = reshape(LOCS,length(LOCS),1);
    VOLLOCS = reshape(VOLLOCS,length(VOLLOCS),1);
   
    
    if verbose.level>=1
        verbose.fig_handles(end+1) = tapas_physio_get_default_fig_params();
        set(gcf,'Name', 'Sync: Thresholding Gradient for slice acq start detection');
        fs(1) = subplot(1,1,1);
        plot(t, y(:,7:9));
        legend('gradient x', 'gradient y', 'gradient z');
        title('Raw Gradient Time-courses');
        hold on,
        ylims = ylim;
        
        plot( [(VOLLOCS(1)-1)/TA    (VOLLOCS(1)-1)/TA]  , ylims, 'k' )
        plot( [(VOLLOCS(1+Ndummies)-1)/TA    (VOLLOCS(1+Ndummies)-1)/TA]  , ylims, 'g' )
        plot( [(VOLLOCS(end)-1)/TA  (VOLLOCS(end)-1)/TA], ylims, 'k' )
        plot( [(LOCS(end)-1)/TA     (LOCS(end)-1)/TA]   , ylims, 'k' )
        plot( [(VOLLOCS(end-1)-1)/TA     (VOLLOCS(end-1)-1)/TA]   , ylims, 'k' )
        
        plot( [(LOC_END_MARKER-1)/TA (LOC_END_MARKER-1)/TA], ylims, 'g' )
    end
    
    % VOLLOCS = find(abs(diff(z2))>thresh.vol);
    if isempty(VOLLOCS) || isempty(LOCS)
        error('No volume start events found, Decrease thresh.vol or thresh.slice after considering the Thresholding figure');
    elseif length(LOCS) < NslicesPerBeat
        error('Too few slice start events found. Decrease thresh.slice after considering the Thresholding figure');
    end
    
