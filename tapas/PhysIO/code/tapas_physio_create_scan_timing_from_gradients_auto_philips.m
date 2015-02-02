function [VOLLOCS, LOCS, verbose] = ...
    tapas_physio_create_scan_timing_from_gradients_auto_philips(log_files, ...
    thresh, sqpar, verbose)
% Automatically extracts slice/volume scan events from gradient timecourse
% SCANPHYSLOG file
%
% [VOLLOCS, LOCS, verbose] = ...
%     tapas_physio_create_scan_timing_from_gradients_auto_philips(log_files, ...
%     thresh, sqpar, verbose)
%
% This function determines slice/volume starts from the gradient time course
% automatically by assuming a regularity of them from the
% sequence timing parameters, in particular TR and number of slices
%
% Therefore, unlike tapas_physio_create_scan_timing_from_gradients_philips,
% no thresholds of slice/volume starts have to be given to determine the
% timing and are inferred on iteratively instead.
%
%   Workflow
%   1. Determine template of each volume gradient time course by using TR.
%   2. Determine volume events using volume template (counting either from
%   start or end of the time series)
%   3. Determine slice events between all detected volumes
%       (again, creating a slice template and matching it to time series
%       between consecutive volumes...)
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
%
%   sqpar                   - sequence timing parameters
%           .nSlices        - number of slices per volume in fMRI scan
%           .nSlicesPerBeat - usually equals nSlices, unless you trigger with the heart beat
%           .TR             - repetition time in seconds
%           .nDummies       - number of dummy volumes
%           .nScans         - number of full volumes saved (volumes in nifti file,
%                             usually rows in your design matrix)
%           .Nprep          - number of non-dummy, volume like preparation pulses
%                             before 1st dummy scan. If set, logfile is read from beginning,
%                             otherwise volumes are counted from last detected volume in the logfile
%           .time_slice_to_slice - time between the acquisition of 2 subsequent
%                             slices; typically TR/nSlices or
%                             minTR/nSlices, if minimal temporal slice
%                             spacing was chosen
%            onset_slice    - slice whose scan onset determines the adjustment of the
%                             regressor timing to a particular slice for the whole volume
%
%                             NOTE: only necessary, if thresh.grad_direction is empty
%   verbose
%
% OUT
%           VOLLOCS         - locations in time vector, when volume scan
%                             events started
%           LOCS            - locations in time vector, when slice or volume scan
%                             events started
%
% EXAMPLE
%   [VOLLOCS, LOCS] = tapas_physio_create_scan_timing_from_gradients_philips(logfile,
%   thresh.scan_timing);
%
%   See also tapas_physio_create_scan_timing_from_gradients_philips
%
% Author: Lars Kasper
% Created: 2015-01-09
% Copyright (C) 2013 Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_create_scan_timing_from_gradients_philips.m 632 2015-01-09 12:36:12Z kasperla $


% smaller than typical single shot EPI slice duration (including waiting
% for TE)

debug = verbose.level >=2 ;

minSliceDuration = 0.040;

doCountSliceEventsFromLogfileStart  = ...
    strcmpi(log_files.align_scan, 'first');


% everything stored in 1 logfile
if ~isfield(log_files, 'cardiac') || isempty(log_files.cardiac)
    logfile = log_files.respiration;
else
    logfile = log_files.cardiac;
end

nScans          = sqpar.Nscans;
nDummies        = sqpar.Ndummies;
nSlicesPerBeat  = sqpar.NslicesPerBeat;
nSlices         = sqpar.Nslices;


if doCountSliceEventsFromLogfileStart
    Nprep = sqpar.Nprep;
end

y = tapas_physio_read_physlogfiles_philips_matrix(logfile);

acq_codes   = y(:,10);
nSamples    = size(y,1);

dt          = log_files.sampling_interval(1);

%default: 500 Hz sampling frequency
if isempty(dt)
    dt      = 2e-3;
end

t           = -log_files.relative_start_acquisition + ((0:(nSamples-1))*dt)';



% finding scan volume starts (svolpulse)

G = y(:,7:9);
switch lower(thresh.grad_direction)
    case 'x'
        gradient_choice = y(:,7);
    case 'y'
        gradient_choice = y(:,8);
    case 'z'
        gradient_choice = y(:,9);
    case {'xyz', 'abs'}
        gradient_choice = sqrt(sum(y(:,7:9).^2,2));
    otherwise
        gradient_choice = sqrt(sum(y(:,7:9).^2,2));
end
gradient_choice         = reshape(gradient_choice, [] ,1);

if debug
    tapas_physio_plot_gradient(G);
end


if verbose.level>=1
    % Depict all gradients, raw
    [fh, fs] = plot_raw_gradients(t, y, acq_codes);
    verbose.fig_handles(end+1) = fh;
end


% For new Ingenia log-files, recorded gradient strength may change after a
% certain time and introduce steps that are bad for recording

minStepDistanceSamples = 1.5*ceil(sqpar.TR/dt);
gradient_choice = tapas_physio_rescale_gradient_gain_fluctuations(...
    gradient_choice, minStepDistanceSamples, verbose);


%% 1. Create slice template
% take template from end of readout to avoid problems with initial
% values...
minVolumeDistanceSamples    = ceil(sqpar.TR*0.95/dt);
minSliceDistanceSamples     = ceil(minSliceDuration/dt);

if doCountSliceEventsFromLogfileStart
    rangeTemplateDetermination = 1:(nDummies + nScans)*...
        minVolumeDistanceSamples;
else
    rangeTemplateDetermination  =  (nSamples-(nDummies+nScans) * ...
        minVolumeDistanceSamples + 1):nSamples;
end

templateTime                = t(rangeTemplateDetermination);
templateG                   = gradient_choice(rangeTemplateDetermination);
thresh_min                  = tapas_physio_prctile(abs(templateG), 95);

[templateGradientSlice, secondGuessLOCS, averageTRSliceSamples] = ...
    tapas_physio_get_cardiac_pulse_template(templateTime, templateG, ...
    verbose, ...
    'thresh_min', thresh_min, ...
    'minCycleSamples', minSliceDistanceSamples, ...
    'shortenTemplateFactor', 0.7);

if debug
    verbose.fig_handles(end+1) = plot_template(t, templateGradientSlice);
end



%% 2. Determine slice events from template using cross-correlation

[LOCS, verbose] = tapas_physio_findpeaks_template_correlation(...
    gradient_choice, templateGradientSlice, secondGuessLOCS,...
    averageTRSliceSamples, verbose);

% Template-correlation algorithm expects regular cycles, but in certain
% parts of the gradient time course (preparation, end of log-file), there
% are no cycles expected and the gradient is silent. Therefore, remove
% those ill-detected LOCS

minTemplateAmplitude = max(abs(templateGradientSlice))*0.1;
LOCS(abs(gradient_choice(LOCS))<minTemplateAmplitude) = [];

VOLLOCS = [];

if verbose.level>=1
    % Plot gradient thresholding for slice timing determination
    axes(fs(2));
    
    % Plot gradient thresholding for slice timing determination
    plot_gradients_thresholds_events(t, gradient_choice, VOLLOCS, LOCS);
    linkaxes(fs(1:2), 'x');
end

if verbose.level>=1
    axes(fs(3));
    plot_diff_LOCS(t, LOCS, dt);
    linkaxes(fs,'x');
end

if debug
    %     verbose.fig_handles(end+1) = plot_slice_events( LOCS, t, ...
    %         gradient_choice, templateGradientSlice, secondGuessLOCS);
    %
    %     plot_diff_LOCS(t, LOCS, dt)
end



%% Select relevant events from detected ones using sequence parameter info


% VOLLOCS-detection via spacing or counting
nLocs = numel(LOCS);

if isfield(thresh, 'vol_spacing') && ~isempty(thresh.vol_spacing)
    iVolLocs = find((diff(LOCS) > thresh.vol_spacing/dt)) + 1;
else
    
    
    if doCountSliceEventsFromLogfileStart
        iVolLocs = Nprep*nSlices + (1:nSlices:(nDummies+nScans)*nSlices);
    else % count from end
        iVolLocs = (nLocs-(nDummies+nScans)*nSlices+1):nSlices:nLocs;
    end
    
end

VOLLOCS = LOCS(iVolLocs(iVolLocs<=nLocs));
LOCS    = reshape(LOCS, [], 1);
VOLLOCS = reshape(VOLLOCS, [], 1);


if verbose.level>=1
    % Plot gradient thresholding for slice timing determination
    axes(fs(2));
    
    % Plot gradient thresholding for slice timing determination
    plot_gradients_thresholds_events(t, gradient_choice, VOLLOCS, LOCS);
    linkaxes(fs, 'x');
end

%% Repair scan event problems...

% remove erroneous volume events, i.e. those due to slice gaps that are
% assumed to be volume end gaps, and create new volumes with not enough
% slices
minVolumeDistanceSamplesError = ceil((1-1/sqpar.Nslices) * sqpar.TR/dt);
idxVolError = find(diff(VOLLOCS) < minVolumeDistanceSamplesError);
if ~isempty(idxVolError)
    VOLLOCS(idxVolError(2:2:end)) = [];
end

ons.acq_slice_all               = LOCS;
ons.vol_all                     = VOLLOCS;
[any_scanevent_repaired, ons]   = ...
    tapas_physio_repair_scan_events_PHILIPS(ons, sqpar, verbose.level > 2);

LOCS                            = ons.acq_slice_all;
VOLLOCS                         = ons.vol_all;

%% Return error if not enough events flund
% VOLLOCS = find(abs(diff(z2))>thresh.vol);
if isempty(VOLLOCS) || isempty(LOCS)
    error('No volume start events found, Decrease thresh.vol or thresh.slice after considering the Thresholding figure');
elseif length(LOCS) < nSlicesPerBeat
    error('Too few slice start events found. Decrease thresh.slice after considering the Thresholding figure');
end

if doCountSliceEventsFromLogfileStart
    if length(VOLLOCS)< (Nprep+nScans+nDummies)
        error(['Not enough volume events found. \n\tFound:  %d\n ' ...
            '\tNeeded: %d+%d+%d (Nprep+nDummies+nScans)\n' ...
            'Please lower thresh.vol or thresh.vol_spacing\n'], ...
            length(VOLLOCS), Nprep, nDummies, nScans);
    end
else
    if length(VOLLOCS)< (nScans+nDummies)
        error(['Not enough volume events found. \n\tFound:  %d\n ' ...
            '\tNeeded: %d+%d (nDummies+nScans)\n' ...
            'Please lower thresh.vol or thresh.vol_spacing\n'], ...
            length(VOLLOCS), nDummies, nScans);
    end
end

end

%% local functions for debugging plots

%% Plot template for volume repetition
function fh = plot_template(t, templateGradientVolume)

stringTitle = 'Template Gradient Timecourse during 1 Slice';
fh = tapas_physio_get_default_fig_params();
set(gcf, 'Name', stringTitle);

nSamplesTemplate = numel(templateGradientVolume);
plot(t(1:nSamplesTemplate), templateGradientVolume);
xlabel('t (s)')
title(stringTitle);
end


%% Plot Detected volume events
function fh = plot_volume_events(VOLLOCS, t, G, ...
    templateGradientVolume, secondGuessVOLLOCS)

stringTitle = 'Template Gradient Timecourse during 1 Volume';
fh = tapas_physio_get_default_fig_params();
set(gcf, 'Name', stringTitle);

ampl    = max(abs(G));
sG      = conv(G, templateGradientVolume/sum(abs(templateGradientVolume)), ...
    'same');

plot(t, G); hold all;
plot(t, sG);
stem(t(VOLLOCS), ampl*ones(size(VOLLOCS)));
stem(t(secondGuessVOLLOCS), ampl*ones(size(secondGuessVOLLOCS)));

stringLegend = {
    'Gradient Timecourse'
    'Gradient Timecourse convolved with Template'
    'Final detected volume events'
    'Prior detected volume events'
    };

xlabel('t (s)')
title(stringTitle);
legend(stringLegend)
end


%% Plot Detected slice events
function fh = plot_slice_events(LOCS, t, G, ...
    templateGradientSlice, secondGuessLOCS)

stringTitle = 'Template Gradient Timecourse during 1 Slice';
fh = tapas_physio_get_default_fig_params();
set(gcf, 'Name', stringTitle);

ampl    = max(abs(G));
sG      = conv(G, templateGradientSlice/sum(abs(templateGradientSlice)), ...
    'same');

plot(t, G); hold all;
plot(t, sG);
stem(t(LOCS), ampl*ones(size(LOCS)));
stem(t(secondGuessLOCS), ampl*ones(size(secondGuessLOCS)));

stringLegend = {
    'Gradient Timecourse'
    'Gradient Timecourse convolved with Template'
    'Final detected slice events'
    'Prior detected slice events'
    };

xlabel('t (s)')
title(stringTitle);
legend(stringLegend)
end


%% Plot difference between detected slice events
function plot_diff_LOCS(t, LOCS, dt)


dLocsSecs = diff(LOCS)*dt*1000;
ymin = tapas_physio_prctile(dLocsSecs, 25);
ymax = tapas_physio_prctile(dLocsSecs, 99);


plot(t(LOCS(1:end-1)), dLocsSecs);
title('duration between scan events - search for bad peaks here!');
xlabel('t (s)');
ylabel('t (ms)');
ylim([0.9*ymin, 1.1*ymax]);

end

%% Plot all raw gradient time-courses
function [fh, fs] = plot_raw_gradients(t, y, acq_codes)
fh = tapas_physio_get_default_fig_params();
set(gcf,'Name', 'Thresholding Gradient for slice acq start detection');
for i=1:3, fs(i) = subplot(3,1,i); end

axes(fs(1));

plot(t, sqrt(sum(y(:,7:9).^2,2)), '--k');
hold all;
plot(t, y(:,7:9));


if ismember(8,acq_codes)
    hold all;
    stem(t, acq_codes*max(max(abs(y(:,7:9))))/20);
end


legend('abs(G_x^2+G_y^2+G_z^2)', 'gradient x', 'gradient y', 'gradient z');
title('Raw Gradient Time-courses');
end


% Plot gradient thresholding for slice timing determination
function plot_gradients_thresholds_events(t, gradient_choice, VOLLOCS, LOCS)

hp = plot(t,gradient_choice); hold all;
lg = {'Chosen gradient for thresholding'};

title({'Thresholding Gradient for slice acq start detection', '- found scan events -'});
legend(hp, lg);
xlabel('t(s)');

if ~isempty(VOLLOCS)
    hp(end+1) = stem(t(VOLLOCS), 1.25*ones(size(VOLLOCS))); hold all
    lg{end+1} = sprintf('Found volume events (N = %d)', numel(VOLLOCS));
end

if ~isempty(LOCS)
    hp(end+1) = stem(t(LOCS), ones(size(LOCS))); hold all
    lg{end+1} = sprintf('Found slice events (N = %d)', numel(LOCS));
end
legend(hp, lg);
end