function [c, r, t, cpulse, acq_codes, verbose] = ...
    tapas_physio_read_physlogfiles_bids(log_files, cardiac_modality, ...
    verbose, varargin)
% Reads in 3-column tsv-file from BIDS Data (cardiac, respiratory, trigger),
% assuming log_files-meta information to be in an accompanying .json-file
% Note: if a JSON file of the same file name exists (but .json instead of .tsv)
% column order of physiological recordings will be read from there as well
% as values for sampling_interval and relative_start_acquisition, if they were
% empty before
%
% Details of the Brain Imaging Data Structure (BIDS) standard for peripheral
% physiological recordings can be found here:
%
% https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/
% 06-physiological-and-other-continous-recordings.html
%
% [c, r, t, cpulse, acq_codes, verbose] = tapas_physio_read_physlogfiles_biopac_txt(...
%    log_files, cardiac_modality, verbose, varargin)
%
% IN    log_files
%       .log_cardiac        *.tsv[.gz] file ([zipped] tab separated file,
%                           contains 3 columns of the form
%                             cardiac   respiratory trigger
%                             -0.949402	-0.00610382	0
%                             -0.949402	-0.00610382	0
%                             -0.951233	-0.00915558	0
%                             -0.951233	-0.00915558	0
%                             -0.953064	-0.0122073	0
%                             -0.953064	-0.0122073	0
%                             -0.95459	-0.0076297	1
%                             -0.95459	-0.0076297	0
%                           - cardiac and respiratory column contain the raw
%                           physiological traces
%                              - for cardiac, alternatively, one can set the
%                               cardiac triggers (cpulse), e.g. detected
%                               R-peaks, as 0/1 time series, as for scan trigger
%                           - trigger is 0 everywhere but at
%                           start of a scanned volume (=1)
%       .log_respiration    same as .log_cardiac
%       .sampling_interval  sampling interval (in seconds)
%                           default: 1 ms (1000 Hz)
%       cardiac_modality    'ECG' or 'PULS'/'PPU'/'OXY' to determine
%                           which channel data to be returned
%                           UNUSED, is always column labeled 'cardiac'
%       verbose
%       .level              debugging plots are created if level >=3
%       .fig_handles        appended by handle to output figure
%
% OUT
%   cpulse              time events of R-wave peak in cardiac time series (seconds)
%                       <UNUSED>, if raw cardiac trace is given...
%   r                   respiratory time series
%   t                   vector of time points (in seconds)
%   c                   cardiac time series (PPU)
%   acq_codes           slice/volume start events marked by number <> 0
%                       for time points in t
%                       10/20 = scan start/end;
%                       1 = ECG pulse; 2 = OXY max; 4 = Resp trigger;
%                       8 = scan volume trigger (on)
%                       16 = scan volume trigger (off)
%
% EXAMPLE
%   tapas_physio_read_physlogfiles_biopac_txt
%
%   See also tapas_physio_read_physlogfiles_siemens tapas_physio_plot_raw_physdata_siemens_hcp

% Author: Lars Kasper
% Created: 2018-12-14
% Copyright (C) 2018 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.

% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% License (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

%% read out values
DEBUG = verbose.level >= 3;

hasRespirationFile = ~isempty(log_files.respiration);
hasCardiacFile = ~isempty(log_files.cardiac);
hasExplicitJsonFile = ~isempty(log_files.scan_timing);
diffCardiacRespirationFile = ~strcmp(log_files.cardiac,log_files.respiration);

if hasCardiacFile && hasRespirationFile && diffCardiacRespirationFile % if the sampling rates of the two signals are different, they are registered in different files according to BIDS.
    fileName{1} = log_files.cardiac;
    fileName{2} = log_files.respiration;
    
    [c,r,t,cpulse,acq_codes] = tapas_physio_read_physlogfiles_bids_separate(fileName,log_files,hasExplicitJsonFile,DEBUG,verbose);
    
elseif hasCardiacFile
    fileName = log_files.cardiac;
    [c,r,t,cpulse,acq_codes] = tapas_physio_read_physlogfiles_bids_unified(fileName,log_files,hasExplicitJsonFile,DEBUG,verbose);
    
elseif hasRespirationFile
    fileName = log_files.respiration;
    [c,r,t,cpulse,acq_codes] = tapas_physio_read_physlogfiles_bids_unified(fileName,log_files,hasExplicitJsonFile,DEBUG,verbose);
end

end

%% Function for separate physio files (cardiac and resp with different sampling rates)
function [c,r,t,cpulse,acq_codes] = tapas_physio_read_physlogfiles_bids_separate(fileName,log_files,hasExplicitJsonFile,DEBUG,verbose)

%% Read cardiac
%% check for .gz files and unzip to temporary file
[~, ~, ext] = fileparts(fileName{1});

isZipped = strcmpi(ext, '.gz');

if isZipped
    fileJson = regexprep(fileName{1}, '\.tsv\.gz', '\.json');
    tempFilePath = tempname;  % tempname is matlab inbuilt
    fileNameCardiac  = gunzip(fileName{1}, tempFilePath);
    fileNameCardiac = fileNameCardiac{1};
else
    fileJson = regexprep(fileName{1}, '\.tsv', '\.json');
end

if hasExplicitJsonFile
    fileJson = log_files.scan_timing;
end

hasJsonFile = isfile(fileJson);

if hasJsonFile
    val = jsondecode(fileread(fileJson));
else
    verbose = tapas_physio_log(...
        ['No .json file found. Please specify log_files.sampling_interval' ...
        ' and log_files.relative_start_acquisition explicitly.'], verbose, 1);
end

dtCardiac = log_files.sampling_interval;
if isempty(dtCardiac)
    if hasJsonFile
        dtCardiac = 1/val.SamplingFrequency;
    else
        verbose = tapas_physio_log(...
            ['No .json file found and empty log_files.sampling_interval. ' ...
            'Please specify explicitly.'], verbose, 2);
    end
end

% sum implicit (.json) and explicit relative shifts of log/scan acquisition
if isempty(log_files.relative_start_acquisition)
    if hasJsonFile
        % in BIDS, start of the phys logging is stated relative to the first volume scan start.
        % PhysIO defines the scan acquisiton relative to the phys log start
        tRelStartScan = -val.StartTime;
    else
        verbose = tapas_physio_log(...
            ['No .json file found and empty log_files.relative_start_acquisition. ' ...
            'Please specify explicitly.'], verbose, 2);
    end
else
    if hasJsonFile
        % add both delays
        tRelStartScan = log_files.relative_start_acquisition - val.StartTime;
    else
        tRelStartScan = log_files.relative_start_acquisition;
    end
end

% default columns in text file for phys recordings; overruled by JSON file
% 1 = cardiac, 2 = resp, 3 = trigger
bidsColumnNamesCardiac = {'cardiac', 'trigger'};
idxCol = 1:2;  %set default values for columns from BIDS
for iCol = 1:2
    if hasJsonFile
        idxCurrCol = find(cellfun(@(x) isequal(lower(x), bidsColumnNamesCardiac{iCol}), val.Columns));
        if ~isempty(idxCurrCol)
            idxCol(iCol) = idxCurrCol;
        end
    end
end

C = tapas_physio_read_columnar_textfiles(fileNameCardiac, 'BIDS');
c = double(C{idxCol(1)});
iAcqStart = (double(C{idxCol(2)})~=0); % trigger has 1, rest is 0;

%% delete temporary unzipped file
if isZipped
    [status,message,messageId] = rmdir(tempFilePath, 's');
    % warning if deletion failed
    if status == 0
        tapas_physio_log(sprintf('%s: %s', messageId, message), verbose, 1)
    end
end

%% Read respiratory
%% check for .gz files and unzip to temporary file
[~, ~, ext] = fileparts(fileName{2});

isZipped = strcmpi(ext, '.gz');

if isZipped
    fileJson = regexprep(fileName{2}, '\.tsv\.gz', '\.json');
    tempFilePath = tempname;  % tempname is matlab inbuilt
    fileNameRespiration  = gunzip(fileName{2}, tempFilePath);
    fileNameRespiration = fileNameRespiration{1};
else
    fileJson = regexprep(fileName{2}, '\.tsv', '\.json');
end

if hasExplicitJsonFile
    fileJson = log_files.scan_timing;
end

hasJsonFile = isfile(fileJson);

if hasJsonFile
    val = jsondecode(fileread(fileJson));
else
    verbose = tapas_physio_log(...
        ['No .json file found. Please specify log_files.sampling_interval' ...
        ' and log_files.relative_start_acquisition explicitly.'], verbose, 1);
end

dtRespiration = log_files.sampling_interval;
if isempty(dtRespiration)
    if hasJsonFile
        dtRespiration = 1/val.SamplingFrequency;
    else
        verbose = tapas_physio_log(...
            ['No .json file found and empty log_files.sampling_interval. ' ...
            'Please specify explicitly.'], verbose, 2);
    end
end

% default columns in text file for phys recordings; overruled by JSON file
% 1 = resp, 2 = trigger
bidsColumnNamesRespiration = {'respiratory', 'trigger'};
idxCol = 1:2;  %set default values for columns from BIDS
for iCol = 1:2
    if hasJsonFile
        idxCurrCol = find(cellfun(@(x) isequal(lower(x), bidsColumnNamesRespiration{iCol}), val.Columns));
        if ~isempty(idxCurrCol)
            idxCol(iCol) = idxCurrCol;
        end
    end
end

C = tapas_physio_read_columnar_textfiles(fileNameRespiration, 'BIDS');
r = double(C{idxCol(1)});

%% Create timing vector from samples
nSamplesCardiac = length(c);
nSamplesRespiration = length(r);

tCardiac = -tRelStartScan + ((0:(nSamplesCardiac-1))*dtCardiac)';
tRespiration = -tRelStartScan + ((0:(nSamplesRespiration-1))*dtRespiration)';

%% Deal with NaNs in c and r timecourse
c(isnan(c)) = interp1(tCardiac(~isnan(c)), c(~isnan(c)), tCardiac(isnan(c)));
r(isnan(r)) = interp1(tRespiration(~isnan(r)), r(~isnan(r)), tRespiration(isnan(r)));

%% occasionally, cardiac time course is instead containing 0/1 cardiac triggers,
% and not raw trace; check this and populate cpulse accordingly
if all(ismember(unique(c), [1 0]))
    cpulse = tCardiac(c==1);
else
    cpulse = [];
end

%% interpolate to greater precision, if both files exist and
% 2 different sampling rates are given
% interpolate acq_codes and trace with lower sampling rate to higher
%rate

%dtCardiac = tCardiac(2)-tCardiac(1);
%dtRespiration = tRespiration(2) - tRespiration(1);

isHigherSamplingCardiac = dtCardiac < dtRespiration;
if isHigherSamplingCardiac
    t = tCardiac;
    rInterp = interp1(tRespiration, r, t);
    %racq_codesInterp = interp1(tRespiration, racq_codes, t, 'nearest');
    %acq_codes = cacq_codes + racq_codesInterp;
    
    if DEBUG
        fh = plot_interpolation(tRespiration, r, t, rInterp, ...
            {'respiratory', 'cardiac'});
        verbose.fig_handles(end+1) = fh;
    end
    r = rInterp;
    
else
    t = tRespiration;
    cInterp = interp1(tCardiac, c, t);
    %cacq_codesInterp = interp1(tCardiac, cacq_codes, t, 'nearest');
    %acq_codes = racq_codes + cacq_codesInterp;
    
    if DEBUG
        fh = plot_interpolation(tCardiac, c, t, cInterp, ...
            {'cardiac', 'respiratory'});
        verbose.fig_handles(end+1) = fh;
    end
    c = cInterp;
    
end

%% Recompute acq_codes as for Siemens (volume on/volume off)
acq_codes = [];

if ~isempty(iAcqStart) % otherwise, nothing to read ...
    % iAcqStart is a columns of 0 and 1, 1 for the trigger event of a new
    % volume start
    
    % sometimes, trigger is on for several samples; ignore these extended
    % phases of "on-triggers" as duplicate values, if trigger distance is
    % very different
    %
    % fraction of mean trigger distance; if trigger time difference below that, will be removed
    outlierThreshold = 0.2;
    
    idxAcqStart = find(iAcqStart);
    dAcqStart = diff(idxAcqStart);
    
    % + 1 because of diff
    iAcqOutlier = 1 + find(dAcqStart < outlierThreshold*mean(dAcqStart));
    iAcqStart(idxAcqStart(iAcqOutlier)) = 0;
    
    acq_codes = zeros(nSamplesCardiac,1);
    acq_codes(iAcqStart) = 8; % to match Philips etc. format
    
    nAcqs = numel(find(iAcqStart));
    
    if nAcqs >= 1
        % report time of acquisition, as defined in SPM
        meanTR = mean(diff(t(iAcqStart)));
        stdTR = std(diff(t(iAcqStart)));
        verbose = tapas_physio_log(...
            sprintf('TR = %.3f +/- %.3f s (Estimated mean +/- std time of repetition for one volume; nTriggers = %d)', ...
            meanTR, stdTR, nAcqs), verbose, 0);
    end
end

%% Plot, if wanted

if DEBUG
    stringTitle = 'Read-In: Raw BIDS physlog data (TSV file)';
    verbose.fig_handles(end+1) = ...
        tapas_physio_plot_raw_physdata_siemens_hcp(t, c, r, acq_codes, ...
        stringTitle);
end

end

%% Function for unified physio file (cardiac + resp)
function [c,r,t,cpulse,acq_codes] = tapas_physio_read_physlogfiles_bids_unified(fileName,log_files,hasExplicitJsonFile,DEBUG,verbose)
%% check for .gz files and unzip to temporary file
[~, ~, ext] = fileparts(fileName);

isZipped = strcmpi(ext, '.gz');

if isZipped
    fileJson = regexprep(fileName, '\.tsv\.gz', '\.json');
    tempFilePath = tempname;  % tempname is matlab inbuilt
    fileName  = gunzip(fileName, tempFilePath);
    fileName = fileName{1};
else
    fileJson = regexprep(fileName, '\.tsv', '\.json');
end

if hasExplicitJsonFile
    fileJson = log_files.scan_timing;
end

hasJsonFile = isfile(fileJson);

if hasJsonFile
    val = jsondecode(fileread(fileJson));
else
    verbose = tapas_physio_log(...
        ['No .json file found. Please specify log_files.sampling_interval' ...
        ' and log_files.relative_start_acquisition explicitly.'], verbose, 1);
end

dt = log_files.sampling_interval;
if isempty(dt)
    if hasJsonFile
        dt = 1/val.SamplingFrequency;
    else
        verbose = tapas_physio_log(...
            ['No .json file found and empty log_files.sampling_interval. ' ...
            'Please specify explicitly.'], verbose, 2);
    end
end

% sum implicit (.json) and explicit relative shifts of log/scan acquisition
if isempty(log_files.relative_start_acquisition)
    if hasJsonFile
        % in BIDS, start of the phys logging is stated relative to the first volume scan start.
        % PhysIO defines the scan acquisiton relative to the phys log start
        tRelStartScan = -val.StartTime;
    else
        verbose = tapas_physio_log(...
            ['No .json file found and empty log_files.relative_start_acquisition. ' ...
            'Please specify explicitly.'], verbose, 2);
    end
else
    if hasJsonFile
        % add both delays
        tRelStartScan = log_files.relative_start_acquisition - val.StartTime;
    else
        tRelStartScan = log_files.relative_start_acquisition;
    end
end

% default columns in text file for phys recordings; overruled by JSON file
% 1 = cardiac, 2 = resp, 3 = trigger
bidsColumnNames = {'cardiac', 'respiratory', 'trigger'};
idxCol = 1:3;  %set default values for columns from BIDS
for iCol = 1:3
    if hasJsonFile
        idxCurrCol = find(cellfun(@(x) isequal(lower(x), bidsColumnNames{iCol}), val.Columns));
        if ~isempty(idxCurrCol)
            idxCol(iCol) = idxCurrCol;
        end
    end
end

C = tapas_physio_read_columnar_textfiles(fileName, 'BIDS');
c = double(C{idxCol(1)});
r = double(C{idxCol(2)});
iAcqStart = (double(C{idxCol(3)})~=0); % trigger has 1, rest is 0;


%% Create timing vector from samples

nSamples = max(numel(c), numel(r));
t = -tRelStartScan + ((0:(nSamples-1))*dt)';

%% Recompute acq_codes as for Siemens (volume on/volume off)
acq_codes = [];

if ~isempty(iAcqStart) % otherwise, nothing to read ...
    % iAcqStart is a columns of 0 and 1, 1 for the trigger event of a new
    % volume start
    
    % sometimes, trigger is on for several samples; ignore these extended
    % phases of "on-triggers" as duplicate values, if trigger distance is
    % very different
    %
    % fraction of mean trigger distance; if trigger time difference below that, will be removed
    outlierThreshold = 0.2;
    
    idxAcqStart = find(iAcqStart);
    dAcqStart = diff(idxAcqStart);
    
    % + 1 because of diff
    iAcqOutlier = 1 + find(dAcqStart < outlierThreshold*mean(dAcqStart));
    iAcqStart(idxAcqStart(iAcqOutlier)) = 0;
    
    acq_codes = zeros(nSamples,1);
    acq_codes(iAcqStart) = 8; % to match Philips etc. format
    
    nAcqs = numel(find(iAcqStart));
    
    if nAcqs >= 1
        % report time of acquisition, as defined in SPM
        meanTR = mean(diff(t(iAcqStart)));
        stdTR = std(diff(t(iAcqStart)));
        verbose = tapas_physio_log(...
            sprintf('TR = %.3f +/- %.3f s (Estimated mean +/- std time of repetition for one volume; nTriggers = %d)', ...
            meanTR, stdTR, nAcqs), verbose, 0);
    end
end

%% Plot, if wanted

if DEBUG
    stringTitle = 'Read-In: Raw BIDS physlog data (TSV file)';
    verbose.fig_handles(end+1) = ...
        tapas_physio_plot_raw_physdata_siemens_hcp(t, c, r, acq_codes, ...
        stringTitle);
end


%% occasionally, cardiac time course is instead containing 0/1 cardiac triggers,
% and not raw trace; check this and populate cpulse accordingly
if all(ismember(unique(c), [1 0]))
    cpulse = t(c==1);
else
    cpulse = [];
end

%% delete temporary unzipped file
if isZipped
    [status,message,messageId] = rmdir(tempFilePath, 's');
    % warning if deletion failed
    if status == 0
        tapas_physio_log(sprintf('%s: %s', messageId, message), verbose, 1)
    end
end

end

%% Local function to plot interpolation result
function fh = plot_interpolation(tOrig, yOrig, tInterp, yInterp, ...
    stringOrigInterp)
fh = tapas_physio_get_default_fig_params();
stringTitle = sprintf('Read-In: Interpolation of %s signal', stringOrigInterp{1});
set(fh, 'Name', stringTitle);
plot(tOrig, yOrig, 'go--');  hold all;
plot(tInterp, yInterp,'r.');
legend({
    sprintf('after interpolation to %s timing', ...
    stringOrigInterp{1}), ...
    sprintf('original %s time series', stringOrigInterp{2}) });
title(stringTitle);
xlabel('time (seconds');
end

