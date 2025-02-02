function [c, r, t, cpulse, verbose] = ...
    tapas_physio_read_physlogfiles_siemens(log_files, cardiac_modality, verbose, varargin)
% reads out physiological time series and timing vector for Siemens
% logfiles of peripheral cardiac monitoring (ECG/Breathing Belt or
% pulse oximetry)
%
%   [cpulse, rpulse, t, c] = tapas_physio_read_physlogfiles_siemens(...
%           logfile, vendor, cardiac_modality)
%
% IN    log_files
%       .log_cardiac        contains ECG or pulse oximeter time course
%                           for GE: ECGData...
%       .log_respiration    contains breathing belt amplitude time course
%                           for GE: RespData...
%       cardiac_modality    'ECG' or 'PULS'/'PPU'/'OXY' to determine
%                           which channel data to be returned
%                           if not given, will be read out from file name
%                           suffix
%       verbose
%       .level              debugging plots are created if level >=3
%       .fig_handles        appended by handle to output figure
%
%       varargin            propertyName/value pairs, as folloes
%           'ecgChannel'    'v1', 'v2', 'mean' (default)
%                           determines which ECG channel to use as
%                           output cardiac curve
%           'sqpar'         sqpar, needed for scan_align to 'last' volume
%                           of a run
%
% OUT
%   cpulse              time events of R-wave peak in cardiac time series (seconds)
%                       for GE: usually empty
%   r                   respiratory time series
%   t                   vector of time points (in seconds)
%                       NOTE: This assumes the default sampling rate of 40
%                       Hz
%   c                   cardiac time series (ECG or pulse oximetry)
%
% EXAMPLE
%   [ons_secs.cpulse, ons_secs.rpulse, ons_secs.t, ons_secs.c] =
%       tapas_physio_read_physlogfiles_siemens(logfile, vendor, cardiac_modality);
%
%   See also tapas_physio_main_create_regressors

% Author: Lars Kasper
%         file structure information from PhLeM Toolbox, T. Verstynen (November 2007);
%                and Deshpande and J. Grinstead, Siemens Medical Solutions (March 2009)
%         additional log information Miriam Sebold, Charite Berlin (2014)
%
% Created: 2014-07-08
% Copyright (C) 2014 Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

%% read out values

if nargin < 3
    verbose.level = 0;
end
DEBUG = verbose.level >=2;

% process optional input parameters and overwrite defaults
defaults.sqpar = [];
defaults.endCropSeconds     = 1;
% used channel depends on cardiac modality
switch cardiac_modality
    case 'ECG'
        defaults.ecgChannel = 'mean'; %'mean'; 'v1'; 'v2'; 'v3'; 'v4'
    otherwise
        defaults.ecgChannel = 'v1';
end

args = tapas_physio_propval(varargin, defaults);
tapas_physio_strip_fields(args);


cpulse              = [];

dt                  = log_files.sampling_interval;

explicit_relative_start_acquisition = log_files.relative_start_acquisition;

if isempty(explicit_relative_start_acquisition)
    explicit_relative_start_acquisition = 0;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Determine relative start of acquisition from dicom headers and
% logfile footers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

hasScanTimingFile = ~isempty(log_files.scan_timing);
hasCardiacData = ~isempty(log_files.cardiac);
hasRespData = ~isempty(log_files.respiration);

if hasScanTimingFile
    % Times are in seconds since start of the day

    [~,~, fileType] = fileparts(log_files.scan_timing);

    switch lower(fileType)
        case '.json'
            % Assuming that AcquisitionDateTime of first slice of first volume Dicom is
            % converted by dcm2niix to AcquisitionTime field, see
            % https://github.com/UNFmontreal/Dcm2Bids/issues/90#issuecomment-708655328
            if strcmpi(log_files.align_scan, 'last')
                verbose = tapas_physio_log(['Inconsistency: ' ...
                    'AcquisitionTime in scan_timing JSON side car typically ' ...
                    'refers to first slice of first volume, but ' ...
                    'align_scan = ''last'' is specified. ' ...
                    'Set align_scan = ''first'' instead'], verbose, 2);
            end
            val = jsondecode(fileread(log_files.scan_timing));
            dateVector = datevec(val.AcquisitionTime, 'HH:MM:SS.FFF');
            tStartScanImageHeader = dateVector(end-2:end)*[3600 60 1]';
            TR = val.RepetitionTime;

        otherwise % assume and try DICOM file

            dicomHeader             = spm_dicom_headers(...
                fullfile(log_files.scan_timing));

            if isempty(dicomHeader) % not a DICOM file
                verbose = tapas_physio_log(['Unknown file type for scan ' ...
                    'timing synchronization (expected DICOM .dcm/.ima or BIDS .json)'], ...
                    verbose, 2);
            else
                try % old example DICOM format
                    tStartScanImageHeader    = dicomHeader{1}.AcquisitionTime;

                    TR = dicomHeader{1}.RepetitionTime/1000;

                    % TODO: Include AcquisitionNumber? InstanceNumber?

                catch % new XA30 DICOM export
                    dc = dicomHeader{1};

                    % tried different fields, Study/SeriesTime refer to 1st volume
                    % ContentTime > InstanceCreationTime > dc.AcquisitionDateTime
                    % difference is about one second each
                    % (1.14 and 0.89s for vol 250, 1.35 and 1.97s for vol 001, TR was 1.2s)
                    % Since naming was most similar to AcquisitionDateTime, we chose
                    % that one...
                    % parse Date/Time format from Dicom field, and reformat it as
                    % seconds since start of day
                    dateVector = datevec(dc.AcquisitionDateTime, 'yyyymmddHHMMSS.FFF');
                    tStartScanImageHeader = dateVector(end-2:end)*[3600 60 1]';
                    TR = dc.SharedFunctionalGroupsSequence{1}.MRTimingAndRelatedParametersSequence{1}.RepetitionTime/1000;
                end
           end
    end

    tStopScanImageHeader     = tStartScanImageHeader + TR;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Read in cardiac data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
referenceClockString = 'MDH'; %'MPCU';or 'MDH' (default)
if hasCardiacData

    [lineData, logFooter] = tapas_physio_read_physlogfiles_siemens_raw(...
        log_files.cardiac, referenceClockString);
    tLogTotal = logFooter.StopTimeSeconds - logFooter.StartTimeSeconds;


    if hasScanTimingFile
        tStartScan = tStartScanImageHeader; % this is the start of the DICOM volume selected for sync
        tStopScan = tStopScanImageHeader;   % this is the end time (start + TR) of the DICOM volume selected for sync
    else
        tStartScan = logFooter.StartTimeSeconds;
        tStopScan = logFooter.StopTimeSeconds;
    end

    switch log_files.align_scan
        case 'first'
            relative_start_acquisition = tStartScan ...
                - logFooter.StartTimeSeconds;
        case 'last'
            % shift onset of first scan by knowledge of run duration and
            % onset of last scan in run
            relative_start_acquisition = ...
                (tStopScan - sqpar.Nscans*sqpar.TR) ...
                - logFooter.StartTimeSeconds;
    end


    % add arbitrary offset specified by user
    relative_start_acquisition = relative_start_acquisition + ...
        explicit_relative_start_acquisition;

    data_table = tapas_physio_siemens_line2table(lineData, cardiac_modality);

    if isempty(dt)
        nSamplesC = size(data_table,1);
        dt_c = tLogTotal/(nSamplesC-1);
    else
        dt_c = dt(1);
    end

    dataCardiac = tapas_physio_siemens_table2cardiac(data_table, ecgChannel, dt_c, ...
        relative_start_acquisition, endCropSeconds);

    if DEBUG
        verbose.fig_handles(end+1) = ...
            tapas_physio_plot_raw_physdata_siemens(dataCardiac);
    end


    %% crop end of log file
    cpulse = dataCardiac.cpulse_on;
    c = dataCardiac.c;
    t_c = dataCardiac.t;
    stopSample = dataCardiac.stopSample;

    cpulse(cpulse > t_c(dataCardiac.stopSample)) = [];
    t_c(stopSample+1:end) = [];
    c(stopSample+1:end) = [];



else
    c = [];
    t_c = [];
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Read in respiratory data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if hasRespData
    [lineData, logFooter] = tapas_physio_read_physlogfiles_siemens_raw(...
        log_files.respiration, referenceClockString);
    tLogTotal = logFooter.StopTimeSeconds - logFooter.StartTimeSeconds;

    if hasScanTimingFile
        tStartScan = tStartScanImageHeader; % this is the start of the DICOM volume selected for sync
        tStopScan = tStopScanImageHeader;   % this is the end time (start + TR) of the DICOM volume selected for sync
    else
        tStartScan = logFooter.StartTimeSeconds;
        tStopScan = logFooter.StopTimeSeconds;
    end

    switch log_files.align_scan
        case 'first'
            relative_start_acquisition = tStartScan - ...
                logFooter.StartTimeSeconds;
        case 'last'
            % shift onset of first scan by knowledge of run duration and
            % onset of last scan in run
            relative_start_acquisition = ...
                (tStopScan - sqpar.Nscans*sqpar.TR) ...
                - logFooter.StartTimeSeconds;
    end


    % add arbitrary offset specified by user
    relative_start_acquisition = relative_start_acquisition + ...
        explicit_relative_start_acquisition;

    data_table = tapas_physio_siemens_line2table(lineData, 'RESP');

    if isempty(dt)
        nSamplesR = size(data_table,1);
        dt_r = tLogTotal/(nSamplesR-1);
    else
        dt_r = dt(end);
    end

    dataResp = tapas_physio_siemens_table2cardiac(data_table, ecgChannel, ...
        dt_r, relative_start_acquisition, endCropSeconds);

    if DEBUG
        verbose.fig_handles(end+1) = ...
            tapas_physio_plot_raw_physdata_siemens(dataResp);
    end


    r = dataResp.c;
    t_r = dataResp.t;

    %
    %% crop end of log file???
    % stopSample = dataResp.stopSample;
    % t(stopSample+1:end) = [];
    % c(stopSample+1:end) = [];


else
    r = [];
    t_r = [];
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Adapt time scales resp/cardiac, i.e. zero fill c or r
% to get equal length with max(c_t, r_t)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nSamplesR = numel(r);
nSamplesC = numel(c);

if  nSamplesR > nSamplesC
    t = t_r; % time scale defined by respiration now, since longer
    if nSamplesC > 0 % zero-fill cardiac data
        c(nSamplesC+1:nSamplesR) = 0;
    end
else
    t = t_c;
    if nSamplesR > 0
        r(nSamplesR+1:nSamplesC) = 0;
    end
end


end
