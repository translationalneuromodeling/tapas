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
        defaults.ecgChannel = 'mean'; %'mean'; 'v1'; 'v2'
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

hasScanTimingDicomImage = ~isempty(log_files.scan_timing);
hasCardiacData = ~isempty(log_files.cardiac);
hasRespData = ~isempty(log_files.respiration);

if hasScanTimingDicomImage
    dicomHeader             = spm_dicom_headers(...
        fullfile(log_files.scan_timing));
    
    tStartScanDicom    = dicomHeader{1}.AcquisitionTime;
    
    % TODO: Include AcquisitionNumber? InstanceNumber?
    tStopScanDicom     = dicomHeader{1}.AcquisitionTime + ...
        dicomHeader{1}.RepetitionTime/1000;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Read in cardiac data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if hasCardiacData
    
    [lineData, logFooter] = tapas_physio_read_physlogfiles_siemens_raw(...
        log_files.cardiac);
    tLogTotal = logFooter.LogStopTimeSeconds - logFooter.LogStartTimeSeconds;
    
    
    if hasScanTimingDicomImage
        tStartScan = tStartScanDicom; % this is just the start of the selected DICOM volume
        tStopScan = tStopScanDicom + sqpar.TR; % is incorrect, i.e., equals start of volume, therefore use tStartScan + TR!
    else
        % Just different time scale, gives bad scaling in plots, and not
        % needed...
        %     tStartScan = logFooter.ScanStartTimeSeconds;
        %     tStopScan = logFooter.ScanStopTimeSeconds;
        tStartScan = logFooter.LogStartTimeSeconds;
        tStopScan = logFooter.LogStopTimeSeconds;
    end
    
    switch log_files.align_scan
        case 'first'
            relative_start_acquisition = tStartScan ...
                - logFooter.LogStartTimeSeconds;
        case 'last'
            % shift onset of first scan by knowledge of run duration and
            % onset of last scan in run
            relative_start_acquisition = ...
                (tStopScan - sqpar.Nscans*sqpar.TR) ... 
                - logFooter.LogStartTimeSeconds;
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
        log_files.respiration);
    tLogTotal = logFooter.LogStopTimeSeconds - logFooter.LogStartTimeSeconds;
    
    if hasScanTimingDicomImage
        tStartScan = tStartScanDicom;
        tStopScan = tStopScanDicom; % is incorrect, use tStartScan + TR!
    else
        % Just different time scale, gives bad scaling in plots, and not
        % needed...
        %     tStartScan = logFooter.ScanStartTimeSeconds;
        %     tStopScan = logFooter.ScanStopTimeSeconds;
        tStartScan = logFooter.LogStartTimeSeconds;
        tStopScan = logFooter.LogStopTimeSeconds; 
    end
    
    switch log_files.align_scan
        case 'first'
            relative_start_acquisition = tStartScan - ...
                logFooter.LogStartTimeSeconds;
        case 'last'
            % shift onset of first scan by knowledge of run duration and
            % onset of last scan in run
            relative_start_acquisition = ...
                (tStartScan - (sqpar.Nscans-1)*sqpar.TR) ... 
                - logFooter.LogStartTimeSeconds;
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
