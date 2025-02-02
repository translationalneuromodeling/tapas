function [lineData, logFooter, linesFooter] = tapas_physio_read_physlogfiles_siemens_raw(...
    fileNameLog, referenceClockString)
% Read in raw data/footer lines of logfiles, without data selection/sorting
%
% [lineData, linesFooter] = tapas_physio_read_physlogfiles_siemens_raw(...
%        fileNameLog, referenceClockString)
%
% IN
%   fileNameLog     file name of physiological log, e.g. *.ecg
%   referenceClockString
%                   'MDH' (scanner, default) or 'MPCU' (physiological
%                   monitoring unit
% OUT
%   lineData        all recording amplitude samples are saved in first line
%                   of file
%   logFooter       struct() of read-out meta information from log file, i.e. 
%                   LogStart/StopTimeSeconds
%                   ScanStart/StopTimeSeconds
%                   => uses MDH time stamp in log file to sync to DICOMs
%   linesFooter     all meta-information (e.g. sampling start/stop) is
%                   saved in remaining lines of log file
%
% EXAMPLE
%   tapas_physio_read_physlogfiles_siemens_raw
%
%   See also

% Author: Lars Kasper
% Created: 2016-02-29
% Copyright (C) 2016 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% License (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

if nargin < 2
    referenceClockString = 'MDH';
end

fid             = fopen(fileNameLog);

if verLessThan('matlab', '8.5') % use buffer size for speed-up, as long as it exists
    C               = textscan(fid, '%s', 'Delimiter', '\n', 'bufsize', 1e9);
else
    C               = textscan(fid, '%s', 'Delimiter', '\n');
end

fclose(fid);

linesFooter = C{1}(2:end);
lineData = C{1}{1};

% Get time stamps from footer:

% MPCU  = Computer who controls the physiological logging in real-time => physio logging happens here  
% MDH   = Computer who is the host (Measurement Data Header); console => DICOM time stamp here
logFooter.StartTimeSecondsScannerClock =   str2num(char(regexprep(linesFooter(~cellfun(@isempty,strfind(linesFooter,...
    'LogStartMDHTime'))),'\D',''))) / 1000;
logFooter.StopTimeSecondsScannerClock =    str2num(char(regexprep(linesFooter(~cellfun(@isempty,strfind(linesFooter,...
    'LogStopMDHTime'))),'\D',''))) / 1000;

logFooter.StartTimeSecondsRecordingClock = str2num(char(regexprep(linesFooter(~cellfun(@isempty,strfind(linesFooter,...
    'LogStartMPCUTime'))),'\D',''))) / 1000;
logFooter.StopTimeSecondsRecordingClock = str2num(char(regexprep(linesFooter(~cellfun(@isempty,strfind(linesFooter,...
    'LogStopMPCUTime'))),'\D',''))) / 1000;

%
% We use the time stamp of the clock of the Measurement Data Header (MDH),
% i.e., computer that controls the scanner, to synchronize with the DICOMs,
% because this computer also controls the creation of the scan data, i.e.,
% reconstructed DICOM images. This is in accordance to other packages
% reading Siemens physiological logfile data, e.g., Chris Rorden's PART
% (https://github.com/neurolabusc/Part#usage),
% with a detailed explanation on the DICOM timestamp in AcquisitionTime
% found here (https://github.com/nipy/heudiconv/issues/450#issuecomment-645003447) 
% 
% MPCU is the clock of the computer that controls the physiological
% recording (same as MARS?), but does not know about the scan volume and DICOM timing

switch upper(referenceClockString)
    case 'MDH'
logFooter.StartTimeSeconds = logFooter.StartTimeSecondsScannerClock;
logFooter.StopTimeSeconds = logFooter.StopTimeSecondsScannerClock;
    case 'MPCU'
logFooter.StartTimeSeconds = logFooter.StartTimeSecondsRecordingClock;
logFooter.StopTimeSeconds = logFooter.StopTimeSecondsRecordingClock;
end