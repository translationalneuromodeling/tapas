function [C, columnNames] = tapas_physio_read_columnar_textfiles(fileName, fileType, nColumns)
% Reads _PULS, _RESP, _ECG, _Info-files from Siemens tics format with
% multiple numbers of columns and different column headers
%
%   [C, columnNames] = tapas_physio_read_columnar_textfiles(fileName, fileType)
%
% IN
%   fileName    *.log from Siemens VD/VE tics file format
%   fileType    'ECG', 'PULS', 'RESP', 'Info', 'BIOPAC_TXT', 'PHILIPS'
%               If not specified, this is read from the last part of the
%               filename after the last underscore, e.g.
%               Physio_*_ECG.log -> log
%   nColumns    optional, number of columns in columnar logfile
%               if not specified, this function tries to estimate from the
%               column headers how many columns exist (default: [], will be
%               estimated from header)
%
% OUT
%   C           cell(1, nColumns) of cells(nRows,1) of values
%   columnNames cell(1, nColumns) of column names
%
% EXAMPLE
%   tapas_physio_read_columnar_textfiles('Physio_RESP.log', 'RESP')
%   % equivalent to (since file name unique!)
%   tapas_physio_read_columnar_textfiles('Physio_RESP.log')
%
%   See also

% Author: Lars Kasper
% Created: 2017-11-16
% Copyright (C) 2017 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% License (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

if nargin < 3
    nColumns = [];
end

%_AcquisitionInfo

if nargin < 2
    % extract what is between last underscore and extension .log in
    % filename:
    fileType = regexp(fileName, '.*_([^_]*)\.log', 'tokens');
    if isempty(fileType)
        fileType = 'ECG';
    else
        fileType = upper(fileType{1}{1});
    end
    
    % to also cover other suffixes, e.g. AcquisitionInfo, ScanInfo, rename
    % all to INFO format name
    isInfoFile = any(strfind(fileType, 'INFO'));
    if isInfoFile
        fileType = 'INFO';
    end
        
end


switch upper(fileType)
    case {'ADINSTRUMENTS_TXT', 'LABCHART_TXT'}
        strColumnHeader = 'ChannelTitle=';

        %sub-02 from CUBRIC
        parsePatternPerNColumns{4} = '%f %f %f %f';
        nEmptyLinesAfterHeader(4) = 1;

        %sub-01 from CUBRIC
        parsePatternPerNColumns{7} = '%f %f %f %f %f %f %f';
        nEmptyLinesAfterHeader(7) = 4;
    case 'BIDS'
        strColumnHeader = '';
        
        if isempty(nColumns)
            nColumns = 3;
        end

        parsePatternPerNColumns{nColumns} = repmat('%f ', 1, nColumns);
        parsePatternPerNColumns{nColumns}(end) = [];
        nEmptyLinesAfterHeader(nColumns) = 0;
    case 'BIOPAC_TXT'
        strColumnHeader = '.*RESP.*';
        parsePatternPerNColumns{4} = '%f %f %f %d';
        nEmptyLinesAfterHeader(4) = 0;
    case 'INFO' % header is a subset of
        % Cologne:
        %   Volume_ID Slice_ID AcqTime_Tics
        % CMRR: (ECHO column optional!)
        %   VOLUME   SLICE   ACQ_START_TICS  ACQ_FINISH_TICS  ECHO
        strColumnHeader = 'VOLUME.* *SLICE';
        
        % depending on the number of columns, value type per column and
        % number of empty lines after header differ, grouped here by
        % entries in iColumn of respective variables
        parsePatternPerNColumns{3} = '%d %d %d';
        parsePatternPerNColumns{4} = '%d %d %d %d';
        parsePatternPerNColumns{5} = '%d %d %d %d %d';
        nEmptyLinesAfterHeader(3) = 0;
        nEmptyLinesAfterHeader(4) = 1;
        nEmptyLinesAfterHeader(5) = 1;
    case 'PHILIPS'
        strColumnHeader = 'v1raw'; % Philips header similar to: # v1raw v2raw  v1 v2  ppu resp vsc  gx gy gz mark mark2
        parsePatternPerNColumns{10} = '%d %d %d %d %d %d %d %d %d %s';
        parsePatternPerNColumns{11} = '%d %d %d %d %d %d %d %d %d %s %s'; % log file version 2 with two marker columns
        parsePatternPerNColumns{12} = '%d %d %d %d %d %d %d %d %d %d %s %s'; %  logfile version 3(?, from Nottingham) with 'vsc' column
        nEmptyLinesAfterHeader(10) = 0;
        nEmptyLinesAfterHeader(11) = 0;
        nEmptyLinesAfterHeader(12) = 0;        
    case {'PULS', 'RESP', 'ECG'} % have similar format
        % Cologne (RESP/PULS/ECG for 2nd column):
        %   Time_tics RESP Signal
        % or CMRR:
        %   ACQ_TIME_TICS  CHANNEL  VALUE  SIGNAL
        strColumnHeader = 'SIGNAL';
        parsePatternPerNColumns{3} = '%d %d %d';
        parsePatternPerNColumns{4} = '%d %s %d %s %s'; % needs 5 columns to address simultaneous pulse/ext trigger entries
        parsePatternPerNColumns{5} = '%d %s %d %s %s %s'; %  needs 6 columns to address simultaneous pulse/ext trigger entries
        nEmptyLinesAfterHeader(3) = 0;
        nEmptyLinesAfterHeader(4) = 1;
        nEmptyLinesAfterHeader(5) = 1;
end


fid = fopen(fileName);

% Determine number of header lines by searching for the column header line,
% which has both Volume and Slice as a keyword in it
haveFoundColumnHeader = isempty(strColumnHeader); % for empty column header search string, don't search (e.g. BIDS no column header)
nHeaderLines = 0;
while ~haveFoundColumnHeader
    nHeaderLines = nHeaderLines + 1;
    strLine = fgets(fid);
    haveFoundColumnHeader = any(regexpi(strLine, strColumnHeader));
end

switch upper(fileType)
    case {'ADINSTRUMENTS_TXT', 'LABCHART_TXT'}
        columnNames = regexp(strLine, '([\t])', 'split');
        nColumns = numel(columnNames);
    case 'BIDS' % will be in separate json-file
        columnNames = {};
    case 'BIOPAC_TXT' % bad column names with spaces...e.g. 'RESP - RSP100C'
        columnNames = regexp(strLine, '([\t])', 'split');
        nColumns = numel(columnNames);
    case 'PHILIPS'
        columnNames = regexp(strLine, '([\w]*)', 'tokens');
        columnNames = [columnNames{:}]; % cell of cell into cell of strings
        nColumns = numel(columnNames);
    otherwise
        columnNames = regexp(strLine, '([\w]*)', 'tokens');
        columnNames = [columnNames{:}]; % cell of cell into cell of strings
        nColumns = numel(regexp(strLine, ' *')) + 1; % columns are separated by arbitrary number of spaces...TODO: Why + 1?
end
fclose(fid);

nHeaderLines = nHeaderLines + nEmptyLinesAfterHeader(nColumns); % since empty line after header for CMRR files (not in Cologne!)

% now read the rest of the file
fid = fopen(fileName);
switch upper(fileType)
    case {'ADINSTRUMENTS_TXT', 'LABCHART_TXT', 'BIDS', 'BIOPAC_TXT', 'INFO', 'PULS', 'RESP'}
        C = textscan(fid, parsePatternPerNColumns{nColumns}, 'HeaderLines', nHeaderLines);
    case 'PHILIPS' % sometimes odd lines with single # occur within text file
        C = textscan(fid, parsePatternPerNColumns{nColumns}, 'HeaderLines', nHeaderLines, 'CommentStyle', '#');
    case {'ECG'}
        C = textscan(fid, parsePatternPerNColumns{nColumns}, 'HeaderLines', nHeaderLines);
        if nColumns == 4 % CMRR, different ECG channels possible!
        end
end
fclose(fid);
