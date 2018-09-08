function [C, columnNames] = tapas_physio_read_files_siemens_tics(fileName, fileType)
% Reads _PULS, _RESP, _ECG, _Info-files from Siemens tics format with
% multiple numbers of columns and different column headers
%
%   output = tapas_physio_read_files_siemens_tics(input)
%
% IN
%   fileName    *.log from Siemens VD/VE tics file format
%   fileType    'ECG', 'PULS', 'RESP', 'Info'
%               If not specified, this is read from the last part of the
%               filename after the last underscore, e.g.
%               Physio_*_ECG.log -> log
% OUT
%   C           cell(1, nColumns) of cells(nRows,1) of values
%   columnNames cell(1, nColumns) of column names
%
% EXAMPLE
%   tapas_physio_read_files_siemens_tics('Physio_RESP.log', 'RESP')
%   % equivalent to (since file name unique!)
%   tapas_physio_read_files_siemens_tics('Physio_RESP.log')
%
%   See also
%
% Author: Lars Kasper
% Created: 2017-11-16
% Copyright (C) 2017 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% License (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: teditRETRO.m 775 2015-07-17 10:52:58Z kasperla $

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


switch fileType
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
haveFoundColumnHeader = false;
nHeaderLines = 0;
while ~haveFoundColumnHeader
    nHeaderLines = nHeaderLines + 1;
    strLine = fgets(fid);
    haveFoundColumnHeader = any(regexp(upper(strLine), strColumnHeader));
end

columnNames = regexp(strLine, '([\w]*)', 'tokens'); 
columnNames = [columnNames{:}]; % cell of cell into cell of strings
fclose(fid);

nColumns = numel(regexp(strLine, ' *')) + 1; % columns are separated by arbitrary number of spaces
nHeaderLines = nHeaderLines + nEmptyLinesAfterHeader(nColumns); % since empty line after header for CMRR files (not in Cologne!)

fid = fopen(fileName);

switch fileType
    case 'INFO'
        C = textscan(fid, parsePatternPerNColumns{nColumns}, 'HeaderLines', nHeaderLines);
    case {'PULS', 'RESP'}
        C = textscan(fid, parsePatternPerNColumns{nColumns}, 'HeaderLines', nHeaderLines);
    case {'ECG'}
        C = textscan(fid, parsePatternPerNColumns{nColumns}, 'HeaderLines', nHeaderLines);
        if nColumns == 4 % CMRR, different ECG channels possible!
        end
end
