function y = tapas_physio_read_physlogfiles_philips_matrix(fileNamePhyslog)
% reads out complete numerical matrix from SCANPHYSLOG-file
% Physlog file version 1: [nSamples,10]
% Physlog file version 2: [nSamples,11]
%
% IN
%   fileNamePhyslog 
%           string, filename of SCANPHYSLOG-file
% OUT
%   y       [nSamples, 10 or 11] SCANPHYSLOG-file matrix with the following
%           columns:
%           Version 1: 
%           v1raw v2raw v2 v2 PPU Pneumatic_belt Gradient_x Gradient_y
%           Gradient_z acq_codes(scan/ECG trigger etc.)
%           Version 2:
%           v1raw v2raw  v1 v2  ppu resp  gx gy gz mark mark2
%
% EXAMPLE
%   y = tapas_physio_read_physlogfiles_philips_matrix('SCANPHYSLOG.log')
%
%   See also tapas_physio_read_physlogfiles_philips
%   See also tapas_physio_create_scan_timing_from_gradients_philips
%   See also tapas_physio_create_scan_timing_from_gradients_auto_philips

% Author: Lars Kasper
% Created: 2015-01-11
% Copyright (C) 2015, Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.



% use textread as long as it exists, for it is much faster (factor 4) than
% textscan; TODO: use fread and sscanf to make it even faster...

%% check log file version by header line
if ~exist(fileNamePhyslog, 'file')
    error('Physlog file not found: %s', fileNamePhyslog);
end

fid  = fopen(fileNamePhyslog, 'r');
stringFirstLine = fgetl(fid);
fclose(fid);

if any(strfind(stringFirstLine, 'Physlog file version = 2'))
    versionLogfile = 2;
else
    versionLogfile = 1;
end

switch versionLogfile
    case 1
        parseString = '%d %d %d %d %d %d %d %d %d %s';
        nColumns = 10;
    case 2
        parseString = '%d %d %d %d %d %d %d %d %d %s %s';
        nColumns = 11;
end


%% Read decimal value colums
if exist('textread')
    [z{1:nColumns}]   = textread(fileNamePhyslog, parseString,'commentstyle', 'shell');
else
    fid     = fopen(fileNamePhyslog, 'r');
    z       = textscan(fid, parseString, 'commentstyle', '#');
    z(1:9)  = cellfun(@double, z(1:9), 'UniformOutput', false);
    fclose(fid);
end


%% Convert hexadecimal acquisition codes
for iCol = 10:nColumns
    z{iCol}       = hex2dec(z{iCol}); 
end


%% Account for incomplete rows
nMinRows    = min(cellfun(@numel,z));
z           = cellfun(@(x) x(1:nMinRows), z, 'UniformOutput', false);
y           = cell2mat(z);
