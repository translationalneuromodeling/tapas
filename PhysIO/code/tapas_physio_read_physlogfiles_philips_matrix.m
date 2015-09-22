function y = tapas_physio_read_physlogfiles_philips_matrix(fileNamePhyslog)
% reads out complete [nSamples,10] numerical matrix from SCANPHYSLOG-file
% IN
%   fileNamePhyslog 
%           string, filename of SCANPHYSLOG-file
% OUT
%   y       [nSamples, 10] SCANPHYSLOG-file matrix with the following
%           columns:
%           v1raw v2raw v2 v2 PPU Pneumatic_belt Gradient_x Gradient_y
%           Gradient_z acq_codes(scan/ECG trigger etc.)
%
% EXAMPLE
%   y = tapas_physio_read_physlogfiles_philips_matrix('SCANPHYSLOG.log')
%
%   See also tapas_physio_read_physlogfiles_philips
%   See also tapas_physio_create_scan_timing_from_gradients_philips
%   See also tapas_physio_create_scan_timing_from_gradients_auto_philips
%
% Author: Lars Kasper
% Created: 2015-01-11
% Copyright (C) 2015, Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_read_physlogfiles_philips_matrix.m 632 2015-01-09 12:36:12Z kasperla $

% use textread as long as it exists, for it is much faster (factor 4) than
% textscan; TODO: use fread ans sscanf to make it even faster...
if exist('textread')
    [z{1:10}]   = textread(fileNamePhyslog,'%d %d %d %d %d %d %d %d %d %s','commentstyle', 'shell');
else
    fid     = fopen(fileNamePhyslog, 'r');
    z       = textscan(fid, '%d %d %d %d %d %d %d %d %d %s', 'commentstyle', '#');
    z(1:9)  = cellfun(@double, z(1:9), 'UniformOutput', false);
    fclose(fid);
end

z{10}       = hex2dec(z{10}); % hexadecimal acquisition codes converted;

% account for incomplete rows
nMinRows    = min(cellfun(@numel,z));
z           = cellfun(@(x) x(1:nMinRows), z, 'UniformOutput', false);
y           = cell2mat(z);
