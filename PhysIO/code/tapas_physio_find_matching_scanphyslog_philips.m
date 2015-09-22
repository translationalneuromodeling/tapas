function fnPhysLogArray = tapas_physio_find_matching_scanphyslog_philips(...
    fnImageArray, pathLogFiles)
%returns file name(s) of matching SCANPHYSLOG-file for a given .nii-file
%whose exported name is the original Philips name
%
%   fnPhyslogArray = tapas_physio_find_matching_scanphyslog_philips(...
%       fnImageArray, pathLogFiles))
%
% IN
%       fnImageArray    cell of image files, e.g. os_20062014_0904060_10_1_wipepitrig1mmtra150dynV42_typ0 
%       pathLogFiles    path with SCANPHYSLOG*.log
% 
% OUT
%       fnPhysLogArray  array of matching log files
%
% EXAMPLE
%   tapas_physio_find_matching_scanphyslog_philips
%
%   See also
%
% Author: Lars Kasper
% Created: 2014-06-19
% Copyright (C) 2014 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: teditRETRO.m 464 2014-04-27 11:58:09Z kasperla $
x = dir(fullfile(pathLogFiles, 'SCANPHYSLOG*'));
nPhys = length(x);
for i = 1:nPhys
    tPhys(i) = str2num(x(i).name(20:25));
end;

isCellfnImage = iscell(fnImageArray);

if isCellfnImage
    nFiles = length(fnImageArray);
else
    fnImageArray = {fnImageArray};
    nFiles = 1;
end


fnPhysLogArray = cell(nFiles,1);

for d = 1:nFiles
    tFun(d) = str2num(fnImageArray{d}(13:18));
    [tmp, iFun] = min(abs(tPhys-tFun(d)));
    fnPhysLogArray{d} = x(iFun).name;
    fprintf(1,'matched %s \n \t --> %s\n\n', fnImageArray{d}, ...
        fnPhysLogArray{d});
end

if ~isCellfnImage
    fnPhysLogArray = fnPhysLogArray{1};
end