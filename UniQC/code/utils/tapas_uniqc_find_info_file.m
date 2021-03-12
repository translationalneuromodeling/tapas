function [fileNameArrayMatch, fileNameArrayResiduals] = ...
    tapas_uniqc_find_info_file(fileNameArray, matchString)
% Searches a fileNameArray for uniqc file names except for matchString
%
%   output = tapas_uniqc_find_info_file(fileNameArray, matchString)
%   dimInfoFiles = tapas_uniqc_find_info_file(fileNameArray, '_dimInfo.mat');
%
% IN
%
% OUT
%
% EXAMPLE
%   tapas_uniqc_find_info_file
%
%   See also

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2018-08-30
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

% find all files that contain the matchString
isMatch = contains(fileNameArray, matchString);
% only continue if match has been found
if ~any(isMatch)
    fileNameArrayMatch = {};
    fileNameArrayResiduals = fileNameArray;
else
    
    fileNameArrayFound = fileNameArray(isMatch);
    fileNameArrayNotFound = fileNameArray(~isMatch);
    % remove match string
    searchFileNames = regexprep(fileNameArrayFound, ['\', matchString,'$'], '');
    % search for matching filenames in search array
    hasMatchingFile = contains(fileNameArrayNotFound, searchFileNames);
    % keep only the files with matching files, all other go into residuals
    fileNameArrayMatch = fileNameArrayFound(hasMatchingFile);
    fileNameArrayResiduals = [fileNameArrayNotFound, ...
        fileNameArrayFound(~hasMatchingFile)];
end
end