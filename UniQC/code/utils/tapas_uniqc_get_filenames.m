function fileArray = tapas_uniqc_get_filenames(cellOrString)
% Returns a cell of filenames given a regular expression, folder name, or
% file prefix.
%
%   fileArray = tapas_uniqc_get_filenames(cellOrString)
%
%
% NOTE: if a cell of file names is given, a cell of the existing ones is
% returned; if a single file name is given, and it exists, it is returned
% within a cell. If it doesn't exist, an empty
%
% IN
%
% OUT
%   fileArray   string of files
%
% EXAMPLE
%   tapas_uniqc_get_filenames('funct_short.nii')
%       -> {'funct_short.nii'} is returned
%   tapas_uniqc_get_filenames('resting_state_ingenia_3T/')
%       -> {'funct_short.nii'; 'struct.nii'; 'meanfunct.nii'} is returned
%   isExact = 1;
%   tapas_uniqc_get_filenames('resting_state_ingenia_3T/f', isExact)
%       -> {} is returned
%   isExact = 0;
%   tapas_uniqc_get_filenames('resting_state_ingenia_3T/f', isExact)
%   tapas_uniqc_get_filenames('resting_state_ingenia_3T/f*')
%   tapas_uniqc_get_filenames('resting_state_ingenia_3T/f.*')
%       -> in all 3 cases, {'funct_short.nii'} is returned
%
%   See also MrDataNd.load

% Author:   Lars Kasper
% Created:  2016-10-21
% Copyright (C) 2016 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

if nargin < 2
    isExact = 0;
else
    warning('isExact not implemented. Will ignore it (isExact = 0)');
end


if ischar(cellOrString)
    if exist(cellOrString, 'dir') % directory, select all files in directory
        pathToPrefix = cellOrString;
        fileArray = dir(cellOrString); % find all files and directories in folder
        fileArray = {fileArray(~cell2mat({fileArray.isdir})).name}; % select only files
    elseif exist(cellOrString, 'file')
        % single file, check, if exists (would include folders, but checked
        % before)
        [pathToPrefix, fn, ext] = fileparts(cellOrString);
        % remove path to be consistent with other cases
        fileArray = {[fn ext]};
    else % fileprefix or regular expression
        % try whether it is a file prefix e.g. spm_*.img
        pathToPrefix = fileparts(cellOrString);
        fileArray = dir([cellOrString '*']);
        
        if ~isempty(fileArray)
            fileArray = {fileArray(~cell2mat({fileArray.isdir})).name}; % select only files
        else
            % normal wildcards did not work => try regular expression
            % identify path part of input, ends with a slash
            % note: for windows, this does not incorporate any recursive
            % search into sub-folders; for unix, a reasonably specified path
            % is assumed without regular expressions extending into the path
            
            % determine any part of the input regex that is a full directory
            % without any wildcards (^$*.?) (leave in +, since some Matlab
            % directories contain them)
            [iStart, iEnd] = regexp(cellOrString, '[^ \^|\*|\.|\?|\$]*/');
            % if no wildcard free directory exists, use .
            if isempty(iStart) || iStart(1) > 1
                pathWithOutRegex = '.';
                iEnd = 0;
            else
                pathWithOutRegex = cellOrString(iStart(1):iEnd(1));
            end
            
            pathToPrefix = pathWithOutRegex;
            
            stringRegex = cellOrString((iEnd(1)+1):end);
            % use find with -regex, if unix/mac system
            % regex is all after the determined patWithoutRegex
            if isunix
                [~, fileArray] = unix(['find ' pathWithOutRegex ' -regex ''' stringRegex '''']);
                if any(strfind(fileArray, 'No such file or directory'))
                    fileArray = {};
                end
            else
                % no recursive dir, just use regexp on current directory listing
                % in directory defined by path-part of regex
                fileArray = dir(pathWithOutRegex);
                fileArray = {fileArray(~cell2mat({fileArray.isdir})).name}; % select only files
                if ~isempty(fileArray)
                    isMatchingRegex = cell2mat(cellfun(@(x) ~isempty(x), ...
                        regexp(fileArray, stringRegex), 'UniformOutput', false));
                    fileArray = fileArray(isMatchingRegex);
                end
            end
            
            if isempty(fileArray)
                error('tapas:uniqc:NoMatchingFilenames', ...
                    'No matching image files with name/location/regex %s', cellOrString);
            end
            
        end
    end
    isValidPathToPrefix = ~isempty(pathToPrefix);
    if isValidPathToPrefix
        fileArray = strcat(pathToPrefix, filesep, fileArray); % prepend dir
    end
    
elseif iscell(cellOrString)
    fileArray = cellOrString;
    iExistingFiles = find(cell2mat(cellfun(@(x) exist(x, 'file'), fileArray, ...
        'UniformOutput', false)));
    fileArray = fileArray(iExistingFiles);
else
    error('tapas:uniqc:InputNotStringOrCell', ...
        'Input must be cell of strings or string');
end