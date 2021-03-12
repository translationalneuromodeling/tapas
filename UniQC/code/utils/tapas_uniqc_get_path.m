function pathOut = tapas_uniqc_get_path(folder)
% Returns absolute paths for given folder
%
%   pathOut = tapas_uniqc_get_path(folder)
%
% IN
%   folder      default: 'code'; returns full path of given folder within
%               UniQC Repository
%               options:
%               'code'
%               'utils'
%               'examples'
%               'classes'
%               TODO: automatic search for subfolders...
%
% OUT
%
% EXAMPLE
%   pathCode = tapas_uniqc_get_path('code');
%   pathUtils = tapas_uniqc_get_path('utils');
%
%   See also

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2014-11-18
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

% keep known between calls to this function, to not always prompt user for
% data folder (which might not be specified relative to this repository
persistent uniqc_pathExampleData;


if nargin < 1
    folder = 'code';
end

pathUtils = fullfile(fileparts(mfilename('fullpath')));

switch folder
    case 'utils'
        pathOut = pathUtils;
    case 'code'
        pathOut = fullfile(pathUtils, '..');
    case 'classes'
        pathOut = fullfile(pathUtils, '..', 'classes');
    case {'example', 'examples', 'data'}
        pathOut = fullfile(pathUtils, '..', '..', 'data');
        
        % at least this nifti file should exist in example folder
        relativePathNiftiFile = fullfile('nifti', 'paradigm_social_learning', ...
            'meanfmri.nii');
        hasNifti = @(pathTest) isfile(fullfile(pathTest, relativePathNiftiFile));
        
        % if not existent, 1) check for global data folder variable
        %                  2) prompt for  specifying data folder (saved in
        %                     global variable)
        if  hasNifti(pathOut) % update persistent path variable, if valid
            uniqc_pathExampleData = pathOut;
        else
            if hasNifti(uniqc_pathExampleData) % return persistent path variable, if current path invalid
                pathOut = uniqc_pathExampleData;
            else
                % prompt for a data path
                myDefaultExamplePath = 'C:\Users\kasperla\Documents\Code\uniqc-code\data';
                pathOut = input(...
                    ['Specify absolute path (with '''') to UniQC Example data [''' ...
                    regexprep(myDefaultExamplePath, '\\','\\\\') ''']: ']);
                % if no response, try standard path
                if isempty(pathOut)
                    pathOut = myDefaultExamplePath;
                end
                uniqc_pathExampleData = pathOut;
            end
        end
    case {'tests', 'test'}
        pathOut = fullfile(pathUtils, '..', '..', 'test');
    case {'demo', 'demos'}
        pathOut = fullfile(pathUtils, '..', '..', 'demo');
end

pathOut = tapas_uniqc_simplify_path(pathOut);