function pathExamples = tapas_physio_get_path_examples(pathPhysioPublic)
% Returns GitLab-internal or TAPAS-public PhysIO Examples folder, based on
% location of public PhysIO directory
%
%   pathExamples = tapas_physio_get_path_examples(pathPhysioPublic)
%
% IN
%   pathPhysioPublic    location of public PhysIO folder, e.g.,
%                       'tapas/PhysIO'
% OUT
%
% EXAMPLE
%   tapas_physio_get_path_examples('tapas/PhysIO')
%
%   See also

% Author:   Lars Kasper
% Created:  2022-09-05
% Copyright (C) 2022 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under
% the terms of the GNU General Public License (GPL), version 3. You can
% redistribute it and/or modify it under the terms of the GPL (either
% version 3 or, at your option, any later version). For further details,
% see the file COPYING or <http://www.gnu.org/licenses/>.

if nargin < 1
    pathPhysioPublic = fullfile(fileparts(mfilename('fullpath')), '..');
end
    
% try PhysIO-internal GitLab examples first
possibleExamplePaths = {
    fullfile(pathPhysioPublic, '..', 'examples')    
    fullfile(pathPhysioPublic, '..', 'physio-examples')
};

iPath = 0;
haveFoundExamplePath = false;
while ~haveFoundExamplePath && iPath < numel(possibleExamplePaths)
    iPath = iPath + 1;
    pathExamples = possibleExamplePaths{iPath};
    haveFoundExamplePath = isfolder(fullfile(pathExamples, 'BIDS'));
end

% otherwise use public TAPAS examples
if ~haveFoundExamplePath
    pathExamples =  fullfile(pathPhysioPublic, ...
        '..', 'examples', tapas_get_current_version(), 'PhysIO');
end

pathExamples = tapas_physio_simplify_path(pathExamples);

% If no examples folder found, suggest to download them via tapas-function
if ~isfolder(fullfile(pathExamples, 'BIDS'))
    physio = tapas_physio_new();
    tapas_physio_log('No PhysIO examples data found. Please download via tapas_download_example_data()', physio.verbose, 2);
end
