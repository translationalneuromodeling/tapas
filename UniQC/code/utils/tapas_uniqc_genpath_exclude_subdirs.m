function reducedPath = tapas_uniqc_genpath_exclude_subdirs(completePath, pathsExcluded)
% works as genpath, but allows to specify excluded subfolders, which are...
% removed incl. their subfolders from the returned path
%
%    reducedPath = tapas_uniqc_genpath_exclude_subdirs(completePath, pathsExcluded)
%
% IN
%   completePath    the input path, as in genpath
%   pathsExcluded   cell(1, nPaths) of excluded sub-folders or paths
%
% OUT
%   reducedPath     string of returned genpath (e.g., use for addpath), 
%                   excluding the specified pathsExcluded and their subfolders.
%
% EXAMPLE
%   tapas_uniqc_genpath_exclude_subdirs
%
%   See also
 
% Author:   Lars Kasper
% Created:  2020-05-22
% Copyright (C) 2020 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.
%

%create cell of folder names from path string, split at path separator ; (win) or : (unix)
recursivePath = genpath(completePath);
reducedPath = regexp(recursivePath, pathsep, 'split'); 
for iExcluded = 1:numel(pathsExcluded)
    % note: regexprep might have some issues w/ escape characters, like \., but helps with Windows \ vs Unix /
    idxIncludedFolders = find(cellfun('isempty', regexp(...
        regexprep(reducedPath, '\\', '/'), ...
        regexprep(pathsExcluded{iExcluded}, '\\', '/')  ) ));
    reducedPath = reducedPath(idxIncludedFolders);
end
% add pathseparator to each path, convert back into path string
reducedPath = cellfun(@(x) [x pathsep], reducedPath, 'UniformOutput', false);
reducedPath = [reducedPath{:}];