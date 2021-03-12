function this = set_save_path(this, pathSave, update)
% sets save path recursively for all saveable objects within MrSeries
% can also update save path if MrSeries folder is renamed (subfolders are
% kept, only top folder name changed)
%
%   Y = MrSeries()
%   Y.set_save_path(inputs)
%
% This is a method of class MrSeries.
%
% IN
%   pathSave        new save path (default: parameters.save.path)
%
% OUT
%
% EXAMPLE
%   set_save_path
%
%   See also MrSeries

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-07-09
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


if nargin < 2
    pathSave = this.parameters.save.path;
else
    % update MrSeries
    this.parameters.save.path = pathSave;
end

if nargin < 3
    update = 0;
end

% update images, rois and glms
updateClasses = {'MrImage', 'MrRoi', 'MrGlm'};
for nClasses = 1:numel(updateClasses)
    handleImageArray = this.find(updateClasses{nClasses});
    for iImage = 1:numel(handleImageArray);
        if update
            allFolders = regexp(handleImageArray{iImage}.parameters.save.path,...
                filesep, 'split');
            nameSubfolder = allFolders{end};
            handleImageArray{iImage}.parameters.save.path = ...
                fullfile(pathSave, nameSubfolder);
        else
            handleImageArray{iImage}.parameters.save.path = ...
                pathSave;
        end
    end
end

