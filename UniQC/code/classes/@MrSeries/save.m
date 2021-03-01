function this = save(this)
%saves MrSeries in different file formats
%
%   MrSeries = save(MrSeries)
%
% This is a method of class MrSeries.
%
% IN
%   parameters.save.format
%
% OUT
%
% EXAMPLE
%   save
%
%   See also MrSeries

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-07-02
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


pathSave = this.parameters.save.path;

if ~exist(pathSave, 'dir')
    mkdir(pathSave)
end

% set save path of all objects to the one of the MrSeries
this.set_save_path();

% detect all image files from object and save as nifti-files in save-path
% of MrSeries
handleImageArray = this.get_all_image_objects();
for iImage = 1:numel(handleImageArray);
    handleImageArray{iImage}.save;
end

% strip data from object and save MrSeries itself
MrObject = this.copyobj('exclude', 'data'); % copies object without data
fileObject = fullfile(pathSave, 'MrObject.mat');
save(fileObject, 'MrObject')
