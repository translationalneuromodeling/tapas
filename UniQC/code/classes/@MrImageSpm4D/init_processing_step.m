function previousPaths = init_processing_step(this, method, otherImages)
% Saves nii-files used by matlabbatches for SPM for different processing
% steps
%
% NOTE: as a side effect, the paths of all given images will be temporarily
% set to the sub-folder 'raw' of the calling MrImage path
% (this will be undone by finish_processing_step)
%
%   Y = MrImage()
%   previousPaths = Y.init_processing_step(inputs)
%
% This is a method of class MrImage.
%
% IN
%   method      string specifying processing step, e.g. 'realign', 't_filter'
%   otherImages
% OUT
%   previousPaths   cell of original paths for current MrImage and all ...
%                   otherImages
%
%
% EXAMPLE
%   init_processing_step
%
%   See also MrImage

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-08-15
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.



hasMatlabbatch = ismember(method, this.get_all_matlabbatch_methods());
doSaveRaw = strcmp(this.parameters.save.items, 'all');


hasOtherImages = nargin>=3;
if hasOtherImages
    allImages = [{this};otherImages(:)];
else
    allImages = {this};
end

% create temporary directory for data saving
mkdir(fullfile(this.parameters.save.path, 'raw'));

nImages = numel(allImages);
previousPaths = cell(nImages,1);
for iImage = 1:nImages
    previousPaths{iImage} = allImages{iImage}.parameters.save.path;
    allImages{iImage}.parameters.save.path = ...
        fullfile(this.parameters.save.path, 'raw');
    if hasMatlabbatch || doSaveRaw
        allImages{iImage}.save();
    end
end