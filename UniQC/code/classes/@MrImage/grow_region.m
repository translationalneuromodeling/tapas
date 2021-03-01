function newSegmentation = grow_region(this, startingSegmentation, threshold, nIter)
% Iterative region growing using starting segmentation and threshold.
%
%   Y = MrImage()
%   Y.grow_region(startingSegmentation, threshold, nIter)
%
% This is a method of class MrImage.
%
% IN
%   startingSegmentation    MrImage-object containing the starting
%                           segmentation, e.g. obtained via binarizing
%                           this
%   threshold               Threshold above which voxel connected to the
%                           object are considered part of the object.
%   nIter                   Number of maximum iterations. Maximum path
%                           length that can be grown.
%
% OUT
%
% EXAMPLE
%   Y.grow_region(Y.binarize(500), 300, 20);
%
%   See also MrImage

% Author:   Saskia Bollmann
% Created:  2020-03-30
% Copyright (C) 2020 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

% initialize
trueLocations = startingSegmentation.copyobj();

for n = 1:nIter
    
    % create searchLocations
    searchLocations = trueLocations.imdilate(strel('disk', 1));
    % exclude already found locations
    searchLocations = searchLocations - trueLocations;
    % apply in this
    searchVoxel = this .* searchLocations;
    % threshold
    newLocations = searchVoxel.binarize(threshold);
    % check new found locations
    disp([num2str(sum(newLocations.data(:))), ' new voxel(s) found.']);
    if ~any(newLocations.data(:))
        break
    end
    % add to true locations
    trueLocations = trueLocations + newLocations;
    
end
newSegmentation = trueLocations;

end

