function this = compute_masks(this)
% Segments defined input image into tissue types & thresholds to get masks
% - input image can be anatomical or mean functional
%
%   Y = MrSeries()
%   Y.compute_masks(inputs)
%
% This is a method of class MrSeries.
%
% IN
%       parameters.compute_masks.
%           nameInputImage          String with image name (or search pattern) 
%                                   from which masks shall be created
%           threshold               Threshold at and above mask shall be equal 1
%           keepExistingMasks       If true, existing Images in masks-cell 
%                                   are retained, new masks appended; 
%                                   If false, masks is overwritten by new masks
%           targetGeometry          String with image name (or search pattern) 
%                                   to which masks shall be resliced
%
% OUT
%
% EXAMPLE
%   compute_masks
%
%   See also MrSeries

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-07-14
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


%% init parameters for masking and file names

% allow cell of strings entries and pre-/suffix them with placeholders
nameInputImages = cellfun(@(x) ['^' x '*'], ...
    cellstr(this.parameters.compute_masks.nameInputImages), ...
    'UniformOutput', false);
threshold = this.parameters.compute_masks.threshold;
keepExistingMasks = this.parameters.compute_masks.keepExistingMasks;
nameTargetGeometry = this.parameters.compute_masks.nameTargetGeometry;

handleInputImages = this.find('MrImage', 'name', ...
    nameInputImages);% find input images...
handleTargetImage = this.find('MrImage', 'name', ...
    ['^' nameTargetGeometry '*']);

nImages = numel(handleInputImages);

inputImages = cell(nImages, 1);
for iImage = 1:nImages
    inputImages{iImage} = ...
        handleInputImages{iImage}.copyobj;
end

targetGeometry = handleTargetImage{1}.geometry.copyobj;

% clear masks, if not wanted to be kept
if ~keepExistingMasks
    this.masks = {}; % TODO: maybe a proper clear?
end

this.init_processing_step('compute_masks', inputImages);

% replicate threshold for all images, if only 1 number given
if numel(threshold) == 1
    threshold = repmat(threshold, nImages,1);
end

% compute masks and link them to MrSeries.masks
for iImage = 1:nImages
    % make sure the final images are saved
    oldKeepCreatedFiles = inputImages{iImage}.parameters.save.keepCreatedFiles;
    inputImages{iImage}.parameters.save.keepCreatedFiles = 'processed';
    inputImages{iImage} = inputImages{iImage}.compute_mask('threshold', threshold(iImage), ...
    'targetGeometry', targetGeometry, ...
    'caseEqual', 'include');
    inputImages{iImage}.name = sprintf('mask (%s)', inputImages{iImage}.name);
  
    this.masks{end+1,1} = inputImages{iImage};
    inputImages{iImage}.parameters.save.keepCreatedFiles = oldKeepCreatedFiles;
end

%% finish processing by deleting obsolete files, depending on save-parameters

this.finish_processing_step('compute_masks', inputImages);