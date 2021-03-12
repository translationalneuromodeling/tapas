function this = coregister(this)
% Affine coregistraton of given images to a stationary image (geometry only) 
%
%   Y = MrSeries()
%   Y.coregister(inputs)
%
% This is a method of class MrSeries.
%
% IN
%       parameters.coregister ->
%           .nameStationaryImage            (e.g. 'mean')
%           .nameTransformedImage           (e.g. 'anatomy')
%           .nameEquallyTransformedImages   (e.g.'tissueProbabilityMap*')
% OUT
%
% EXAMPLE
%   coregister
%
%   See also MrSeries

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-07-28
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

nameStationaryImage = this.parameters.coregister.nameStationaryImage;
nameTransformedImage = this.parameters.coregister.nameTransformedImage;
nameEquallyTransformedImages = ...
    this.parameters.coregister.nameEquallyTransformedImages;

stationaryImage = this.find('MrImage', 'name', ['^' nameStationaryImage '*']);
stationaryImage = stationaryImage{1};
transformedImage = this.find('MrImage', 'name', ['^' nameTransformedImage '*']);
transformedImage = transformedImage{1};
equallyTransformedImages = this.find('MrImage', 'name',...
    ['^' nameEquallyTransformedImages '*']);

% TODO: if transformed image has more than one volume, make sure to apply
% to all volumes!

% set paths of all images correctly
this.init_processing_step('coregister', transformedImage, ...
    equallyTransformedImages);


[~, affineCoregistrationGeometry] = transformedImage.coregister_to(...
    stationaryImage, 'applyTransformation', 'geometry');
this.parameters.coregister.affineCoregistrationGeometry = ...
    affineCoregistrationGeometry;


% now apply geometry Update to all other listed images that shall be
% transformed
% inverse transformation used, since coregister_to gives mapping from
% stationary to transformed image
nImages = numel(equallyTransformedImages);
for iImage = 1:nImages
    equallyTransformedImages{iImage}.affineTransformation.apply_inverse_transformation(...
        affineCoregistrationGeometry);
    equallyTransformedImages{iImage}.save;
end

this.finish_processing_step('coregister', transformedImage, ...
    equallyTransformedImages);