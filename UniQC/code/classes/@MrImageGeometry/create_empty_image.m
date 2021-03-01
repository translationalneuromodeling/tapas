function emptyImage = create_empty_image(this, varargin)
% Creates all-zeroes image with the same geometry as defined by this and
% the select option.
%
%   Y = MrImageGeometry()
%   Y.create_empty_image('t', 1:10)
%
% This is a method of class MrImageGeometry.
%
% IN    MrImageGeomtry object and optional select par/val pairs.
%
% EXAMPLE
%   % create 3D version of empty image from current geometry
%   Y.create_empty_image('selectedVolumes', 1);
%
%   See also MrImageGeometry

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2015-12-10
% Copyright (C) 2015 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

emptyImage = MrImage();

% add DimInfo
emptyImage.dimInfo = MrDimInfo(...
    'dimLabels', {'x', 'y', 'z', 't'}, ...
    'units', {'mm', 'mm', 'mm', 's'}, ...
    'nSamples', this.nVoxels, ...
    'resolutions', [this.resolution_mm, this.TR_s], ...
    'firstSamplingPoint', [-this.FOV_mm./2 + this.resolution_mm/2, this.TR_s]);

% add AffineTrafo
emptyImage.affineTransformation.update_from_affine_matrix(...
    this.get_affine_matrix()/emptyImage.dimInfo.get_affine_matrix());

emptyImage.data = zeros(emptyImage.geometry.nVoxels);
emptyImage.parameters.save.fileName = 'emptyImageTargetGeometry.nii';
if nargin > 1
    emptyImage.select(varargin{:});
end
