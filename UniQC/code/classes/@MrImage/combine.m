function imageCombined = combine(this, varargin)
% Combines multiple MrImages into a single one along specified
% dimensions. Basically MrDataNd.combine with additional
% affineTransformation-check
%
%   Y = MrImage()
%   imageCombined = Y.combine(imageArray, combineDims, tolerance)
%
% This is a method of class MrImage.
%
% IN
%
%   imageArray      cell of MrImage to be combined
%   combineDims     [1, nCombineDims] vector of dim indices to be combined
%                       OR
%                   cell(1, nCombineDims) of dimLabels to be combined
%                   default: all singleton dimensions (i.e. dims with one 
%                   sample only within each individual dimInfo)
%                   NOTE: if a non-singleton dimension is given, images are
%                         concatenated along this dimension
%
%   tolerance                   dimInfos are only combined, if their
%                               information is equal for all but the
%                               combineDims (because only one
%                               representation is retained for those,
%                               usually from the first of the dimInfos). 
%                               However, sometimes numerical precision,
%                               e.g., rounding errors, preclude the
%                               combination. Then you can increase this
%                               tolerance; 
%                               default: single precision (eps('single')
%                               ~1.2e-7)
%
% OUT
%
% EXAMPLE
%   combine
%
%   See also MrImage MrDataNd.combine MrDimInfo.combine

% Author:   Lars Kasper
% Created:  2018-05-17
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

imageCombined = combine@MrDataNd(this, varargin{:});

%% Check whether affine geometries of all individual objects in match,
% otherwise issue warning
imageArray = varargin{1};

if nargin < 4
    tolerance = eps('single');
else
    tolerance = varargin{3};
end

nSplits = numel(imageArray);
for iSplit = 1:nSplits
    % recursive isequal of MrCopyData
    isAffineTrafoEqual = isequal(imageCombined.affineTransformation, ...
        imageArray{iSplit}.affineTransformation, tolerance);
    if ~isAffineTrafoEqual
        warning('Affine Transformation of combined image differs from array entry %d', ...
            iSplit);
    end
    % check if common prefix available
    [~, ~, pfx{iSplit}, ext{iSplit}] = tapas_uniqc_get_dim_labels_from_string(imageArray{iSplit}.parameters.save.fileName);
end

% check if filename prefix and file extension are the same
uniquePrefix = unique(pfx);
uniqueExtension = unique(ext);
if (numel(uniquePrefix) == 1) && (numel(uniqueExtension) == 1)
    imageCombined.parameters.save.fileName = [uniquePrefix{1} uniqueExtension{1}];
end

%% Recast (e.g. MrImageSpm4D) as MrImage, if more than 4 non-singleton
% dimensions to avoid inconsistencies of high-dim MrImageSpm4D
if isa(imageCombined, 'MrImageSpm4D') && numel(imageCombined.dimInfo.get_non_singleton_dimensions()) > 4
    imageCombined = imageCombined.recast_as_MrImage();
end