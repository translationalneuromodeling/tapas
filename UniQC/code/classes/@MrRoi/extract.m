function this = extract(this, image, mask)
% Extracts data within mask from given image
%
%   Y = MrRoi()
%   Y.extract(image, mask)
%
% This is a method of class MrRoi.
%
% IN
%   image       MrImage of which data shall be extracted within mask
%   mask        MrImage (binary mask), of which voxels shall be extracted
% OUT
%   this.perSlice.data
%   this.perVolume.data
%
% EXAMPLE
%   extract
%
%   See also MrRoi

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-07-18
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

nSlices  = image.geometry.nVoxels(3);
nVolumes = image.geometry.nVoxels(4);
iDimOther = image.dimInfo.get_dim_index({'x', 'y','z'}, 'invert', true);
nX = image.dimInfo.nSamples('x');
nY = image.dimInfo.nSamples('y');
nVoxelsPerSlice = nX*nY;
       
this.data = cell(nSlices,1);
this.perSlice.nVoxels = zeros(nSlices,1);

maskGeometry = mask.geometry.copyobj;

tolerance = 1e-4;
[diffGeometry, isEqual, isEqualGeom3D] = ...
    image.geometry.diffobj(maskGeometry, tolerance);

if ~isEqualGeom3D
    error('tapas:uniqc:MrRoi:UnmatchingImageGeometries', ...
        'Roi extraction: Image geometries do not match. Resize Image or Mask');
else
    for iSlice = 1:nSlices    
        [currentSlice, selectionIndexArray] = image.select('z', iSlice);
        % reshape data of slice into nVoxelX * nVoxelY, nVolumes 2D Matrix
        nSamplesOtherDims = image.dimInfo.nSamples(iDimOther);
        if isempty(nSamplesOtherDims), nSamplesOtherDims = 1; end
        dataSlice = reshape(currentSlice.data, [nVoxelsPerSlice, nSamplesOtherDims]);
        
        % create 1-dimensional vector of indices for voxels within mask
        maskSlice = find(mask.select('z', iSlice).data);
        
        this.data{iSlice} = dataSlice(maskSlice, selectionIndexArray{iDimOther});
        this.perSlice.nVoxels(iSlice,1) = (size(this.data{iSlice}, 1));
    end
    
    
    this.nSlices = nSlices;
    this.nVolumes = nVolumes;
    this.perVolume.nVoxels = sum(this.perSlice.nVoxels);
    this.name = sprintf('roi (%s), image (%s)', mask.name, image.name);
end