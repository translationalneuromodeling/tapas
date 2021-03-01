function this = set_from_affine_geometry(this, affineTransformation, nVoxels, TR_s)
% Sets dimInfo from affine affineTransformation, assuming 4D nifti data
%
%   Y = MrDimInfo()
%   Y.set_from_affine_geometry(affineTransformation, nVoxels, TR_s)
%
% This is a method of class MrDimInfo.
%
% IN
%
% OUT
%
% EXAMPLE
%   set_from_affine_geometry
%
%   See also MrDimInfo

% Author:   Lars Kasper
% Created:  2017-11-07
% Copyright (C) 2017 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


if nargin < 3
    nVoxels = [1 1 1 1];
end

if nargin < 4
    TR_s = 1;
end

% or add them...
dimLabelsGeom = {'x','y','z', 't'};
units = {'mm', 'mm', 'mm', 's'};
iDimGeom = 1:4;
% update existing geom dimensions, add new ones for
% non-existing
iValidDimLabels = this.get_dim_index(dimLabelsGeom);
iDimGeomExisting = find(iValidDimLabels);
iDimGeomAdd = setdiff(iDimGeom, iDimGeomExisting);

% need nifti to reference first sampling point as offcenter
resolutions = [affineTransformation.scaling TR_s];

% voxel position by voxel center, time starts at 0 
firstSamplingPoint = [affineTransformation.scaling 0]/2; 

% if dimension labels exist, just update values
this.set_dims(dimLabelsGeom(iDimGeomExisting), ...
    'resolutions', resolutions(iDimGeomExisting), ...
    'nSamples', nVoxels(iDimGeomExisting), ...
    'firstSamplingPoint', firstSamplingPoint(iDimGeomExisting), ...
    'units', units(iDimGeomExisting));

% if they do not exist, create dims
this.add_dims(dimLabelsGeom(iDimGeomAdd), ...
    'resolutions', resolutions(iDimGeomAdd), ...
    'nSamples', nVoxels(iDimGeomAdd), ...
    'firstSamplingPoint', firstSamplingPoint(iDimGeomAdd), ...
    'units', units(iDimGeomAdd));