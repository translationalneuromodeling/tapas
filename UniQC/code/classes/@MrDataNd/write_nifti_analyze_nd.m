function this = write_nifti_analyze_nd(this, filename)
% saves MrImage to nifti/analyze file depending on file extension
% (.nii/.img)
%
%   Y = MrImage(origFileName);
%   Y.write_nifti_analyze_nd(newFileName)
%
% This is a method of class MrImage.
%
% IN
%   fileName    string of file identifier to save to; if not specified
%               MrImage.parameters.save.path/name is used
%   dataType    number format for saving voxel values; see also spm_type
%               specified as one of the following string identifiers
%                'uint8','int16','int32','float32','float64','int8','uint16','uint32';
%               default (3D): float64
%               default (4D or size > 30 MB): int32
%
% OUT
%
% EXAMPLE
%   write_nifti_analyze_nd
%
%   See also MrImage

% Author:   Saskia Klein & Lars Kasper
% Created:  2021-02-21
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

% to write nifti files with nDims > 4 and no splits, use the matlab
% internal niftiwrite function

% calculate affine matrix
affine3d = this.geometry.get_affine_matrix()';
affine3d(4,1:3) = affine3d(4,1:3) + this.geometry.resolution_mm(); 

% first create info
info.Filename = filename;
info.Filemoddate = datestr(datetime);
% info.Filesize = [];
info.Version = 'NIfTI1';
info.Description = '';
info.ImageSize = this.dimInfo.nSamples;
info.PixelDimensions =this.dimInfo.resolutions;
info.Datatype = 'single';
info.BitsPerPixel = 32;
info.SpaceUnits = 'Millimeter';
info.TimeUnits = 'Second';
info.AdditiveOffset = 0;
info.MultiplicativeScaling = 1;
info.TimeOffset = 0;
info.SliceCode = 'Unknown';
info.FrequencyDimension = 0;
info.PhaseDimension = 0;
info.SpatialDimension = 0;
info.DisplayIntensityRange = [0 0];
info.TransformName = 'Sform';
info.Transform.Dimensionality = 3;
info.Transform.T = affine3d;
info.Qfactor = -1;

niftiwrite(single(this.data), filename, info);


