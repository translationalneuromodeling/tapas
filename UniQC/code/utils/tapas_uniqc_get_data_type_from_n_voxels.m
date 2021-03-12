function dataType = tapas_uniqc_get_data_type_from_n_voxels(nVoxels)
% Specifies memory-efficient data type for saving from number of voxels
% TODO: incorporate dynamic range of data to be saved as well! add option
% for user to specify data type and bit depth?
%
% dataType = tapas_uniqc_get_data_type_from_n_voxels(nVoxels)
%
% IN
%   nVoxels    [1,n] voxels per dimension to be saved
%               see also MrImageGeometry
% OUT
%
% EXAMPLE
%   tapas_uniqc_get_data_type_from_n_voxels
%
%   See also

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2015-12-06
% Copyright (C) 2015 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

% - use double precion (64 bit) for structural images (3D and more than
%   220x220x120 voxel)
% - use int16 for very large data sets (when float32 would exceed 2GB)
% - use single precision (32 bit) for everything in-between

is3D = numel(nVoxels) <= 3 || nVoxels(4) == 1;
isStructural = prod(nVoxels(1:3)) >= 220*220*120;
floatExceeds2GB = prod(nVoxels) >= 2*1024*1024*1024*8/32;

if is3D && isStructural % highest bit resolution for structural images
    dataType   = 'float64';    
elseif floatExceeds2GB % int16 for large raw data
    dataType = 'int16';
    warning(['Due to the large number of samples, ', ...
        'data will be converted and saved as 16-bit integer.']);
else
    dataType   = 'float32'; %float32 for everything in between
end