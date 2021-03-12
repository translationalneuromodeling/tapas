function this = read_recon6_image_data(this, obj)
% Reads from ImageData format of Recon 6 (Zurich Recon Repository)
%
%   Y = MrDataNd()
%   Y.read_recon6_image_data(inputs)
%
% This is a method of class MrDataNd.
%
% IN
%
% OUT
%
% EXAMPLE
%   read_recon6_image_data
%
%   See also MrDataNd

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2017-04-26
% Copyright (C) 2017 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


this.data = obj.data;

ranges = obj.geometry.FOV;
if abs(ranges(3)) < eps, ranges(3) = obj.geometry.slice_thickness; end
dimLabels = obj.dataDimensions;
nDims = numel(dimLabels);
ranges(end+1:nDims) = 1;
for iDim = 1:nDims
    nSamples(iDim) = size(obj.data,iDim);
    if numel(ranges) < iDim
        ranges(iDim) = nSamples(iDim);
    end
    switch dimLabels{iDim}
        case 'm'
            dimLabels{iDim} = 'x';
            units{iDim} = 'm';
        case 'p'
            dimLabels{iDim} = 'y';
            units{iDim} = 'm';
        case {'s', 'sli'}
            dimLabels{iDim} = 'z';
            units{iDim} = 'm';
        case 'channels'
            units{iDim} = '1';
    end
    resolutions(iDim) = ranges(iDim)/size(this.data, iDim);
    
end

% retrieve rotation in scanner system via rotation matrix
R = obj.geometry.rot_traj_xyz_to_mps;
R(4,4) = 1;
P = tapas_uniqc_spm_imatrix(R);
rotation = P(4:6);

this.dimInfo = MrDimInfo(...
    'units', units, ...
    'dimLabels', dimLabels, ...
    'nSamples', nSamples, 'resolutions', resolutions);

% TODO: update offcenter and rotation in affineGeometry
%  'offcenter', obj.geometry.offcentre_xyz_slice, ...
%     'rotation', rotation