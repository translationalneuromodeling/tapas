function outputImage = remove_clusters(this, varargin)
% Removes all voxel clusters within a given range of voxel counts, and/or
% a given range of percent of the area that is (approximately) filled;
% either slice-wise or volume-wise (8 or 26 neighbours) using Matlab's
% bwareaopen
%
%   Y = MrImage()
%   clusterRemovedImage = Y.remove_clusters('nPixelRange', nPixelRange, ...
%           'pAreaRange', pAreaRange, 'applicationDimension', applicationDimension)
%
% This is a method of class MrImage.
%
% IN
%   nPixelsPerClusterRange
%   percentAreaFilledRange
%           percent area of the cluster filled estimated using the
%           Solidity property
%   applicationDimension
%           dimensionality to perform operation
%           '2d' = slicewise application, separate 2d images (cluster
%           neighbours only considered within slice)
%           '3d' = neighbourhood considered as volume
%
% OUT
%   outputImage
%           MrImage where data matrix does not contain removed clusters
%
% EXAMPLE
%   Y = MrImage();
%   % remove all pixel clusters with 15 or less pixels (3D neighbourhood)
%   Y.remove_clusters('nPixelRange', [0 15], 'applicationDimension', '3D');
%   Y.remove_clusters('areaRange', [0 40], 'applicationDimension', '2D');
%
%   See also MrImage imdilate MrImage.imerode perform_unary_operation bwareaopen

% Author:   Lars Kasper
% Created:  2019-11-03
% Copyright (C) 2019 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

defaults.nPixelRange = [0 0];
defaults.pAreaRange = [0 0];
defaults.applicationDimension = '3D';

args = tapas_uniqc_propval(varargin, defaults);
tapas_uniqc_strip_fields(args);

areaRange = pAreaRange/100;
is3D = strcmpi(applicationDimension, '3D');

BW = this.copyobj.binarize(0, 'exclude');

if is3D
    conn = 26; % use 26 neighbours in 3D (all faces and corners connected)
else
    conn = 8; % use 8 neighbours (in 2D)
end

%% step 1 - remove cluster based on voxel count

% only crop, if required (i.e., connectivity components greater than 1)
if nPixelRange(1) > 1
    % since we want to reinclude this mask later on (range!), we do not add
    % the +1 usually needed for bwareaopen, i.e., we only remove smaller
    % clusters here
    BW1 = BW.perform_unary_operation(@(x) bwareaopen(x, nPixelRange(1), conn), ...
        applicationDimension);
else
    BW1 = BW.copyobj;
end

if nPixelRange(2) > 0
    % since we indeed use this as a subtractive mask, we have to add +1 to
    % remove the clusters at the upper end of the cluster range as well
    BW2 = BW.perform_unary_operation(@(x) bwareaopen(x, nPixelRange(2) + 1, conn), ...
        applicationDimension);
else
    BW2 = BW.copyobj;
end

% by the following subtraction, excluded voxels in both BW1 and BW2 will be
% reincluded, hence, an exclusive or is reached, only removing clusters in
% the range between BW1 and BW2
BW = abs(BW - BW1 - BW2);

%% step 2 - remove cluster based on percent area filled
% extract region properties
if is3D
    stats = bwconncomp(BW.data, conn);
    props = regionprops3(stats, 'Solidity');
    % threshold solidity
    solidity = [props.Solidity];
    removeClusters = (solidity > areaRange(1) & solidity < areaRange(2));
    removeVoxels = vertcat(stats.PixelIdxList{removeClusters});
    % remove voxel
    BW.data(removeVoxels) = 0;
else
    % regionprops does not automatically take into account the defined
    % connectivity, so we have to do the split by hand
    nSamplesZ = BW.dimInfo.nSamples('z');
    for z = 1:nSamplesZ
        tempData = BW.select('z', z).data;
        stats = bwconncomp(tempData, conn);
        props = regionprops(stats, 'Solidity');
        % threshold solidity
        solidity = [props.Solidity];
        removeClusters = (solidity > areaRange(1) & solidity < areaRange(2));
        removeVoxels = vertcat(stats.PixelIdxList{removeClusters});
        % remove voxel
        tempData(removeVoxels) = 0;
        BW.data(:,:,z) = tempData;
    end
end

outputImage = this.*BW;

end