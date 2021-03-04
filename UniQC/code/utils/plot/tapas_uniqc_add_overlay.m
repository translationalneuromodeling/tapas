function [rgbMatrix, rangeOverlay, rangeImage] = tapas_uniqc_add_overlay(...
    imageMatrix, overlayMatrix, overlayColorMap, ...
    overlayThreshold, overlayAlpha, displayRange, verbose)
% Overlays image with and overlay in given colormap, output is RGB (colormap-independent)
%
%   rgbMatrix = tapas_uniqc_add_overlay(imageMatrix, overlayMatrix, overlayColorMap, ...
%    overlayThreshold, overlayAlpha)
%
% IN
%   imageMatrix         [nX, nY, nSlices] data matrix or [nX, nY, 3, nSlices]
%                       RGB-matrix => will be plotted in grayscale
%   overlayMatrix       [nX, nY, nSlices] data matrix
%   overlayColorMap     Colormap for overlay
%                       [nColors,3] colormap or
%                       colormap function handle (e.g. @jet) or
%                       string of inbuilt colormaps (e.g. 'jet')
%   overlayThreshold    [minValue, maxValue]
%                       default: [min(overlayMatrix) max(overlayMatrix)]
%                       values in overlayMatrix <= minValue will be mapped
%                       on overlayColorMap(1,:)
%                       values in overlayMatrix >= maxValue will be mapped
%                       on overlayColorMap(end,:)
%   overlayAlpha        transparency value for overlay: 0...1
%                       default: 0.1
%                       1 = opaque
%                       0 = transparent
%   verbose             default: false; if true, transformation to rgb
%                       results are plotted for both image and overlay
%
% OUT
%   rgbMatrix           image matrix with overlay merged into RGB-matrix
%   rangeOverlay        value range of overlay reflected by colormap of
%                       overlay
%   rangeImage          value range of image reflected by grayscale
%                       depiction in RGB matrix
% EXAMPLE
%   tapas_uniqc_add_overlay
%
%   See also

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2014-11-25
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

if nargin < 7
    verbose = 0;
end

if nargin < 6
    displayRange = [];
end

if nargin < 5
    overlayAlpha = 1;
end

if nargin < 4
    overlayThreshold = [];
end

if nargin < 3
    overlayColorMap = 'jet';
end

nColors = 256;



%% Determine min/max of image and overlay
valindIndices = find(~(overlayMatrix == 0 & ...
    isinf(overlayMatrix) & isnan(overlayMatrix)));
minOverlay = min(overlayMatrix(valindIndices));
maxOverlay = max(overlayMatrix(valindIndices));

if isempty(displayRange)
    valindIndices = find(~(imageMatrix == 0 & ...
        isinf(imageMatrix) & isnan(imageMatrix)));
    minImage = min(imageMatrix(valindIndices));
    maxImage = max(imageMatrix(valindIndices));
else
    minImage = displayRange(1);
    maxImage = displayRange(2);
    imageMatrix(imageMatrix < minImage) = minImage;
    imageMatrix(imageMatrix > maxImage) = maxImage;
end



%% Determine thresholds and colormap values

if isempty(overlayThreshold)
    overlayThreshold = [minOverlay, maxOverlay];
end


if ischar(overlayColorMap)
    overlayColorMap = eval(sprintf('%s(nColors);', overlayColorMap));
end

if ~isnumeric(overlayColorMap)
    overlayColorMap = overlayColorMap(nColors);
end


nColors = size(overlayColorMap,1);



%% Convert imageMatrix to grayscale-RGB, if not already RGB

% reproduce grayscale image 3x to make it RGB
if size(imageMatrix,3) ~= 3
    rgbImage =  repmat(mat2gray(permute(imageMatrix, [1 2 4 3])), ...
        [1 1 3 1]);
else
    rgbImage = imageMatrix;
end

plot_montage(rgbImage, 'rgbImage', verbose);


%% Convert overlayMatrix to colormap-RGB

overlayMatrix = gray2ind(mat2gray(overlayMatrix, overlayThreshold), nColors);

rgbOverlay  = zeros(size(rgbImage));
nSlices     = size(rgbImage, 4);
indZeros    = find(overlayMatrix==0);

for iSlice = 1:nSlices
    
    rgbOverlay(:,:,:,iSlice) = ind2rgb(overlayMatrix(:,:,iSlice), ...
        overlayColorMap);
end

plot_montage(rgbOverlay, 'rgbOverlay', verbose);


%% replace zero with color of underlay image
rgbOverlay = permute(rgbOverlay, [1 2 4 3]);
rgbImage = permute(rgbImage, [1 2 4 3]);

for iChannel = 1:3
    colorChannelOverlay{iChannel}   = rgbOverlay(:,:,:,iChannel);
    colorChannelImage{iChannel}     = rgbImage(:,:,:,iChannel);
    
    colorChannelOverlay{iChannel}(indZeros)    =  ...
        colorChannelImage{iChannel}(indZeros);
end

rgbOverlay  = cat(4, colorChannelOverlay{:});

rgbOverlay  = permute(rgbOverlay, [1 2 4 3]);
rgbImage    = permute(rgbImage, [1 2 4 3]);

plot_montage(rgbOverlay, 'rgbOverlay - merged', verbose);



%% Add overlay to image with transparency in RGB space
rgbMatrix = (1-overlayAlpha)*rgbImage + overlayAlpha*rgbOverlay;

plot_montage(rgbMatrix, 'rgbMatrix', verbose);



%% Assemble return parameters

rangeOverlay    = overlayThreshold;
rangeImage      = [minImage, maxImage];

end



%% local function for debugging plots
function plot_montage(dataMatrix, stringTitle, verbose)

if verbose
    figure('Name', stringTitle);
    montage(dataMatrix, 'DisplayRange', []);
    title(tapas_uniqc_str2label(stringTitle));
end
end