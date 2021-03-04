function [fh, dataPlot, allColorMaps, allImageRanges, allImageNames] = ...
    plot_overlays(this, overlayImages, varargin)
% Plots this image with other images overlayed
%
%   Y = MrImage()
%   Y.plot_overlays(overlayImages)
%
% This is a method of class MrImage.
%
% IN
%   overlayImages               MrImage or cell of MrImages that shall be
%                               overlayed
%               'colorMap'      char or function handle; colormap for image
%                               underlay
%                               default: 'gray'
%               'windowStyle'   'docked' or 'default' to group Matlab
%                               figure windows
%               'overlayColorMaps'      
%                               cell(nOverlays,1) of chars or function 
%                               handles; colormaps for image overlays
%                               default: {'hot'; 'cool'; ;spring; ...
%                                         'summer'; winter'; 'jet'; 'hsv'}
%               'overlayAlpha'  transparency value of overlays
%                               (0 = transparent; 1 = opaque; default: 0.2)
%                               defaults to 1 for edge/mask overlayMode
%               'overlayMode'  'edge', 'mask', 'map'
%                                   'blend' (default); overlay is
%                                           transparent on top of underlay
%                                   'edge'  only edges of overlay are
%                                           displayed
%                                   'mask'  every non-zero voxel is
%                                           displayed (different colors for
%                                           different integer values, i.e.
%                                           clusters'
%                                   'map'   thresholded map in one colormap
%                                           is displayed (e.g. spmF/T-maps)
%                                           thresholds from
%                                           overlayThreshold
%               'overlayThreshold'  [minimumThreshold, maximumThreshold]
%                                   thresholds for overlayMode 'map'
%                                   default: [] = [minValue, maxValue]
%                                   everything below minValue will not be
%                                   displayed;
%                                   everything above maxValue
%                                   will have brightest color
%               'edgeThreshold'     determines where edges will be drawn,
%                                   the higher, the less edges
%                                   Note: logarithmic scale, e.g. try 0.005
%                                   if 0.05 has too little edges
%               'plotMode'          transformation of data before plotting
%                                   'linear' (default), 'log'
%               'selectedVolumes'   [1,nVols] vector of selected volumes to
%                                             be displayed
%               'selectedSlices'    [1,nSlices] vector of selected slices to
%                                               be displayed
%                                   choose Inf to display all volumes
%               'sliceDimension'    (default: 3) determines which dimension
%                                   shall be plotted as a slice
%               'rotate90'          default: 0; 0,1,2,3; rotates image
%                                   by multiple of 90 degrees AFTER
%                                   flipping slice dimensions
%               'doPlot'            false or true (default)
%                                   if false, only the data to be plotted
%                                   (rgb) is computed and returned without
%                                   actual plotting (e.g. to use in other
%                                   plot functions);
%               for montage plots:
%               'nRows'             default: NaN (automatic calculation)
%               'nCols'             default NaN (automatic calculation)
%               'FontSize'          font size of tile labels of montage
%               'plotTitle'         if true, title i.e. readible name of
%                                   image is put in plot
%               'plotLabels'        if true, slice labels are put into
%                                   montage%
% OUT
%   fh              figure handle;
%   dataPlot        [nVoxelX, nVoxelY, 3, nSelectedSlices, nSelectedVolumes]
%                   of RGB data created by overlaying overlayImages on this
%                   image
%   allColorMaps    cell(nOverlays+1,1) of all colormaps (including
%                   underlay image)
%   allImageRanges  cell(nOverlays+1,1) of [minValue, maxValue]
%                   representing image/overlay ranges for min/max color
%   allImageNames   cell(nOverlays+1,1) of image and overlay names (1st
%                   element: image name)
%
% EXAMPLE
%   X = MrImage('struct.nii');
%   Z = MrImage('spmF_0001.nii');
%   X.plot_overlays(Z, 'overlayMode', 'map', 'overlayThreshold', ...
%               [4.5, 100], 'selectedSlices', [40:45])
%
%   See also MrImage

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2014-11-24
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


if isreal(this)
    defaults.signalPart         = 'all';
else
    defaults.signalPart         = 'abs';
end

defaults.windowStyle            = 'docked'; %'default' or 'docked' to group Matlab figures
defaults.colorMap               = 'gray'; % colormap for underlay
defaults.overlayColorMaps = {
    'hot'
    'cool'
    'spring'
    'summer'
    'winter'
    'jet'
    'hsv'
    };
defaults.plotMode               = 'linear';
defaults.selectedVolumes        = 1;
defaults.selectedSlices         = Inf;
defaults.sliceDimension         = 3;
defaults.rotate90               = 0;
defaults.overlayMode            = 'mask';
defaults.overlayThreshold       = [];
defaults.overlayAlpha           = []; % depends on overlayMode
defaults.edgeThreshold          = [];
defaults.colorBar               = 'on';
defaults.doPlot                 = true;
defaults.plotLabels             = true;
defaults.plotTitle              = true;

defaults.nRows                  = NaN;
defaults.nCols                  = NaN;
defaults.FontSize               = 10;

args = tapas_uniqc_propval(varargin, defaults);
tapas_uniqc_strip_fields(args);

%% convert color map chars to function handels
if ischar(colorMap)
    funcColorMapUnderlay = str2func(colorMap);
else
    funcColorMapUnderlay = colorMap;
end

for c = 1:numel(overlayColorMaps)
    overlayColorMap = overlayColorMaps{c};
    if ischar(overlayColorMap)
        funcColorMapsOverlay{c} = str2func(overlayColorMap);
    else
        funcColorMapsOverlay{c} = overlayColorMap;
    end
end

% set default Alpha depending on define mode'
if isempty(overlayAlpha)
    switch overlayMode
        case {'mask', 'edge'}
            overlayAlpha = 1;
        case 'blend'
            overlayAlpha = 0.2;
        case 'map'
            overlayAlpha = 0.7;
        otherwise
            overlayAlpha = 0.1;
    end
end

doPlotColorBar = strcmpi(colorBar, 'on');

if ~iscell(overlayImages)
    overlayImages = {overlayImages};
end

overlayImages   = reshape(overlayImages, [], 1);

% Assemble parameters for data extraction into one structure
argsExtract     = struct('sliceDimension', sliceDimension, ...
    'selectedSlices', selectedSlices, 'selectedVolumes', selectedVolumes, ...
    'plotMode', plotMode, 'rotate90', rotate90, 'signalPart', signalPart);

nColorsPerMap   = 256;

dataPlot        = this.extract_plot4D_data(argsExtract);


%% Resize overlay images and extract data from all of them

nOverlays       = numel(overlayImages);
dataOverlays    = cell(nOverlays,1);


for iOverlay = 1:nOverlays
    overlay = overlayImages{iOverlay};
    
    useSpmReslice = false;
    
    if useSpmReslice
        % TODO: can we remove that? Or rename "reslice?"
        resizedOverlay = overlay.copyobj;
        resizedOverlay.parameters.save.fileName = ['forPlotOverlays_', ...
            resizedOverlay.parameters.save.fileName];
        resizedOverlay.parameters.save.keepCreatedFiles = 'none';
        resizedOverlay = resizedOverlay.reslice(this.geometry);
    else
        resizedOverlay = overlay.resize(this.dimInfo);
    end
    
    %% for map: overlayThreshold image only,
    %  for mask: binarize
    %  for edge: binarize, then compute edge
    
    switch overlayMode
        case {'map', 'maps'}
            resizedOverlay.threshold(overlayThreshold);
        case {'mask', 'masks'}
            resizedOverlay.threshold(0, 'exclude');
        case {'edge', 'edges'}
            resizedOverlay.threshold(0, 'exclude');
            % for cluster mask with values 1, 2, ...nClusters,
            % leave values of edge same as cluster values
            resizedOverlay = edge(resizedOverlay,'log', edgeThreshold);
               
%            resizedOverlay = edge(resizedOverlay,'log', edgeThreshold).*...
%                 imdilate(resizedOverlay, strel('disk',4));
    end
    dataOverlays{iOverlay} = resizedOverlay.extract_plot4D_data(argsExtract);
    
end



%% Define color maps for different cases:
%   map: hot
%   mask/edge: one color per mask image, faded colors for different
%   clusters within same mask


overlayColorMap = cell(nOverlays,1);
switch overlayMode
    case {'mask', 'edge', 'masks', 'edges'}
        baseColors = hsv(nOverlays);
        
        % determine unique color values and make color map
        % a shaded version of the base color
        for iOverlay = 1:nOverlays
            indColorsOverlay = unique(dataOverlays{iOverlay});
            nColorsOverlay = max(2, round(...
                max(indColorsOverlay) - min(indColorsOverlay)));
            overlayColorMap{iOverlay} = tapas_uniqc_get_brightened_color(...
                baseColors(iOverlay,:), 1:nColorsOverlay - 1, ...
                nColorsOverlay -1, 0.7);
            
            % add for transparency
            overlayColorMap{iOverlay} = [0,0,0; ...
                overlayColorMap{iOverlay}];
        end
        
    case {'map', 'maps'}
        for iOverlay = 1:nOverlays
            overlayColorMap{iOverlay} = ...
                funcColorMapsOverlay{iOverlay}(nColorsPerMap);
        end
        
end



%% Assemble RGB-image for montage by adding overlays with transparency as
% RGB in right colormap
rangeOverlays   = cell(nOverlays, 1);
rangeImage      = cell(nOverlays, 1);
for iOverlay = 1:nOverlays
    [dataPlot, rangeOverlays{iOverlay}, rangeImage{iOverlay}] = ...
        tapas_uniqc_add_overlay(dataPlot, dataOverlays{iOverlay}, ...
        overlayColorMap{iOverlay}, ...
        overlayThreshold, ...
        overlayAlpha);
end



%% Plot as montage
% TODO: implement this via MrImage.plot as well!

stringTitle = sprintf('Overlay Montage - %s', this.name);
fh = figure('Name', stringTitle, 'WindowStyle', windowStyle);

if isinf(selectedSlices)
    selectedSlices = 1:this.geometry.nVoxels(3);
end

if plotLabels
    stringLabelSlices = cellfun(@(x) num2str(x), ...
        num2cell(selectedSlices), 'UniformOutput', false);
else
    stringLabelSlices = {};
end


tapas_uniqc_labeled_montage(dataPlot, 'LabelsIndices', stringLabelSlices, ...
    'Size', [nRows nCols], 'FontSize', FontSize);

resolution_mm = this.dimInfo.get_dims({'y', 'x', 'z'}).resolutions;

switch sliceDimension
    case 1
        resolution_mm = resolution_mm([3 2 1]);
    case 2
        resolution_mm = resolution_mm([1 3 2]);
    case 3
        %   as is...
end

if mod(rotate90, 2)
    resolution_mm(1:2) = resolution_mm([2 1]);
end

set(gca, 'DataAspectRatio', abs(resolution_mm));

if plotTitle
    title(tapas_uniqc_str2label(stringTitle));
end


%% Add colorbars as separate axes

imageColorMap   = funcColorMapUnderlay(nColorsPerMap);
allColorMaps    = [{imageColorMap}; overlayColorMap];
allImageRanges  = [rangeImage(1); rangeOverlays];
allImageNames   = cellfun(@(x) x.name, overlayImages, ...
    'UniformOutput', false);
allImageNames   = [{this.name}; allImageNames];

if doPlotColorBar
    tapas_uniqc_add_colorbars(gca, allColorMaps, allImageRanges, allImageNames);
end



end

