function [fh, plotImage] = plot(this, varargin)
% plots an MR image
%
%   Y  = MrImage
%   fh = Y.plot('ParameterName', ParameterValue)
%
% Note:
%   The first (specified) dimension will be plotted from left to right
%   The 2nd (specified) dimensions will be plotted from down to up
%   The 3rd (and higher) specified dimensions will be plotted in tiles or
%   figures (depending on options)
%
% IN
%   varargin    'ParameterName', 'ParameterValue'-pairs for the following
%               properties:
%               'plotType'          Type of plot that is created
%                                       'montage'   images are plotted as
%                                                   montages
%                                       'labeledMontage'
%                                                   as montage, but with
%                                                   labels (default)
%                                       'spm'       uses display functions
%                                                   from SPM (spm_display/
%                                                   spm_check_registration)
%                                                   to visualize 3D volumes
%                                                   with header information
%                                                   applied ("world space")
%                                                   Note: if multiple
%                                                   selected volumes are
%                                                   specified,
%                                                   spm_check_registration
%                                                   is used
%                                       'spmInteractive' /'spmi'
%                                                   same as SPM, but keeps
%                                                   temporary nifti files to
%                                                   allow clicking into spm
%                                                   figure
%                                       '3D'/'3d'/'ortho'
%                                                   See also tapas_uniqc_view3d
%                                                   Plots 3 orthogonal
%                                                   sections
%                                                   (with CrossHair) of
%                                                   3D image interactively
%               'linkOptions'       link another real-time plot to input
%                                   (e.g. mouse) on this one, using
%                                   positions on current plot as update to
%                                   plotDim-values for linked plot
%
%                                   shortcut string:
%                                   'timeseries_<plotDim>'/'ts_<plotDim>'
%                                       default: 'ts_4'/'ts_t'
%                                           plots time series of voxel that
%                                           mouse is currently pointing to
%                                       OR
%
%                                   struct of class MrLinkPlotOptions with fields/values
%                                   'plotType'  - 'timeseries' ...
%                                   'plotDim'   - dimension which should be
%                                                 plotted, default: 4
%                                   'fixedDimsPoint'
%                                               - cell(1,2*nFixedDims)
%                                                 dimension label/index array pairs that
%                                                 shall remain fixed/selected
%                                                 and are not on original plot
%                                                 default: {5 ,1, 6, 1,
%                                                 ..., nDims, 1}
%
%               'displayRange'      [1,2] vector for pixel value = black and
%                                                    pixel value = white
%               'overlay'           false (default) or true if an overlay
%                                   image is plotted
%               'signalPart'        for complex data, defines which signal
%                                   part shall be extracted for plotting
%                                       'all'       - take signal as is
%                                                     (default for
%                                                     real-valued data)
%                                       'abs'       - absolute value
%                                                     (default for complex
%                                                     data)
%                                       'phase'     - phase of signal
%                                       'real'      - real part of signal
%                                       'imag'      - imaginary part of
%                                                     signal
%               'plotMode'          transformation of data before plotting
%                                   'linear' (default), 'log'
%               'rotate90'          default: 0; 0,1,2,3; rotates image
%                                   by multiple of 90 degrees AFTER
%                                   flipping slice dimensions
%               'useSlider'         true or false
%                                   provides interactive slider for
%                                   slices/volumes
%               'colorMap'          string, any matlab colormap name
%                                   e.g. 'jet', 'gray'
%               'colorBar',         'on' or 'off' (default)
%                                   where applicable, determines whether
%                                   colorbar with displayRange shall be plotted
%                                   in figure;
%               'overlayImages'     (cell of) MrImages that will be
%                                   overlayed
%               'overlayMode'       'edge', 'mask', 'map'
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
%                                   tresholds for overlayMode 'map'
%                                   default: [-Inf, Inf] = [minValue, maxValue]
%                                   everything below minValue will not be
%                                   displayed;
%                                   everything above maxValue
%                                   will have brightest color
%               'overlayAlpha'      transparency value of overlays
%                                   (0 = transparent; 1 = opaque; default: 0.1)
%               'edgeThreshold'     determines where edges will be drawn,
%                                   the higher, the less edges
%                                   Note: logarithmic scale, e.g. try 0.005
%                                   if 0.05 has too little edges
%
%               data selection      data selection uses MrImage.select/
%                                   MrImage.dimInfo.select
%               'dimLabel'          scalar or vector with array indices or
%                                   sampling points
%               'imagePlotDim'      1x3 vector of dimensions or cell array
%                                   of labels that constitue the image that
%                                   will be plotted, i.e. which three
%                                   dimensions define the volume that will
%                                   be plotted within one figure, e.g.
%                                   [1,2,3] (default)
%                                   {'x', 'y', 'z'}
%               'selectionType'     'index' (default) or 'label' selects
%                                   hoew 'dimlabel' vector is interpreted,
%                                   as array indices or sampling points
%
%               Orientation changes:
%               'sliceDimension'    (default: 3) determines which dimension
%                                   shall be plotted as a slice
%                                   can be entered as index or dimLabel
%               'rotate90'          default: 0; 0,1,2,3; rotates image
%                                   by multiple of 90 degrees AFTER
%                                   flipping slice dimensions
%               for montage plots:
%               'nRows'             default: NaN (automatic calculation)
%               'nCols'             default NaN (automatic calculation)
%               'FontSize'          font size of tile labels of montage
%               'plotTitle'         if true, title i.e. readible name of
%                                   image is put in plot
%               'plotLabels'        if true, slice labels are put into
%                                   montage
%
% OUT
%   fh          [nFigures,1] vector of figure handles
%
% EXAMPLE
%
%   Y.plot('z', [6:10])
%   Y.plot('displayRange', [0 1000])
%   Y.plot('useSlider', true, 'z', []);
%
%   See also MrLinkPlotOptions

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-05-21
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.
%


% check whether image object has data
if isempty(this.data)
    error('tapas:uniqc:MrImageEmptyDataMatrix', ...
        'Data matrix empty for MrImage-object %s', this.name);
end
%% set defaults
% default signal part and plot mode
if isreal(this)
    defaults.signalPart         = 'all';
else
    defaults.signalPart         = 'abs';
end
defaults.plotMode               = 'linear';

% data selection
defaults.selectionType          = 'index';


% plot appearance
defaults.plotType               = 'labeledMontage';

defaults.FigureSize             = [1600 900];
defaults.nRows                  = NaN; % automatically determined by image size
defaults.nCols                  = NaN;
defaults.FontSize               = 10;
defaults.plotTitle              = true;
defaults.plotLabels             = true;

defaults.rotate90               = 0;
defaults.sliceDimension         = 3;
defaults.displayRange           = [];
defaults.useSlider              = false;
defaults.colorMap               = 'gray';
defaults.colorBar               = 'off';
defaults.windowStyle            = 'docked';
defaults.imagePlotDim           = [1,2,3];

% overlay parameters
defaults.overlay                = false;
defaults.overlayImages          = {};
defaults.overlayMode            = 'mask';
defaults.overlayThreshold       = [];
defaults.overlayAlpha           = []; % depends on overlayMode
defaults.edgeThreshold          = [];

% linked plot options
defaults.linkOptions             = [];

% get arguments
[args, ~] = tapas_uniqc_propval(varargin, defaults);
tapas_uniqc_strip_fields(args);

% check colorbar and overlays
doPlotColorBar = strcmpi(colorBar, 'on');
% plot overlays if overlay images are given, but not when plotType is spm -
% they are added via spm_check_registration
% make sure overlay images are cell
if ~iscell(overlayImages)
    overlayImages = {overlayImages};
end
doPlotOverlays = (overlay || ~isempty(overlayImages)) ...
    && ~(strcmpi(plotType, 'spm') || strcmpi(plotType, 'spmi'));

% convert imagePlotDim from label to index
if iscell(imagePlotDim)
    [~, imagePlotDim] = ismember(imagePlotDim, this.dimInfo.dimLabels);
end

doLinkPlot = ~isempty(linkOptions);
doMontage = ismember(lower(plotType), {'montage', 'labeledmontage'});

if doLinkPlot
    if ~isa(linkOptions, 'MrLinkPlotOptions')
        if ischar(linkOptions) % shortcut string to create options
            if contains(linkOptions, {'ts', 'timeseries'})
                % of the form timeseries_<iDim>, retrieve the second)
                splitString = regexp(linkOptions, '_', 'split');
                iDimLinkedPlot = str2num(splitString{2});
                linkOptions = MrLinkPlotOptions('ts', this.dimInfo, ...
                    imagePlotDim, iDimLinkedPlot);
            else
                error('tapas:uniqc:MrImage:InvalidLinkOptions', ...
                    'linkOptions must be a MrLinkOptions object or a shortcut string');
            end
        end
    end
end

%% select plot data as plotImage (dimension selection)

% check whether any input parameters specify which data to plot
plotDataSpecified = ismember(varargin(1:2:end), this.dimInfo.dimLabels);
% copy plot image for selection
plotImage = this.copyobj;

isPlotDataSpecified = any(plotDataSpecified);

% select plot data
if isPlotDataSpecified
    plotDataSpecified = repmat(plotDataSpecified, 2, 1);
    plotDataSpecified = reshape(plotDataSpecified, 1, []);
    stringSelection = varargin(plotDataSpecified);
else
    stringSelection = {};
    if ~useSlider % default: no slider used
        % 1 image with all samples of first three dimensions, for all further
        % dimensions only first sample is plotted
        if plotImage.dimInfo.nDims > 3
            nDimsSelect = plotImage.dimInfo.nDims - 3;
            dimLabelsSelect = plotImage.dimInfo.dimLabels;
            stringSelection(1:2:nDimsSelect*2) = dimLabelsSelect(4:end);
            stringSelection(2:2:nDimsSelect*2) = {1};
            [plotImage, selectionIndexArray] = plotImage.select('type', selectionType, ...
                stringSelection{:});
        else
            % create selectionIndexArray for labels later
            % simply select everything (by selecting everything in the
            % first dimension
            [~, selectionIndexArray] = plotImage.dimInfo.select(...
                plotImage.dimInfo.dimLabels{1}, 1:plotImage.dimInfo.nSamples(1));
        end
    else % use slider
        % 1 image with all samples of first FOUR dimensions, for all further
        % dimensions only first sample is plotted
        if plotImage.dimInfo.nDims > 4
            nDimsSelect = plotImage.dimInfo.nDims - 4;
            dimLabelsSelect = plotImage.dimInfo.dimLabels;
            stringSelection(1:2:nDimsSelect*2) = dimLabelsSelect(5:end);
            stringSelection(2:2:nDimsSelect*2) = {1};
        end
        
    end
end
[plotImage, selectionIndexArray] = plotImage.select('type', selectionType, ...
    stringSelection{:});


%% extract signal part, plot mode and display range

% signal part
switch signalPart
    case 'all'
        % do nothing, leave dataPlot as is
    case 'abs'
        plotImage = abs(plotImage);
    case {'angle', 'phase'}
        plotImage = angle(plotImage) + pi;
    case 'real'
        plotImage = real(plotImage);
    case 'imag'
        plotImage = imag(plotImage);
end

% linear or logarithmic plot
switch plotMode
    case 'linear' % nothing happens
    case 'log'
        plotImage = log(abs(plotImage));
end

% display range
if isempty(displayRange)
    if islogical(plotImage.data) % for logical arrays (masks)
        displayRange = [0 1];
    else
        displayRange = [min(plotImage), ...
            prctile(plotImage,98)];
        % check whether valid display range has been specified
        % if values equal or second range larger then first, specify by
        % hand
        if diff(displayRange) <= 0
            displayRange(2) = displayRange(1) + 1;
        end
    end
end


% Manipulate orientation for plot
if ischar(sliceDimension) % convert dimLabel to index
    sliceDimension = plotImage.dimInfo.get_dim_index(sliceDimension);
end
nDims = plotImage.dimInfo.nDims;
switch sliceDimension
    case 1
        permuteArray = [3 2 1 4 5:nDims];
        plotImage = permute(plotImage, permuteArray(1:nDims));
        selectionIndexArray = selectionIndexArray(permuteArray(1:nDims));
    case 2
        permuteArray = [1 3 2 4 5:nDims];
        plotImage = permute(plotImage, permuteArray(1:nDims));
        selectionIndexArray = selectionIndexArray(permuteArray(1:nDims));
    case 3
        %   as is...
    otherwise
        plotImage = permute(plotImage, [1 2 sliceDimension]);
        selectionIndexArray = selectionIndexArray([3 2 sliceDimension]);
end

if rotate90
    plotImage = rot90(plotImage, rotate90);
end

%% extract data for overlay image
if doPlotOverlays
    
    % check background image is 3D image
    nDimsPlotImage = sum(plotImage.dimInfo.nSamples > 1);
    is3dBackground = nDimsPlotImage < 4;
    if ~is3dBackground
        error('tapas:uniqc:MrImage:plot:UnderlayNot3D', ...
            ['Background image is not 3D but has ', ...
            num2str(nDimsPlotImage), ' dimensions.']);
    end
    % extract data from background image
    % extract plot data and sort
    plotData = squeeze(plotImage.data);
    backgroundNSamples = size(plotData);
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
    
    % settings
    nColorsPerMap   = 256;
    overlayImages   = reshape(overlayImages, [], 1);
    
    % loop over overlays and extract data
    nOverlays       = numel(overlayImages);
    dataOverlays    = cell(nOverlays,1);
    
    for iOverlay = 1:nOverlays
        thisOverlay = overlayImages{iOverlay};
        
        %% for map: overlayThreshold image only,
        %  for mask: binarize
        %  for edge: binarize, then compute edge
        
        switch overlayMode
            case {'map', 'maps'}
                thisOverlay.threshold(overlayThreshold);
            case {'mask', 'masks'}
                thisOverlay.threshold(0, 'exclude');
            case {'edge', 'edges'}
                thisOverlay.threshold(0, 'exclude');
                % for cluster mask with values 1, 2, ...nClusters,
                % leave values of edge same as cluster values
                thisOverlay = edge(thisOverlay,'sobel', edgeThreshold);
        end
        
        if any(plotDataSpecified)
            stringSelection = varargin(plotDataSpecified);
            [plotOverlay, ~, ~] = thisOverlay.select('type', selectionType, ...
                stringSelection{:});
        else
            plotOverlay = thisOverlay.copyobj;
        end
               
        switch sliceDimension
            case 1
                permuteArray = [3 2 1 4];
                plotOverlay = permute(plotOverlay, permuteArray(1:nDims));
            case 2
                permuteArray = [1 3 2 4];
                plotOverlay = permute(plotOverlay, permuteArray(1:nDims));
            case 3
                %   as is...
            otherwise
                plotOverlay = permute(plotOverlay, [1 2 sliceDimension]);
        end
        
        % apply rotation
        if rotate90
            plotOverlay = rot90(plotOverlay, rotate90);
        end
        % extract plot data and sort
        dataOverlays{iOverlay} = squeeze(plotOverlay.data);
        
        % check that background and overlay image have same dimension
        overlayNSamples = size(dataOverlays{iOverlay});
        equalDimBackgroundOverlay = ...
            numel(overlayNSamples) == numel(backgroundNSamples) && ...
            all(overlayNSamples == backgroundNSamples);
        if any(~equalDimBackgroundOverlay)
            error('tapas:uniqc:MrImage:plot:NumberOfSamplesUnderlayOverlayMismatch', ...
                ['Different number of samples for background (', ...
                num2str(backgroundNSamples), ') and overlay image (', ...
                num2str(size(dataOverlays{iOverlay})), ').']);
        end
    end
    
    
    % Define color maps for different cases:
    %   map: hot
    %   mask/edge: one color per mask image, faded colors for different
    %   clusters within same mask
    
    functionHandleColorMaps = {
        @hot
        @cool
        @spring
        @summer
        @winter
        @jet
        @hsv
        };
    
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
                    functionHandleColorMaps{iOverlay}(nColorsPerMap);
            end
            
    end
    
    % Assemble RGB-image for montage by adding overlays with transparency as
    % RGB in right colormap
    rangeOverlays   = cell(nOverlays, 1);
    rangeImage      = cell(nOverlays, 1);
    
    for iOverlay = 1:nOverlays
        [plotData, rangeOverlays{iOverlay}, rangeImage{iOverlay}] = ...
            tapas_uniqc_add_overlay(plotData, dataOverlays{iOverlay}, ...
            overlayColorMap{iOverlay}, ...
            overlayThreshold, ...
            overlayAlpha, ...
            displayRange);
    end
end
%% plot

% slider view
if useSlider
    % extract plot data
    plotData = squeeze(plotImage.data);
    % make sure only data with 4 or less dimension is used
    nDimsPlotImage = sum(plotImage.dimInfo.nSamples > 1);
    is4dor3dPlotImage = (nDimsPlotImage == 3 || nDimsPlotImage == 4);
    if ~is4dor3dPlotImage
        error('tapas:uniqc:MrImage:plot:SliderImageTooManyDimensions', ...
            ['Selected plot image is not 3D or 4D but has ', ...
            num2str(nDimsPlotImage), ' dimensions.']);
    end
    nSlices = plotImage.dimInfo.nSamples(3);
    tapas_uniqc_slider4d(plotData, @(Y, iDynSli, fh, yMin, yMax) ...
        tapas_uniqc_plot_abs_image(Y, iDynSli, fh, yMin, yMax, colorMap, colorBar), ...
        nSlices, displayRange(1), displayRange(2), this.name);
    
else % different plot types: montage, 3D, spm
    switch lower(plotType)
        case {'montage', 'labeledmontage'} % this is the default setting
            % make labels
            if strcmpi(plotType, 'labeledMontage') && plotImage.dimInfo.nDims >= 3
                stringLabels = cellfun(@(x,y) sprintf('%3.1f [%d]',x,y), ...
                    num2cell(plotImage.dimInfo.samplingPoints{imagePlotDim(3)}),...
                    num2cell(selectionIndexArray{imagePlotDim(3)}),...
                    'UniformOutput', false);
            else
                stringLabels = [];
            end
            
            % which dims need their own figure, i.e. are not in the image?
            dimsWithFig = setdiff(1:plotImage.dimInfo.nDims, imagePlotDim);
            if isempty(dimsWithFig), dimsWithFig = 4; end % for 3D data
            % how many additional dims are given
            nDimsWithFig = length(dimsWithFig);
            % extract plot data and sort
            if ~doPlotOverlays
                plotData = permute(plotImage.data, [imagePlotDim, dimsWithFig]);
                % number of samples in imagePlotDim
                nSamplesImagePlotDim = plotImage.dimInfo.nSamples(imagePlotDim(1:min(plotImage.dimInfo.nDims,3)));
                % reshape plot data to 4D matrix
                if plotImage.dimInfo.nDims > 3
                    % number of samples in dimsWithFig
                    nSamplesDimsWithFig = plotImage.dimInfo.nSamples(dimsWithFig);
                    plotData = reshape(plotData, ...
                        nSamplesImagePlotDim(1), nSamplesImagePlotDim(2), nSamplesImagePlotDim(3), []);
                else
                    % number of samples in dimsWithFig
                    nSamplesDimsWithFig = 1;
                end
                % total number of figures
                nFigures = size(plotData, 4);
            else
                nDimsWithFig = 1;
                nFigures = 1;
                nSamplesDimsWithFig = 1;
            end
            
            % now plot
            for n = 1:nFigures
                % make title string
                titleString = [];
                % sampling positions for titleString
                samplingPosArray = cell(1,nDimsWithFig);
                % convert index to subscript values for titleString
                [samplingPosArray{:}] = ind2sub(nSamplesDimsWithFig, n);
                % loop over nDimsWithFig
                for nTitle = 1:nDimsWithFig % number of labels in the title
                    % pos of label in dimInfo.dimLabel
                    labelPos = dimsWithFig(nTitle);
                    % build title string from label and corresponding sampling position
                    
                    if labelPos <= plotImage.dimInfo.nDims % 3D and smaller, have no label!
                        titleString = [titleString, ...
                            plotImage.dimInfo.dimLabels{labelPos}, ...
                            num2str(plotImage.dimInfo.samplingPoints{labelPos}(samplingPosArray{nTitle}), ...
                            '%4.0f') ' ']; %#ok<AGROW>
                    end
                end
                
                % add info to figure title, if only one slice
                if numel(stringLabels) == 1
                    titleString = plotImage.dimInfo.index2label(1,3);
                    titleString = titleString{1}{1};
                end
                
                titleString = tapas_uniqc_str2label([plotImage.name, ' ', titleString]);
                % open figure
                fh(n,1) = figure('Name', titleString, 'Position', ...
                    [1 1 FigureSize(1), FigureSize(2)], 'WindowStyle', windowStyle);
                % montage
                if doPlotOverlays
                    thisPlotData = plotData;
                else
                    thisPlotData = permute(plotData(:,:,:,n), [1, 2, 4, 3]);
                end
                if plotLabels
                    [~, montageSize] = tapas_uniqc_labeled_montage(thisPlotData, ...
                        'DisplayRange', displayRange, ...
                        'LabelsIndices', stringLabels, ...
                        'Size', [nRows nCols], ...
                        'FontSize', FontSize);
                else
                    montage(thisPlotData, ...
                        'DisplayRange', displayRange, ...
                        'ThumbnailSize', [], ...
                        'Size', [nRows nCols]);
                end
                
                resolutions = abs(plotImage.dimInfo.resolutions);
                samplingWidths = abs(plotImage.dimInfo.samplingWidths);
                % use sampling width if resolution if lost
                posNanResolutions = find(isnan(resolutions));
                if ~isempty(posNanResolutions)
                    for idxRes = posNanResolutions
                        % check sampling widths
                        if isnan(samplingWidths)
                            resolutions(idxRes) = 1;
                        else
                            resolutions(idxRes) = samplingWidths(idxRes);
                        end
                    end
                end
                resolutions((end+1):3) = 1;
                resolutions(4:end) = [];
                set(gca, 'DataAspectRatio', resolutions);
                
                % Display title, colorbar, colormap, if specified
                if plotTitle
                    title(titleString);
                end
                
                if doPlotColorBar
                    colorbar;
                end
                colormap(gca, colorMap);
                drawnow;
            end
            %
            
        case {'3d', 'ortho'}
            
            % check plot image is 3D image
            nDimsPlotImage = sum(plotImage.dimInfo.nSamples > 1);
            is3dPlotImage = nDimsPlotImage == 3;
            if ~is3dPlotImage
                error('tapas:uniqc:MrImage:plot:3DPlotImageNot3D', ...
                    ['Selected plot image is not 3D but has ', ...
                    num2str(nDimsPlotImage), ' dimensions.']);
            end
            % get voxel size ratio
            nonSingleDims = plotImage.dimInfo.nSamples ~=1;
            voxelSizeRatio = abs(plotImage.dimInfo.resolutions);
            voxelSizeRatio = abs(voxelSizeRatio(nonSingleDims));
            % call tapas_uniqc_view3d on plotImage data
            tapas_uniqc_view3d(squeeze(plotImage.data), voxelSizeRatio);
            if doPlotOverlays
                disp('Overlay function for plotType 3d not yet implemented.');
            end
        case {'spm', 'spminteractive', 'spmi'}
            % calls spm_image-function (for single volume) or
            % spm_check_registration (multiple volumes)
            
            fileNameNifti = plotImage.write_temporary_nifti_for_spm();
            if iscell(fileNameNifti) && numel(fileNameNifti) == 1
                fileNameNifti = fileNameNifti{1};
            elseif ischar(fileNameNifti)
            else
                error('tapas:uniqc:MrImage:plot:SPMPlotTooManyDimensions', ...
                    'SPM plots not implemented for 5+dimensional data yet');
            end
            
            % select Volumes
            fileNameVolArray = strvcat(tapas_uniqc_get_vol_filenames(fileNameNifti));
            
            % check if additional (overlay) images have been specified
            doPlotAdditionalImages = ~isempty(overlayImages);
            if doPlotAdditionalImages
                nAddImages = numel(overlayImages);
                for iAddImages = 1:nAddImages
                    fileNameAdditionalNiftis = overlayImages{iAddImages}.write_temporary_nifti_for_spm();
                    if iscell(fileNameAdditionalNiftis) && numel(fileNameAdditionalNiftis) == 1
                        fileNameAdditionalNiftis = fileNameAdditionalNiftis{1};
                    elseif ischar(fileNameAdditionalNiftis) && ~isempty(fileNameAdditionalNiftis)
                    else
                        error('tapas:uniqc:MrImage:plot:SPMPlotTooManyDimensions', ...
                            'High dimensional plotting with SPM not implemented yet');
                    end
                    volArrayFileNameNiftiAddImages{iAddImages} = strvcat(tapas_uniqc_get_vol_filenames(fileNameAdditionalNiftis));
                end
                
                fileNameVolArray = strvcat(fileNameVolArray, ...
                    volArrayFileNameNiftiAddImages{:});
            else
                fileNameAdditionalNiftis = '';
            end
            
            % display image
            nImages = size(fileNameVolArray, 1);
            if nImages == 1
                % use display option if only one image selected
                spm_image('Display', fileNameVolArray);
                
            else
                nMaxImages = 24;
                % check reg all if less than 25 (SPM only supports up to 24
                % volumes)
                % check if filenames are unique first
                spm_check_registration(fileNameVolArray(1:min(nImages,nMaxImages), :));
                if nImages > nMaxImages
                    warning('Only first 24 volumes are displayed');
                end
            end
            
            switch lower(plotType)
                case {'spminteractive', 'spmi'}
                    input('Press Enter to leave interactive mode');
            end
            
            % clean up temporary nifti files
            tapas_uniqc_delete_with_hdr(fileNameNifti);
            [~,~] = rmdir(fileparts(fileNameNifti));
            tapas_uniqc_delete_with_hdr(fileNameAdditionalNiftis);
            [~,~] = rmdir(fileparts(fileNameAdditionalNiftis));
            
    end % plotType
end % use Slider

if doLinkPlot
    ha = gca;
    hi = findobj(ha.Children,'Type','Image');
    hf = gcf;
    stringTitle = sprintf('Linked timeseries Plot %s', this.name);
    hFigLinePlot = figure('Name', stringTitle);
    hAxLinePlot = axes;
    
    iZ = find(cellfun(@(x) strcmpi(x, 'z'), stringSelection));
    if ~isempty(iZ)
        idxSlicePlotted = stringSelection{iZ+1};
    else
        idxSlicePlotted = 1;
    end
    
    if doMontage
        dimInfoSelection = plotImage.dimInfo;
        % conversion of coordinates follows from image size and number of
        % slices put into montage rows/columns
        linkOptions.convertMousePosToSelection = ...
            @(x) tapas_uniqc_convert_montage_position_to_selection(x, montageSize, ...
            dimInfoSelection, selectionIndexArray);
    else
        % single slice plot
        linkOptions.convertMousePosToSelection = @(x) [x(2) x(1) idxSlicePlotted];
    end
    
    hCallback = @(x,y) tapas_uniqc_lineplot_callback(x, y, this, hAxLinePlot, ...
        linkOptions.convertMousePosToSelection);
    ha.ButtonDownFcn = hCallback;
    hi.ButtonDownFcn = hCallback;
    hf.WindowButtonMotionFcn  = hCallback;
    
end
end