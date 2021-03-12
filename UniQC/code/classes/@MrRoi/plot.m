function figureHandles = plot(this, varargin)
% Different plot modalities for Roi data, including time series & box plots
% showing a couple of statistics for time series or volume ROI data, e.g.
%   mean +/- sd,
%   min/median/max
%   boxplot/histogram (distribution over voxels)
% All plots can show statistics either for individual slices or pooled over
% the whole volume
%
%
%   Y = MrRoi()
%   Y.plot('ParameterName', ParameterValue, ...);
%)
%
% This is a method of class MrRoi.
%
% IN
%   varargin    'ParameterName', 'ParameterValue'-pairs for the following
%               properties:
%               'plotType'  type of plot that is created
%                           'timeSeries'/'ts'/'time'/'line'
%                               (default for 4D data extracted)
%                           'histogram'/'hist'
%                               (default for 3D data extracted)
%                           'boxplot'/'box' TODO!!!
%               'axisType' 'absolute' (default) or 'relative'
%                           for histograms. If relative, percentage of
%                           voxels are displayed
%               'dataGrouping'
%                           'perSlice'
%                           'perVolume',
%                           'both/all' (default)
%               'statType'   string or cell of strings specifying what
%                            statistic to plot;
%                            'mean', 'sd', 'snr', 'min', 'max', 'median'
%                               - if one of these strings is specified,
%                               only this statistics is plotted for each
%                               subplot (subplots = slices for 4D data)
%                               - if a cell of these strings is specified,
%                                 all statistis are plotted as different
%                                 lines within one plot, e.g.
%                                   {'min', 'median', 'max'} plots these
%                                   three together
%                            'mean+sd'  mean with shaded +/- standard
%                                       deviation area
%                            'mean+sem' mean with shaded +/- standard error
%                                       of the mean area (default for 4D)
%  TODO:                     'data' plot raw data (of all voxels,
%                                   warning: BIG!
%  TODO:                    'nVoxels'     integer for statType 'data': plot how many voxels?
%               'indexVoxels' vector of voxel indices to be plot (mutually
%                             exclusive to 'nVoxel')
%
%               'fixedWithinFigure' determines what dimension is plotted in
%                                  (subplots of) 1 figure
%                             'slice(s)'    all slices in 1 figure; new figure
%                                           for each volume
%                             'volume(s)'   all volumes in 1 figurel new figure
%                                           for each slice
%                                               pixel value = white
%               'selectedVolumes' [1,nVols] vector of selected volumes to
%                                           be displayed
%               'selectedSlices' [1,nSlices] vector of selected slices to
%                                           be displayed
%                                 choose Inf to display all volumes
%
% TODO!!!
%               'useSlider'     true or false
%                               provides interactive slider for
%                               slices/volumes;
%                               assumes default: selectedSlices = Inf
%                                                selectedVolumes = Inf
%
% OUT
%
% EXAMPLE
%   plot
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


%% Defines and checks input arguments
defaults.displayRange = 0.8*[this.perVolume.min, this.perVolume.max];
defaults.selectedVolumes = Inf;
defaults.selectedSlices = Inf;
defaults.useSlider = false;
defaults.dataGrouping = '';
defaults.LineWidth = 2;
defaults.plotType = ''; % hist or line(time/series) or box?
defaults.statType = '';
defaults.fixedWithinFigure ='slice';
defaults.axisType = 'absolute';
defaults.windowStyle = 'docked';

% display
defaults.FigureSize             = [1600 900];
args = tapas_uniqc_propval(varargin, defaults);
tapas_uniqc_strip_fields(args);


% slider enables output of all Slices and Volumes per default, strip data
% again under this assumption, if slider is used for display
if useSlider
    defaults.selectedVolumes = Inf;
    defaults.selectedSlices = Inf;
    args = tapas_uniqc_propval(varargin, defaults);
    tapas_uniqc_strip_fields(args);
end

% convert Inf to actual number of volumes/slices
if isinf(selectedVolumes)
    selectedVolumes = 1:this.nVolumes;
end

if isinf(selectedSlices)
    selectedSlices = 1:this.nSlices;
end

% discard slices without data

emptyDataSlices = find(cellfun(@isempty, this.data));
selectedSlices = setdiff(selectedSlices, emptyDataSlices);

% discard non-existing slices/volumes;
selectedSlices = intersect(1:this.nSlices, selectedSlices);
selectedVolumes = intersect(1:this.nVolumes, selectedVolumes);

nSlices = numel(selectedSlices);
nVolumes = numel(selectedVolumes);

% if specific slices are specified, we assume no whole volume plot is
% needed!
if isempty(dataGrouping)
    dataGrouping = 'both';
end

is3D = nVolumes == 1;

% set new default plot type
if isempty(plotType)
    if is3D
        plotType = 'hist';
    else
        plotType = 'line';
    end
end

if isempty(statType)
    if is3D
        statType = 'mean';
    else
        statType = 'mean+sem';
    end
end

if ismember(statType, {'mean+sd', 'mean+sem'})
    statTypeArray = {'mean', 'sd'};
else
    
    if ~iscell(statType)
        statType = {statType};
    end
    
    statTypeArray = cellstr(statType);
    
end


isStandardErrorMean = strcmpi(statTypeArray, 'mean+sem');


if isempty(this.data)
    error('tapas:uniqc:MrRoi:EmptyDataMatrix', ...
        'Data matrix empty for MrImage-object %s', this.name);
end


%% Create one data stack for volumes and slices:
% all selected slices first, then volume stats!

nStatTypes = numel(statTypeArray);

switch dataGrouping
    case {'sli', 'slice'}
        dataGrouping = 'perSlice';
    case {'vol', 'volume'}
        dataGrouping = 'perVolume';
end

switch dataGrouping
    case 'perSlice'
        nPlots = nSlices;
    case 'perVolume'
        nPlots = 1;
    case {'both', 'all'}
        nPlots = nSlices + 1;
end

% other dims = not x,y,z; e.g. volumes, coils...
nSamplesOtherDims = max(cell2mat(cellfun(@size, this.data, 'UniformOutput', false)));
nSamplesOtherDims = nSamplesOtherDims(2:end);
nOtherDims = numel(nSamplesOtherDims);
selectionStringOtherDims = repmat({':'}, 1, nOtherDims);

dataPlotArray = zeros([nPlots, nSamplesOtherDims, nStatTypes]);
doPlotSliceOnly = strcmpi(dataGrouping, 'perSlice');

for iStatType = 1:nStatTypes
    for iPlot = 1:nPlots-1
        indSlice = selectedSlices(iPlot);
        dataPlotArray(iPlot, selectionStringOtherDims{:}, iStatType) = ...
            this.perSlice.(statTypeArray{iStatType})(indSlice,selectionStringOtherDims{:});
        
        if isStandardErrorMean(iStatType) && isequal(statTypeArray{iStatType}, 'sd')
            dataPlotArray(iPlot, selectionStringOtherDims{:}, iStatType) = ...
                dataPlotArray(iPlot, selectionStringOtherDims{:}, iStatType) ./ ...
                sqrt(this.perSlice.nVoxels(indSlice));
        end
        
    end
    
    if doPlotSliceOnly
        % TODO: 4D...selected slices!
        % last row is slice
        dataPlotArray(nPlots, selectionStringOtherDims{:}, iStatType) = ...
            this.perSlice.(statTypeArray{iStatType})(indSlice,selectionStringOtherDims{:});
    else
        % last row is volume
        dataPlotArray(nPlots, selectionStringOtherDims{:}, iStatType) = ...
            this.perVolume.(statTypeArray{iStatType});
        
        if isStandardErrorMean(iStatType) && isequal(statTypeArray{iStatType}, 'sd')
            dataPlotArray(nPlots, selectionStringOtherDims{:}, iStatType) = ...
            dataPlotArray(nPlots, selectionStringOtherDims{:}, iStatType) ./ ...
            sqrt(this.perVolume.nVoxels);
        end
        
    end
    
end

% Create one new plot for each specified plot type, and also for different
% slices or volumes
switch lower(plotType)
    case {'line', 'ts', 'timeseries'}
        % TODO: get time series information from image geometry!
        
        t = selectedVolumes;
        
        % create string mean+std or min+median+max etc for title of plot
        nameStatType = sprintf('%s+', statTypeArray{:});
        nameStatType(end) = [];
        
        
        stringTitle = sprintf('Roi plot (%s) for %s', nameStatType, ...
            this.name);
        figureHandles(1, 1) = figure('Name', stringTitle, 'Position', ...
            [1 1 FigureSize(1), FigureSize(2)], 'WindowStyle', windowStyle);
        % create one subplot per slice, and one for the whole volume
        nRows = floor(sqrt(nPlots));
        nCols = ceil(nPlots/nRows);
        
        for iPlot = 1:nPlots
            subplot(nRows, nCols, iPlot);
            
            switch nameStatType
                case {'mean+sd', 'mean+sem'}
                    y = squeeze(dataPlotArray(iPlot,selectionStringOtherDims{:},1))';
                    SD = squeeze(dataPlotArray(iPlot,selectionStringOtherDims{:},2))';
                    harea = area(t,[y-SD,SD,SD]);
                    hold on;
                    
                    % create shaded colors for background
                    colors = get(gca, 'DefaultAxesColorOrder');
                    nColors = size(colors,1);
                    faceAlpha  = 0.7;
                    shadedColors = faceAlpha*repmat([1 1 1], nColors,1) + ...
                        (1-faceAlpha)*colors;
                    if ~tapas_uniqc_isNewGraphics
                        harea = num2cell(harea);
                        for h = 1:numel(harea)
                            harea{h} = get(harea{h},'Children');
                        end
                        harea = cell2num(harea);
                    end
                    
                    set(harea(1),'EdgeColor','None', ...
                        'FaceColor','none');
                    set(harea(2),'FaceColor', ...
                        shadedColors(1,:));
                    set(harea(3),'FaceColor',...
                        shadedColors(1,:));
                    h(2) = plot(t,y);
                    set(h(2),'LineWidth', LineWidth, 'Color', colors(1,:), ...
                        'LineStyle', '-');
                    
                otherwise % any other combination...
                    plot(t, squeeze(dataPlotArray(iPlot, ...
                        selectionStringOtherDims{:},:)));
            end
            
            if ~doPlotSliceOnly && iPlot == nPlots
                title('Whole Volume')
            else
                title(sprintf('Slice %d', selectedSlices(iPlot)));
            end
            
            xlim([t(1) t(end)]);
        end
        % add legend
        switch nameStatType
            case 'mean+sd'
                legend([h(2), harea(2)], {'mean', 'sd'});
            case 'mean+sem'
                legend([h(2), harea(2)], {'mean', 'sem'});
            otherwise
                legend(statTypeArray, 'location', 'best');
        end
        if exist('suptitle')
            suptitle(sprintf('Line plot (%s) for ROI %s ', ...
                tapas_uniqc_str2label(nameStatType), str2label(this.name)));
        end
    case {'hist', 'histogram'}
        for iStatType = 1:nStatTypes
            currentStatType = statTypeArray{iStatType};
            
            stringTitle = sprintf('Roi plot (%s) for %s', currentStatType, ...
                tapas_uniqc_str2label(this.name));
            
            figureHandles(iStatType, 1) = figure('Name', stringTitle);
            
            if is3D
                dataAllSlices = cat(1, this.data{:});
                set(figureHandles(iStatType, 1), 'Name', ...
                    [stringTitle, ' - Abs']);
                
                if any(~isreal(dataAllSlices(:)))
                    figureHandles(iStatType, 2) = figure('Name', ...
                        [stringTitle, ' - Angle']);
                    funArray = {@abs, @angle};
                else
                    funArray = {@(x) x};
                end
                
                for iFun = 1:numel(funArray)
                    figure(figureHandles(iStatType, iFun));
                    % Plot phase and abs separately for complex data
                    
                    %nPlots = nSlices+1;
                    nRows = floor(sqrt(nPlots));
                    nCols = ceil(nPlots/nRows);
                    
                    % plot all selected slices, but leave one subplot for
                    % volume summary, if requested
                    if doPlotSliceOnly
                        nPlotsSlices = nPlots;
                    else
                        nPlotsSlices = nPlots -1 ;
                    end
           
                    for iPlot = 1:nPlotsSlices
                        indSlice = selectedSlices(iPlot);
                        nVoxels = this.perSlice.nVoxels(indSlice);
                        plotMedian = this.perSlice.median(indSlice,end);
                        plotMean = this.perSlice.mean(indSlice,end);
                        nBins = max(10, nVoxels/100);
                        dataPlot = funArray{iFun}(this.data{indSlice});
                        
                        subplot(nRows,nCols, iPlot);
                        hist(dataPlot, nBins); hold all;
                        
                        tapas_uniqc_vline([funArray{iFun}(...
                            plotMean), ...
                            funArray{iFun}(plotMedian)], ...
                            {'r', 'g'}, {'mean', 'median'});
                        
                        % create percentage of voxels labels for y Axis
                        if strcmpi(axisType, 'relative')
                            ha = gca;
                            set(ha, 'YTickLabel', ...
                                cellfun(@(x) sprintf('%2.0f %%', x), num2cell((ha.YTick/nVoxels)*100), 'UniformOutput', false));
                        end
                        
                        
                        title({sprintf('Slice %d (%d voxels)', ...
                            indSlice, nVoxels), ...
                            sprintf('Mean = %.2f, Median = %.2f', plotMean, plotMedian)});
                    end
                    
                    %% volume summary histogram plot
                    if ~doPlotSliceOnly
                        subplot(nRows,nCols, nPlots);
                        nVoxels = this.perVolume.nVoxels;
                        plotMedian = this.perVolume.median;
                        plotMean = this.perVolume.mean;
                        nBins = max(10, nVoxels/100);
                        dataPlot = funArray{iFun}(dataAllSlices);
                        
                        hist(dataPlot, nBins);
                        hold on;
                        tapas_uniqc_vline([funArray{iFun}(plotMean), ...
                            funArray{iFun}(plotMedian)], ...
                            {'r', 'g'}, {'mean', 'median'});
                        
                        % create percentage of voxels labels for y Axis
                        if strcmpi(axisType, 'relative')
                            ha = gca;
                            set(ha, 'YTickLabel', ...
                                cellfun(@(x) sprintf('%2.0f %%', x), num2cell((ha.YTick/nVoxels)*100), 'UniformOutput', false))
                        end
                        
                        title({sprintf('Whole Volume (%d voxels)', nVoxels), ...
                            sprintf('Mean = %.2f, Median = %.2f', plotMean, plotMedian)});
                    end
                    if exist('suptitle')
                        suptitle(get(figureHandles(iStatType, iFun), 'Name'));
                    end
                end
            else
                % don't know yet...
            end
            
        end
    otherwise % mean sd etc
        
end