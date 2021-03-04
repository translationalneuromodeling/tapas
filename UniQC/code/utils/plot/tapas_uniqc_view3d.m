function [] = tapas_uniqc_view3d(img, voxelSizeRatio)
% Display 3D matrices along their 3 dimensions.
%
%   [] = tapas_uniqc_view3d(img)
%
% IN
%   img             arbitrary matrix (can be complex)
%   voxelSizeRatio  allows display of non-isotropic voxel size
%                   default: [1 1 1] for isotropic display
%
% NOTE: This function is an adaptation of vis3d from matlab fileexchange
% http://ch.mathworks.com/matlabcentral/fileexchange/37268-3d-volume-visualization/content/vis3d.m
%
%
%   fromain (froidevaux@biomed.ee.ethz.ch), ibt, university and eth zurich, switzerland
%   Lars Kasper (kasper@biomed.ee.ethz.ch):
%   - adapted for clicking cross hair and unequal voxel size
%   - changed default coordinate system to MNI/Talairach space instead of Philips
%% Figure handle with its callbacks
if nargin < 2
    voxelSizeRatio = [1 1 1];
end

mainHandle = figure('Position', [50 50 900 850], 'WindowButtonDownFcn', @buttonDownCallback, ...
    'WindowButtonUpFcn', @update3DViz); hold on;

%Show toolbar
set(mainHandle, 'Toolbar', 'figure');

%% Color map popup menu

%popup title
colormapText = uicontrol('Units','normalized', 'FontUnits', 'Normalized','Style', 'text', 'Position', [0.46 0.14 0.10 0.025],'FontSize', 0.7, 'FontWeight', 'bold');
set(colormapText, 'String', 'Colormap', 'ForegroundColor', 'k');

%popup menu
colormapPopup = uicontrol('Units','normalized','Style', 'popupmenu', 'Position', [0.46 0.1059 0.1444 0.0235], 'Callback', @specifyColormap);
set(colormapPopup, 'FontUnits', 'Normalized','String', {'jet'; 'gray'; 'multi-label';},'Value', 1, 'FontSize', 0.7, 'FontWeight', 'bold');

%define the colormaps.
multi = [0 0 0; 1 0 1; 0 .7 1; 1 0 0; .3 1 0; 0 0 1; 1 1 1; 1 .7 0];
graymap = colormap('gray');
jetmap = colormap('jet');
the_cmaps = {jetmap; graymap; multi;};

%% Description for user
newFigText = uicontrol('Units','normalized', 'FontUnits', 'Normalized', 'Style', 'text', 'Position', [0.11 0.05 0.78 0.018],'FontSize', 0.7);
set(newFigText, 'String', ...
    'Usage: 1) Click on subplot position to update slice position to voxel', ...
    'ForegroundColor', 'k');
newFigText2 = uicontrol('Units','normalized', 'FontUnits', 'Normalized', 'Style', 'text', 'Position', [0.11 0.02 0.78 0.018],'FontSize', 0.7);
set(newFigText2, 'String', ...
    'Alternative: 1) click to select subplot      2) scroll to navigate      3) click to update slice plot', ...
    'ForegroundColor', 'k');
%% Edit band (user can write a description of the figures)
Description = uicontrol('Units','normalized', 'FontUnits', 'Normalized','Style', 'edit','Position', [0.46 0.0706 0.4333 0.0235],'FontWeight', 'bold');

%% Plot selected slices in external figures

%popup title
newFigText = uicontrol('Units','normalized', 'FontUnits', 'Normalized', 'Style', 'text', 'Position', [0.63 0.14 0.08 0.025],'FontSize', 0.7, 'FontWeight', 'bold');
set(newFigText, 'String', 'Plot', 'ForegroundColor', 'k');

%popup menu
newFigPopup = uicontrol('Units','normalized','Style', 'popupmenu','Position', [0.63 0.1059 0.1111 0.0235],'Callback', @newFig);
set(newFigPopup, 'FontUnits', 'Normalized', 'String', {'XY'; 'XZ'; 'YZ'},'Value', 1, 'FontSize', 0.7, 'FontWeight', 'bold');

%% Chose between norm, angle, real or imaginary

%popup title
dataPartText = uicontrol('Units','normalized', 'FontUnits', 'Normalized','Style', 'text', 'Position', [0.76 0.14 0.1 0.025],'FontSize', 0.7, 'FontWeight', 'bold');
set(dataPartText, 'String', 'Data part', 'ForegroundColor', 'k');

%popup menuangle
dataPartPopup = uicontrol('Units','normalized', 'Style', 'popupmenu','Position', [0.76 0.1059 0.1333 0.0235],'Callback', @dataPart);
set(dataPartPopup,  'FontUnits', 'Normalized','String', {'norm'; 'angle'; 'real'; 'imaginary'},'Value', 3, 'FontSize', 0.7, 'FontWeight', 'bold');


%% Create Subplots

%Rotate image for it to correspond to scanner referential xyz
img0    = tapas_uniqc_flip_compatible(permute(img,[2 1 3]),1);
img     = real(img0);

%Initial slice number;
sizeImg = size(img);
sn = floor(round(sizeImg/2));

%Intensity range
imgRange = [min(img(:)) max(img(:))];
% verify that valid imgRange was specified
if imgRange(1) < imgRange(2)
else
    % set manually
    imgRange(2) = imgRange(1) + 1
end
imgRangeSliderValues = [prctile(img(:),3), prctile(img(:),97)];

origImgRange = imgRange;

%Initial color map
mycmap = 'jet';

%subplot titles
titleLines = {'XY view: ','XZ view: ','YZ view: '};
titleColors = {[.3 .3 .8], [.8 .3 .3], [.3 .8 .3]};

%% Create subplots
dimIndicesPlot = [
    2 1
    3 1
    3 2
    ];

positionSubPlots = [
    0.12,0.63,0.32,0.32
    0.57,0.63,0.32,0.32
    0.12,0.22,0.32,0.32
    0.57,0.22,0.32,0.32
    ];

for iPlot = 1:3
    dim1 = dimIndicesPlot(iPlot,1);
    dim2 = dimIndicesPlot(iPlot, 2);
    otherDim = setdiff(1:3, [dim1, dim2]);
    handles{iPlot} = subplot('Position', positionSubPlots(iPlot,:));
    imHandles{iPlot} = imagesc(getPlotData(iPlot), imgRange); colormap(mycmap);
    axis xy;
    titleHandles{iPlot} = title(handles{iPlot}, sprintf('%s%03d', titleLines{iPlot}, sn(otherDim)) ,'Color', ...
        titleColors{iPlot}, 'FontSize', 14, 'FontWeight', 'bold');
    set(gca, 'DataAspectRatio', [voxelSizeRatio(dim2) voxelSizeRatio(dim1), 1]);
end


%% Generate the lines representing the orthogonal planes.

lineHandles = {};

for iPlot = 1:3
    dim1 = dimIndicesPlot(iPlot,1);
    dim2 = dimIndicesPlot(iPlot,2);
    lines{iPlot}.x = [1 sizeImg(dim1); sn(dim1) sn(dim1)]';
    lines{iPlot}.y = [sn(dim2) sn(dim2); 1 sizeImg(dim2)]';
    
    subplot(handles{iPlot});
    lineHandles{iPlot}(1) = line(lines{iPlot}.x(:,1), lines{iPlot}.y(:,1), ...
        'Color', titleColors{dimIndicesPlot(4-iPlot,1)}, 'LineWidth', 1);
    lineHandles{iPlot}(2) = line(lines{iPlot}.x(:,2), lines{iPlot}.y(:,2), ...
        'Color', titleColors{dimIndicesPlot(4-iPlot,2)}, 'LineWidth', 1);
end


%% 3D plot

handles{4}      = subplot('Position', positionSubPlots(4,:));
camPos3D        = get(handles{4}, 'CameraPosition');

dimPermute3D    = [3 2 1];
sliceImg        = permute(img, dimPermute3D);
update3DViz();

set(gca, 'DataAspectRatio', voxelSizeRatio(dimPermute3D));


%% Color scale uicontrols

% Title
IntWinText = uicontrol('Units','normalized', 'FontUnits', 'Normalized','Style', 'text', 'Position', [0.11 0.14 0.17 0.025],'FontSize', 0.7, 'FontWeight', 'bold');
set(IntWinText, 'String', 'Color scale range', 'ForegroundColor', 'k');

% Min range slider
IntWinSliderHandle{1} = uicontrol('Units','normalized','Style', 'slider', 'Position', [0.11 0.0706 0.2222 0.023],'Callback', @IntWinBound);
set(IntWinSliderHandle{1}, 'Value', imgRangeSliderValues(1));

% Min range edit
IntWinTextBound{1} = uicontrol('Units','normalized','FontUnits', 'Normalized','Style', 'edit', 'Position', [0.3611  0.0706  0.0667 0.0235], 'Callback', @defineBound);
set(IntWinTextBound{1}, 'String', imgRangeSliderValues(1));

% Max range slider
IntWinSliderHandle{2} = uicontrol('Units','normalized','Style', 'slider','Position', [0.11 0.1059  0.2222 0.0235], 'Callback', @IntWinBound);
set(IntWinSliderHandle{2}, 'Value', imgRangeSliderValues(2));

% Max range edit
IntWinTextBound{2} = uicontrol('Units','normalized','FontUnits', 'Normalized','Style', 'edit','Position', [0.3611 0.1059 0.0667 0.0235], 'Callback', @defineBound);
set(IntWinTextBound{2}, 'String', imgRangeSliderValues(2));

% Define sliders limits
for i = 1:2
    set(IntWinSliderHandle{i}, 'Min', imgRange(1));
    set(IntWinSliderHandle{i}, 'Max', imgRange(2));
end


%% Callbacks

%Variable definition
whichView       = 1;
offsliceCoord   = [3 2 1];


%% Returns mouse position coordinates within current axis
% based on http://www.mathworks.com/matlabcentral/fileexchange/24861-41-complete-gui-examples/content/GUI_27.m
    function [Cx, Cy] = getPositionInAxis()
        S.ax = handles{whichView};
        S.un = get(S.ax, 'unit');
        set(S.ax,'unit','pix')
        S.fh = get(S.ax, 'Parent');
        S.XLM = get(S.ax,'xlim');
        S.YLM = get(S.ax,'ylim');
        S.AXP = get(S.ax,'pos');
        S.DFX = diff(S.XLM);
        S.DFY = diff(S.YLM);
        
        F = get(S.ax, 'currentPoint');
        % ranges always go from 0.5 to nVoxel +0.5, correct here
        S.XLM = S.XLM + [0.5 -0.5];
        S.YLM = S.YLM + [0.5 -0.5];
        % Choose coordinate within range of limits
        Cx = min(max(F(1,1), S.XLM(1)), S.XLM(2));
        % Y coordinate always seems 1 too little...
        Cy = min(max(F(1,2)+1, S.YLM(1)), S.YLM(2));
        
        set(S.ax, 'unit', S.un);
        
    end


%% Update slice and title and move lines
    function updateAllSliceViz()
        for iView = 1:3
            subplot(handles{whichView});
            switch iView
                case 1
                    set(imHandles{1}, 'CData', getPlotData(iView));
                    set(lineHandles{2}(2), 'XData', [sn(3) sn(3)]');
                    set(lineHandles{3}(2), 'XData', [sn(3) sn(3)]');
                case 2
                    set(imHandles{2}, 'CData', getPlotData(iView));
                    set(lineHandles{1}(2), 'XData', [sn(2) sn(2)]');
                    set(lineHandles{3}(1), 'YData', [sn(2) sn(2)]');
                case 3
                    set(imHandles{3}, 'CData', getPlotData(iView));
                    set(lineHandles{1}(1), 'YData', [sn(1) sn(1)]');
                    set(lineHandles{2}(1), 'YData', [sn(1) sn(1)]');
            end
            set(titleHandles{iView}, 'String', ...
                sprintf('%s%03d', titleLines{iView}, sn(offsliceCoord(iView))));
        end
    end


%% Defines what subplot will be active
    function buttonDownCallback(varargin)
        
        whichView = find(cell2mat(cellfun( @(x) isequal(x,gca), ...
            handles, 'UniformOutput', false)));
        
        [axX, axY] = getPositionInAxis();
        
        if ~(whichView==4 || isempty(axX) || isempty(axY))
            % define coordinate index with position change
            % 1st plot Z=3 missing, 2nd plot Y=2 missing, 3rd plot X=1 missing, i.e 4-whichView
            iCoords = setdiff(1:3, 4-whichView);
            % X&Y always swapped in imagesc in Matlab
            sn(iCoords) = [round(axY), round(axX)];
            updateAllSliceViz();
        end
        
        %By getting the camera position on a click, we can maintain the 3d
        %perspective in the case of changing slices.
        camPos3D = get(handles{4}, 'CameraPosition');
        set(mainHandle, 'WindowScrollWheelFcn', @scrollCallback);
    end

%% Update 3D plot
    function update3DViz(varargin)
        axes(handles{4});
        
        hslc = slice(sliceImg, sn(2), sn(3), sn(1));
        
        set(hslc(1:3),'LineStyle','none');
        xlabel 'Y' ;ylabel 'Z' ;zlabel 'X';
        
        set(handles{4}, 'CameraPosition', camPos3D);
        set(handles{4} , 'CLim', imgRange);
    end

%% Change slice when scrolling
    function scrollCallback(src,evnt)
        if whichView ~= 4
            if evnt.VerticalScrollCount > 0
                newslice = sn(offsliceCoord(whichView)) - 1;
            else
                newslice = sn(offsliceCoord(whichView)) + 1;
            end
            updateSliceViz(newslice);
        else
            return;
        end
    end

%% Update slice and title and move lines
    function updateSliceViz(newslice)
        if newslice > 0 && newslice <= sizeImg(offsliceCoord(whichView))
            sn(offsliceCoord(whichView)) = newslice;
        end
        subplot(handles{whichView});
        if whichView == 1
            set(imHandles{1}, 'CData', getPlotData(whichView));
            set(lineHandles{2}(2), 'XData', [sn(3) sn(3)]');
            set(lineHandles{3}(2), 'XData', [sn(3) sn(3)]');
        elseif whichView == 2
            set(imHandles{2}, 'CData', getPlotData(whichView));
            set(lineHandles{1}(2), 'XData', [sn(2) sn(2)]');
            set(lineHandles{3}(1), 'YData', [sn(2) sn(2)]');
        else
            set(imHandles{3}, 'CData', getPlotData(whichView));
            set(lineHandles{1}(1), 'YData', [sn(1) sn(1)]');
            set(lineHandles{2}(1), 'YData', [sn(1) sn(1)]');
        end
        set(titleHandles{whichView}, 'String', ...
            sprintf('%s%03d', titleLines{whichView}, sn(offsliceCoord(whichView))));
    end

%% Change the color scale range
    function IntWinBound(varargin)
        %Write slider values in imgRange
        for whichSlider = 1:2
            slider_value = get(IntWinSliderHandle{whichSlider}, 'Value');
            imgRange(whichSlider) = slider_value;
            set(IntWinTextBound{whichSlider}, 'String', imgRange(whichSlider));
        end
        %change the CLim of the four axes.
        for j=1:4
            set(handles{j} , 'CLim', imgRange);
        end
    end

%% Change color range manually (by writting in the edit block)
    function defineBound(varargin)
        for whichBound = 1:2
            bound_str = get(IntWinTextBound{whichBound},'String');
            if strcmp(bound_str, 'max')
                bound_value = max(origImgRange);
            elseif strcmp(bound_str, 'min')
                bound_value = min(origImgRange);
            else
                bound_value = str2double(bound_str);
            end
            set(IntWinTextBound{whichBound},'String', num2str(bound_value));
            imgRange(whichBound) = bound_value;
            set(IntWinSliderHandle{whichBound}, 'value', bound_value);
        end
        
        for j=1:4
            set(handles{j} , 'CLim', imgRange);
        end
    end


%% Specify color maps
    function specifyColormap(hObject, ~)
        choice = get(hObject, 'Value');
        colormap(the_cmaps{choice});
    end


%% Plot external figure
    function newFig(hObject, ~)
        iView = get(hObject,'value');
        figure
        imagesc(getPlotData(iView), imgRange);
        colormap(the_cmaps{get(colormapPopup, 'Value')});
        axis xy;
        switch iView
            case 1
                stringTitle = sprintf('XY slice, Z = %03d', sn(3));
            case 2
                stringTitle = sprintf('XZ slice, Y = %03d', sn(2));
            case 3
                stringTitle = sprintf('YZ slice, X = %03d', sn(1));
        end
        
        dim1 = dimIndicesPlot(iView,1);
        dim2 = dimIndicesPlot(iView, 2);
        otherDim = setdiff(1:3, [dim1, dim2]);
        
        set(gca, 'DataAspectRatio', voxelSizeRatio([dim1 dim2 otherDim]));
        
        pfxString = get(Description, 'String');
        if ~isempty(pfxString)
            stringTitle = sprintf('%s, %s',pfxString, stringTitle);
        end
        set(gcf, 'Name', stringTitle);
        title(stringTitle);
    end


%% Change data part (norm, angle, real, imaginary)
    function dataPart(hObject,~)
        choice = get(hObject,'value');
        switch choice
            case 1
                img=abs(img0);
            case 2
                img=angle(img0);
            case 3
                img=real(img0);
            case 4
                img=imag(img0);
        end
        
        %Redefine color scaling for plots 1-3
        imgRange = [min(img(:)) max(img(:))];
        
        for j = 1:2
            set(IntWinSliderHandle{j}, 'Min', imgRange(1));
            set(IntWinSliderHandle{j}, 'Max', imgRange(2));
        end
        
        for j=1:3
            set(handles{j} , 'CLim', imgRange);
        end
        
        imgRangeSliderValues = [prctile(img(:),3), prctile(img(:),97)];
        set(IntWinSliderHandle{1}, 'Value', imgRangeSliderValues(1));
        set(IntWinSliderHandle{2}, 'Value', imgRangeSliderValues(2));
        set(IntWinTextBound{1}, 'String', imgRangeSliderValues(1));
        set(IntWinTextBound{2}, 'String', imgRangeSliderValues(2));
        
        for iView = 1:3
            set(imHandles{iPlot}, 'CData', getPlotData(iView));
        end
        
        % Redefine color scaling for plot 4
        axes(handles{4});
        sliceImg = permute(img, dimPermute3D);
        
        update3DViz();
        
    end

% Returns correct 2D slice data to plot for specified subplot
    function plotData = getPlotData(iView)
        switch iView
            case 1
                plotData = img(:,:,sn(3));
            case 2
                plotData = squeeze(img(:,sn(2),:));
            case 3
                plotData = squeeze(img(sn(1),:,:));
        end
    end
end

