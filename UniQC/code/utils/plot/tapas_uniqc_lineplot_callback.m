function currentMousePosition = tapas_uniqc_lineplot_callback(hObject, eventdata, Img, ...
    hAxLinePlot, fnConvertMousePosToSelection)
% provides a callback function to plot the non-displayed dimension in a
% figure display
%
%   currentMousePosition = tapas_uniqc_lineplot_callback(hObject, eventdata, Img, ...
%                               hAxLinePlot)
%
% IN
%   fnConvertMousePosToSelection
%                       function handle to convert mouse cursor position to
%                       selection (x,y,z)
%                       default:    swap first and second coordinate to
%                                   reflect dim order difference in Matlab
%                                   image plot and matrix representation
% OUT
%   currentMousePosition
%
% EXAMPLE
%    hCallback = @(x,y) tapas_uniqc_lineplot_callback(x, y, this, hAxLinePlot);
%    ha.ButtonDownFcn = hCallback;
%    MrImage.plot('linkOptions', 'ts_4')
%
%   See also MrImage.plot demo_plot_images

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2018-12-28
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.
%

if nargin < 5
    % default: swap first and second coordinate to reflect dim order
    % difference in Matlab image plot and matrix representation
    fnConvertMousePosToSelection = @(x) [x(2) x(1) 1];
end

% mouse position found on different (sub-)0bjects, depending on caller
% (figure, axes or image itself)
switch class(hObject)
    case 'matlab.ui.Figure'
        currentMousePosition = round(hObject.Children(1).CurrentPoint(1,1:2));
        hf = hObject;
    case 'matlab.graphics.axis.Axes'
        currentMousePosition = round(hObject.CurrentPoint(1,1:2));
        hf = hObject.Parent;
    case 'matlab.graphics.primitive.Image'
        currentMousePosition = round(hObject.Parent.CurrentPoint(1,1:2));
        hf = hObject.Parent.Parent;
end

stringTitle = sprintf('%s on %s at (%d,%d)', eventdata.EventName, class(hObject), ...
    currentMousePosition(1), currentMousePosition(2));

% command line verbosity, but already in figure title
% disp(currentMousePosition);
% disp(stringTitle);

% mix-up of 1st and second dim in arrays and their display in Matlab
currentSelection = fnConvertMousePosToSelection(currentMousePosition);

stringLegend = sprintf('voxel [%d %d %d]', currentSelection(1), ...
    currentSelection(2), currentSelection(3));

isRightClick = strcmpi(eventdata.EventName, 'Hit') && strcmpi(hf.SelectionType, 'alt');

% update current plot data by time series from current voxel
nSamples =  Img.dimInfo.nSamples({'x', 'y', 'z'}); % TODO: dimensionality independence
if all(currentSelection <= nSamples) && all(currentSelection >= 1)
    % add current mouse position and respective plot data to figure
    % UserData variable
    currentPlotData = squeeze(Img.data(currentSelection(1), ...
        currentSelection(2),currentSelection(3),:));
    hf.UserData.PlotData(:,1) = currentPlotData;
    hf.UserData.MousePositions(1,:) = currentMousePosition;
    hf.UserData.selections(1,:) = currentSelection;
    hf.UserData.stringLegend{1} = stringLegend;
    switch eventdata.EventName
        case 'Hit'
            if ~isRightClick
                % add new fixed line from time series of current voxel to plot
                hf.UserData.selections(end+1,:) = currentSelection;
                hf.UserData.MousePositions(end+1,:) = currentMousePosition;
                hf.UserData.PlotData(:,end+1) = currentPlotData;
                hf.UserData.stringLegend{end+1} = stringLegend;
            end
        otherwise
            % just replace current line
    end
    guidata(hf);
    hLines = plot(hAxLinePlot, hf.UserData.PlotData);
    title(hAxLinePlot, stringTitle);
    legend(hAxLinePlot, hf.UserData.stringLegend);
end

%% option to stop interactive call back by right-click and remove current line
if isRightClick
    % TODO: find right axis e.g., via generic .Tag in plot axes, set by .plot
    ha = findobj(hf.Children, 'Type', 'Axes');
    hi = findobj(ha.Children,'Type','Image');
    
    % detach call backs to keep static figure as is
    ha.ButtonDownFcn = '';
    hi.ButtonDownFcn = '';
    hf.WindowButtonMotionFcn  = '';
    
    % first children is current line that alters when moving mouse and is
    % deleted when stopping the interactive plotting
    delete(hLines(1));
end