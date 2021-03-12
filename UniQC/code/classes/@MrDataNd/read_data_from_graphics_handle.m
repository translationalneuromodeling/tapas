function this = read_data_from_graphics_handle(this, inputHandle)
% Reads in data from graphics handle, trying to infer from plot type
%
%   Y = MrDataNd()
%   Y.read_data_from_graphics_handle(inputHandle)
%
% This is a method of class MrDataNd. It tries to retrieve reasonable
% plotted data from the specified figure/axes handle (e.g., CData) and
% convert them into and MrDataNd (usually 2D for Images).
% Additionally, it updates dimInfo according to data dimensions

%
% IN
%   inputHandle     graphics handle, i.e., figure/axes handle
% OUT
%
% EXAMPLE
%   Y.read_data_from_graphics_handle(gcf)
%   Y.read_data_from_graphics_handle(gca)
%   Y.read_data_from_graphics_handle(figure(121)); % to make distinction between fig handle and 1-element integer image
%
%   See also MrDataNd

% Author:   Lars Kasper
% Created:  2019-02-04
% Copyright (C) 2019 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

% check whether valid dimInfo now
% TODO: update dimInfo, but keeping information that is unaltered by
% changing data dimensions...
% e.g. via dimInfo.merge
hasDimInfo = isa(this.dimInfo, 'MrDimInfo');

% try to find image as handle children
hImage = findobj(inputHandle, 'Type', 'Image');
hLine = findobj(inputHandle, 'Type', 'Line');

isImagePlot = ~isempty(hImage);
isLinePlot = ~isempty(hLine);

if isImagePlot
    data = hImage.CData;
    samplingPoints = {};
elseif isLinePlot
    % specify raw data from classical line plots as well
    nSamples = numel(hLine(1).YData);
    nLines = numel(hLine);
    data = zeros(nSamples,nLines);
    for l = 1:nLines
        data(:,l) = hLine(l).YData.';
    end
    samplingPoints{1} = hLine(l).XData;
    samplingPoints{2} = 1:nLines;
    dimLabels = {'t', 'y'}; % TODO: read from axis labels
    units = {'s', '1'};
else
    error('tapas:uniqc:MrDataNd:ImageNotFound', 'No children of type Image found for specified handle');
end

this.data = data;

% remove singleton 2nd dimension kept by size command
nSamples = size(this.data);
if numel(nSamples) == 2
    nSamples(nSamples==1) = [];
end
resolutions = ones(1, numel(nSamples));

% set dimInfo or update according to actual number of samples
if ~hasDimInfo
    if isLinePlot
        this.dimInfo = MrDimInfo('samplingPoints', samplingPoints, ...
            'dimLabels', dimLabels, 'units', units);
    else
        this.dimInfo = MrDimInfo('nSamples', nSamples, ...
            'resolutions', resolutions);
    end
else
    if any(nSamples) % only update dimInfo, if any samples loaded
        if (numel(nSamples) ~= this.dimInfo.nDims)
            % only display the warning of an non-empty dimInfo (i.e. nDims
            % ~=0) has been given
            if (this.dimInfo.nDims ~=0)
                warning('Number of dimensions in dimInfo (%d) does not match dimensions in data (%d), resetting dimInfo', ...
                    this.dimInfo.nDims, numel(nSamples));
            end
            if isLinePlot
                this.dimInfo = MrDimInfo('samplingPoints', samplingPoints, ...
                    'dimLabels', dimLabels, 'units', units);
            else
                this.dimInfo = MrDimInfo('nSamples', nSamples, ...
                    'resolutions', resolutions);
            end
        elseif ~isequal(this.dimInfo.nSamples, nSamples)
            % if nSamples are correct already, leave it at that, otherwise:
            
            currentResolution = this.dimInfo.resolutions;
            
            isValidResolution = ~any(isnan(currentResolution)) || ...
                ~any(isinf(currentResolution)) && ...
                numel(currentResolution) == numel(nSamples);
            
            if isValidResolution
                % use update of nSamples to keep existing offset of samplingPoints
                this.dimInfo.nSamples = nSamples;
            else % update with default resolutions = 1
                this.dimInfo.set_dims(1:this.dimInfo.nDims, 'nSamples', ...
                    nSamples, 'resolutions', resolutions);
            end
            
        end
    end
end
