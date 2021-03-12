function [dataPlot, displayRange, resolution_mm] = extract_plot_data(this, varargin)
% Extracts (and manipulates) data for plotting with arguments from MrImage.plot
%
%   Y = MrImage()
%   Y.extract_plot_data('ParameterName', 'ParameterValue' ...)
%
% This is a method of class MrImage.
%
% IN
%   varargin    'ParameterName', 'ParameterValue'-pairs for the following
%               properties:
%
%               'signalPart'        for complex data, defines which signal
%                                   part shall be extracted for plotting
%                                       'all'       - do not change data
%                                                     (default)
%                                       'abs'       - absolute value
%                                       'phase'     - phase of signal
%                                       'real'      - real part of signal
%                                       'imag'      - imaginary part of
%                                                     signal
%               'plotMode',         transformation of data before plotting
%                                   'linear' (default), 'log'
%               'selectedX'         [1, nPixelX] vector of selected
%                                   pixel indices in 1st image dimension
%               'selectedY'         [1, nPixelY] vector of selected
%                                   pixel indices in 2nd image dimension
%               'selectedVolumes'   [1,nVols] vector of selected volumes to
%                                             be displayed
%               'selectedSlices'    [1,nSlices] vector of selected slices to
%                                               be displayed
%                                   choose Inf to display all volumes
%               'sliceDimension'    (default: 3) determines which dimension
%                                   shall be plotted as a slice
%               'exclude'           false (default) or true
%                                   if true, selection will be inverted, i.e.
%                                   selectedX/Y/Slices/Volumes will NOT be
%                                   extracted, but all others in dataset
%               'rotate90'          default: 0; 0,1,2,3; rotates image
%                                   by multiple of 90 degrees AFTER
%                                   flipping slice dimensions
% OUT
%   dataPlot    data matrix
%               [nVoxelX, nVoxelY, nSelectedSlices, nSelectedVolumes],
%               permuted via slice dimension
%   displayRange
%               suggested display range [min(dataPlot), 0.8*max(dataPlot)]
%
%   resolution_mm
%               permuted resolution (in mm) vector, corresponding to
%               sliceDimension-permutation
% EXAMPLE
%   Y.extract_plot_data('selectedVolumes', 1, 'selectedSlices', 3:5, ...
%                       'sliceDimension', 2);
%
%   See also MrImage MrImage.plot

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

defaults.exclude            = false;
defaults.selectedX          = Inf;
defaults.selectedY          = Inf;
defaults.signalPart         = 'all';
defaults.selectedVolumes    = Inf;
defaults.selectedSlices     = Inf;
defaults.sliceDimension     = 3;
defaults.plotMode           = 'linear';
defaults.rotate90           = 0;

args = tapas_uniqc_propval(varargin, defaults);
tapas_uniqc_strip_fields(args);

% permute data dimensions for adjustible slice direction
switch sliceDimension
    case 1
        dataPlot = permute(this.data, [3 2 1 4]);
        resolution_mm = this.dimInfo.resolutions([3 2 1]);
    case 2
        dataPlot = permute(this.data, [1 3 2 4]);
        resolution_mm = this.dimInfo.resolutions([1 3 2]);
    case 3
        dataPlot = this.data;
        resolution_mm = this.dimInfo.resolutions;
end

% convert Inf to actual number of pixels/volumes/slices
nX = size(dataPlot,1);
nY = size(dataPlot,2);
nSlices = size(dataPlot, 3);
nVolumes = size(dataPlot, 4);



% invert selection, if exclude-flag set
if exclude
    selectedX       = setdiff(1:nX, selectedX, 'same');
    selectedY       = setdiff(1:nY, selectedY, 'same');
    selectedSlices  = setdiff(1:nSlices, selectedSlices, 'same');
    selectedVolumes = setdiff(1:nVolumes, selectedVolumes, 'same');
else
    % determine all available items for each dimension, if inf-flag
    % indicates to select all
    
    if isinf(selectedX)
        selectedX = 1:nX;
    end
    
    if isinf(selectedY)
        selectedY = 1:nY;
    end
    
    if isinf(selectedVolumes)
        selectedVolumes = 1:nVolumes;
    end
    
    if isinf(selectedSlices)
        selectedSlices = 1:nSlices;
    end
end

dataPlot = dataPlot(selectedX, selectedY, selectedSlices, selectedVolumes);

switch signalPart
    case 'all'
        % do nothing, leave dataPlot as is
    case 'abs'
        dataPlot = abs(dataPlot);
    case {'angle', 'phase'}
        dataPlot = angle(dataPlot) + pi;
    case 'real'
        dataPlot = real(dataPlot);
    case 'imag'
        dataPlot = imag(dataPlot);
end

switch plotMode
    case 'linear' % nothing happens
    case 'log'
        dataPlot = log(abs(dataPlot));
end

% Select non-zero data only, if there is any...
displayRange = [0 0];
if any(dataPlot(:))
    iValidData = find(dataPlot~=0 &~isnan(dataPlot) & ~isinf(dataPlot));
    % if has valid data
    if ~isempty(iValidData)
        displayRange = [min(dataPlot(iValidData)), ...
            prctile(dataPlot(iValidData),99.9)];
        displayRange = [min(dataPlot(iValidData)), ...
            prctile(dataPlot(iValidData),99.9)];
    end
end

% for masks etc, most values are 0, so percentile might not be a good
% measure
if displayRange(2) == displayRange(1)
    displayRange = [0 displayRange(1)];
end

% set [0, 1] display range, if no other found
if isequal(displayRange,[0,0])
    displayRange = [0 1];
end

if rotate90
    tempImage = MrImage(dataPlot);
    dataPlot = tempImage.rot90(rotate90).data;
end
