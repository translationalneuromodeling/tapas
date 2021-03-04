function this = plot3d(this, varargin)
% Plots 3 orthogonal sections (with CrossHair) of 3D image interactively
%
%   Y = MrImage()
%   Y.plot3d(inputs)
%
% This is a method of class MrImage.
% This method interfaces with the general 3d plot utility of tapas_uniqc_view3d,
% offering similar capabilities in terms of data selection as MrImage.plot
%
% IN
%   varargin    'ParameterName', 'ParameterValue'-pairs for the following
%                data selection/extraction options as in
%               MrImage.extract_plot4D_data
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
%
% EXAMPLE
%   plot3d
%
%   See also MrImage tapas_uniqc_view3d MrImage.plot extract_plot_data

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2015-03-09
% Copyright (C) 2015 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

warning off images:imshow:magnificationMustBeFitForDockedFigure 
warning off MATLAB:Figure:SetPosition

defaults = [];
[argsPlot, argsExtract] = tapas_uniqc_propval(varargin, defaults);
argsExtract = struct(argsExtract{:});

if ~isfield(argsExtract, 'selectedVolumes')
    argsExtract.selectedVolumes = 1;
end
dataPlot = this.extract_plot4D_data(argsExtract);

voxelSizeRatio = this.geometry.resolution_mm;
%this.geometry.nVoxels(1:3).*this.geometry.resolution_mm;
tapas_uniqc_view3d(dataPlot, voxelSizeRatio);

warning on images:imshow:magnificationMustBeFitForDockedFigure 
warning on MATLAB:Figure:SetPosition
set(gcf, 'WindowStyle', 'normal');