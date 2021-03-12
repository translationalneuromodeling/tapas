function emptyImage = create_empty_image(this, varargin)
% Creates all-zeroes image with 
%
%   Y = MrImage()
%   Y.create_empty_image(varargin)
%
% This is a method of class MrImage.
%
% IN
%
%   varargin    'ParameterName', 'ParameterValue'-pairs for the following
%               properties:
%
%               Parameters for data extraction:
%
%               'signalPart'        for complex data, defines which signal
%                                   part shall be extracted for plotting
%                                       'all'       - do not change data (default)
%                                       'abs'       - absolute value
%                                       'phase'     - phase of signal
%                                       'real'      - real part of signal
%                                       'imag'      - imaginary part of
%                                                     signal
%               'plotMode',         transformation of data before plotting
%                                   'linear' (default), 'log'
%               'x'         [1, nPixelX] vector of selected
%                                   pixel indices in 1st image dimension
%               'y'         [1, nPixelY] vector of selected
%                                   pixel indices in 2nd image dimension
%               't'         [1,nVols] vector of selected volumes to
%                                             be displayed
%               'z'         [1,nSlices] vector of selected slices to
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
%   % create 3D version of empty image from current geometry
%   Y.create_empty_image('selectedVolumes', 1);
%
%   See also MrImageGeometry.create_empty_image

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2015-12-10
% Copyright (C) 2015 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

emptyImage = this.geometry.create_empty_image(varargin{:});
emptyImage.parameters.save.path = this.parameters.save.path;
