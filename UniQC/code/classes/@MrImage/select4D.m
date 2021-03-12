function [outputImage, selectedSlices, selectedVolumes] = ...
    select4D(this, varargin)
% Creates new image from selected data range, allows interactive picking of
% volumes/slices via clicking on montage
%
%   Y = MrImage()
%   Y.select4D(inputs)
%
% This is a method of class MrImage.
%
% IN
%
%   varargin    'ParameterName', 'ParameterValue'-pairs for the following
%               properties:
%               'method'            'manual' or 'indexed' (default)
%                                   if 'manual', montage plot is presented
%                                   to select4D slices/volumes
%                                   if 'indexed', extraction options (s.b.)
%                                   are used
%               'fixedWithinFigure' 'slices' or 'volumes' (default)
%                                   Defines which dimension is plotted
%                                   within the same figure when doing
%                                   manual selection
%               'combine'           'and'/'all' or 'or'/'once' (default)
%                                   Specifies whether combination over the
%                                   not-fixed dimension (e.g. slices, if
%                                   fixedWithinFigure = 'volumes') shall be
%                                   performed via an 
%                                   and-operation
%                                   i.e. selection only if slice/volume 
%                                   selected in all montages
%                                   or an or-operation
%                                   i.e. selection as soon as slice/volume
%                                   selected in at least one montage
%                               
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
%   select4D
%
%   See also MrImage MrImage.extract_plot4D_data

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-12-01
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


switch nargin
    case 0  
        defaults.method     = 'manual';
    otherwise
        defaults.method     = 'indexed';
end

defaults.fixedWithinFigure  = 'slices'; 
defaults.exclude            = 'false';

[argsUsed, argsExtract] = tapas_uniqc_propval(varargin, defaults);
tapas_uniqc_strip_fields(argsUsed);

if strcmpi(method, 'manual')
    % plot and click montage
    switch fixedWithinFigure
        
        %% Montage plot and selection per selected slice
        case {'slice', 'slices'}
            
            
            %% Take slice selection, if given
            if isfield(argsExtract, 'selectedSlices') && ...
                    ~isinf(argsExtract.selectedSlices)
                selectedSlices = argsExtract.selectedSlices;
            else
                nSlices = this.geometry.nVoxels(3);
                selectedSlices = 1:nSlices;
            end
            
            
            argsExtract.fixedWithinFigure = 'slice';
            
            %% Montage plot per slice
            for indSlice = selectedSlices
                argsExtract.selectedSlices = indSlice;
                
                this.plot(argsExtract);
                
                % Now pick volumes from montage via clicking
            
            end
        case {'volume', 'volumes'}
    end
                
else
    [dataSelected, displayRange] = this.extract_plot_data(argsExtract);
end

outputImage                     = this.copyobj('exclude', 'data');
outputImage.data                = dataSelected;
nVoxelsOriginal                 = size(dataSelected);
nVoxelsOriginal(end+1:4)        = 1;

% Update nVoxels,FOV; keep resolution
outputImage.update_geometry_dim_info('nVoxels', nVoxelsOriginal);