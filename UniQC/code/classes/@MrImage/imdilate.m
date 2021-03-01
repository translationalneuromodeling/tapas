function outputImage = imdilate(this, structureElement, nD)
% Dilates image clusters slice-wise; mimicks imdilate in matlab functionality
%
%   Y = MrImage()
%   dilatedY = Y.imdilate(structureElement, nD)
%
% This is a method of class MrImage.
%
% IN
%   structureElement
%           morphological structuring element for dilation, e.g.
%           strel('disk', 2) for a disk of radius 2
%           default: strel('disk', 2) 
%   nD      dimensionality to perform operation
%           '2d' = slicewise application, separate 2d images
%           '3d' = as volume
%
% OUT
%   outputImage    
%           MrImage where data matrix is inflated
%
% EXAMPLE
%   Y = MrImage();
%   dilatedY = Y.imdilate()
%   dilatedY = Y.imdilate(strel('disk', 5))
%
%
%   See also MrImage imdilate MrImage.imerode perform_unary_operation

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-08-04
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


% update geometry-header with flipping left-right
if nargin < 2
    structureElement = strel('disk', 2);
end

if nargin < 3
    nD = '2d';
end

if isreal(this)
    outputImage = this.perform_unary_operation(...
        @(x) imdilate(x, structureElement), nD);
else
    outputImage = this.abs.perform_unary_operation(...
        @(x) imdilate(x, structureElement), nD);
end