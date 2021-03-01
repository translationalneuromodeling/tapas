function outputImage = imfill(this, locations, nD)
% Fills image slice-wise; mimicks imfill in matlab functionality
%
%   Y = MrImage()
%   filledY = Y.imfill(locations, nD)
%
% This is a method of class MrImage.
%
% IN
%   locations
%           array of 2D coordinates or string 'holes' to fill all holes
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
%   filledY = Y.imfill()
%   filledY = Y.imfill('holes')
%
%
%   See also MrImage imfill MrImage.imerode perform_unary_operation

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
    locations = 'holes';
end

if nargin < 3
    nD = '2d';
end

if isreal(this)
    outputImage = this.perform_unary_operation(...
        @(x) imfill(x, locations), nD);
else
    outputImage = this.abs.perform_unary_operation(...
        @(x) imfill(x, locations), nD);
end