function outputImage = power(this, otherImage)
% Computes input image to the power of other image per pixel as new image
%
% NOTE: If voxel dimensions of 2nd image do not match - or a scalar is
% given as 2nd argument - data in the 2nd argument is automatically
% replicated to match this image geometry.
%
%
%   Y = MrImage()
%   outputImage = power(Y, otherImage, ...
%   functionHandle)
%
%   OR
%
%   outputImage = Y.^otherImage
%
% This is a method of class MrImage.
%
%
% IN
%   otherImage              image that will be subtracted from this one
%
% OUT
%   outputImage             new MrImage, difference of this and otherImage
%
% EXAMPLE
%
%   % Compute difference of 2 images
%		Y = MrImage();
%		Z = MrImage();
%		X = Y.power(Z);
%
%   % OR (cool overload!):
%       X = Y.^Z
%     
%   e.g. take square of image
%       X = Y.^2
%   
%   e.g. take square root of image
%       X = Y.^(1/2)
%
%   See also MrImage perform_binary_operation MrImage.plus

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2014-11-13
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


outputImage = this.perform_binary_operation(otherImage, @power);