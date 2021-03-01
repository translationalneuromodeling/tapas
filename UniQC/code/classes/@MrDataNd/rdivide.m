function outputImage = rdivide(this, otherImage, doRemoveDivideByZero)
% Divides data of this image by otherImage and writes output in new image
%
% NOTE: If voxel dimensions of 2nd image do not match - or a scalar is
% given as 2nd argument - data in the 2nd argument is automatically
% replicated to match this image geometry.
%
%
%   Y = MrImage()
%   outputImage = rdivide(Y, otherImage, ...
%   functionHandle)
%
%   OR
%
%   outputImage = Y./otherImage
%
% This is a method of class MrImage.
%
%
% IN
%   otherImage              image that divides this one
%   doRemoveDivideByZero    default: true. Sets all divisions by zero to
%                           zero values (instead of NaN or Inf);
%
% OUT
%   outputImage             new MrImage, ratio of this and otherImage
%
% EXAMPLE
%
%   % Compute difference of 2 images
%		Y = MrImage();
%		Z = MrImage();
%		X = Y.rdivide(Z);
%
%   % OR (cool overload!):
%       X = Y./Z
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


if nargin < 3
    doRemoveDivideByZero = true;
end

outputImage = this.perform_binary_operation(otherImage, @rdivide);

% extra: remove divide by zero
if doRemoveDivideByZero
    indIsInfNan = find(isnan(outputImage.data) | isinf(outputImage.data));
    outputImage.data(indIsInfNan) = 0;
end

