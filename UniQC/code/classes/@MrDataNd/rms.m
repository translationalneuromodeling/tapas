function outputImage = rms(this, varargin)
% Computes root mean square along specified dimension, i.e. sqrt(mean(Y.^2))
%
%
%   Y = MrImage()
%   Y.rms(applicationDimension)
%
% This is a method of class MrImage.
%
% IN
%   applicationDimension    image dimension along which operation is
%                           performed (e.g. 4 = time, 3 = slices)
%                           default: The last dimension with more than one
%                           value is chosen 
%                           (i.e. 3 for 3D image, 4 for 4D image)
%
% OUT
%   outputImage             rms of all images along application dimension
%
% EXAMPLE
%   rms
%
%   See also MrImage MrImage.perform_unary_operation

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-12-23
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


if nargin > 1
    applicationDimension = varargin{1};
    outputImage = mean(this.^2, applicationDimension).^(1/2);
else
    outputImage = mean(this.^2).^(1/2);
end