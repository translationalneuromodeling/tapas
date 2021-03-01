function outputImage = snr(this, varargin)
% Computes snr along specified dimension, uses Matlab mean/std function
%
%   Y = MrImage()
%   Y.snr(applicationDimension)
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
%   outputImage             new MrImage being the snr of this image
% EXAMPLE
%   snr
%
%   See also MrImage MrImage.perform_unary_operation MrImage.std
%   See also MrImage.mean

% Author:   Saskia Klein & Lars Kasper
% Created:  2015-04-23
% Copyright (C) 2015 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.
%


outputImage = this.mean(varargin{:})./this.std(varargin{:});
outputImage.name = sprintf('snr( %s )', this.name);