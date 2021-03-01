function [scaledImage, oldMin, oldMax, newMin, newMax]...
    = scale(this, varargin)
% Scales to image to a given new (intensity) range (i.e. normalization)
%
%   Y = MrImage()
%   Y.scale(newRange)
%
% This is a method of class MrImage.
%
% IN
%   newRange       defines the new intensity range
%                  defaults: [0 1]
%
% OUT
%
% EXAMPLE
%   MrImage.scale([-10 10])
%
%   See also MrImage

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2015-02-11
% Copyright (C) 2015 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.
%


% check whether newRangewas provided, otherwise set default
if nargin == 1
    newRange = [0 1];
else
    newRange = varargin{1};
end
% get new range values
newMin = newRange(1);
newMax = newRange(2);

oldMin = this.min;
oldMax = this.max;

% compute scaling factor
scalingFactor = (newMax - newMin)/(oldMax - oldMin);

% scale Image
scaledImage = (this - oldMin).*scalingFactor + newMin;

end