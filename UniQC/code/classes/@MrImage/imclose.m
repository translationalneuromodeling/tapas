function outputImage = imclose(this, structureElement)
% Closes image clusters slice-wise; mimicks imclose in matlab functionality 
%
%   Y = MrImage()
%   Y.imclose(inputs)
%
% This is a method of class MrImage.
%
% IN
%   structureElement
%           morphological structuring element for morphological closing, e.g.
%           strel('disk', 2) for a disk of radius 2
%           default: strel('disk', 2) 
% OUT
%   outputImage    
%           MrImage returning the closed image
%
% EXAMPLE
%   Y = MrImage();
%   dilatedY = Y.imclose()
%   dilatedY = Y.imclose(strel('disk', 5))
%
%
%   See also MrImage.imdilate MrImage.imerode perform_unary_operation

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2015-04-20
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


if nargin < 2
    structureElement = strel('disk', 2);
end

if isreal(this)
    outputImage = this.perform_unary_operation(...
        @(x) imclose(x, structureElement), '2d');
else
    outputImage = this.abs.perform_unary_operation(...
        @(x) imclose(x, structureElement), '2d');
end