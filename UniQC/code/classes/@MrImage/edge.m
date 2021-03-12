function outputImage = edge(this, method, thresh)
% Computes edges of a 3D or 4D image per 2D-slice
%
%   Y = MrImage()
%   outputImage = edge(Y, method, thresh)
%
% This is a method of class MrImage.
%
% IN
%   method      'sobel', 'roberts', 'prewitt' See also edge
%   thresh      custom threshold for edge detection; default: [] determines
%               threshold automatically
% OUT
%   outputImage binary image with detection edges == 1
%
% EXAMPLE
%   edgeY = Y.edge('prewitt', 300);
%
%   See also MrImage perform_unary_operation

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2014-11-25
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

if nargin < 2
    method = 'sobel';
end

if nargin < 3
    thresh = [];
end

if isreal(this)
    outputImage = this.perform_unary_operation(...
        @(X) edge(X, method, thresh), '2D');
else % perform on abs for complex data
    outputImage = this.abs.perform_unary_operation(...
        @(X) edge(X, method, thresh), '2D');
end