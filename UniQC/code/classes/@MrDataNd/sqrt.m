function outputImage = sqrt(this)
% Computes image of square root per pixel
%
%   Y = MrImage();
%   sqrtY = Y.sqrt()
%   sqrtY = sqrt(Y);
%   
%
% This is a method of class MrImage.
%
% IN
%
% OUT
%   outputImage             square root per pixel value image   
%
% EXAMPLE
%   sqrtY = sqrt(Y)
%
%   See also MrImage MrImage.perform_unary_operation

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-11-29
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


outputImage = this.perform_unary_operation(@sqrt);