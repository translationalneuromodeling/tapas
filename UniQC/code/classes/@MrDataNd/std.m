function outputImage = std(this, applicationDimension)
% Computes std along specified dimension, uses Matlab std function
%
%   Y = MrImage()
%   Y.std(applicationDimension)
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
%   outputImage             new MrImage being the std of this image
% EXAMPLE
%   std
%
%   See also MrImage MrImage.perform_unary_operation

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-11-02
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


if nargin < 2
    applicationDimension = this.dimInfo.nDims;
else
    applicationDimension = this.dimInfo.convert_application_dimensions(...
        applicationDimension);
end

% applicationDimension has to be given explicitly to function handle @mean,
% because otherwise unexpected behavior occurs that next non-singleton
% dimensions is selected!
% OLD and deprecated:
% outputImage = this.perform_unary_operation(@(x) std(x, 0), applicationDimension);
outputImage = this.perform_unary_operation(@(x) std(x, 0, applicationDimension));
outputImage.name = sprintf('std( %s )', this.name);