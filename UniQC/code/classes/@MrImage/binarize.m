function otherImage = binarize(this, threshold, caseEqual)
% transforms image into binary image with pixels >= threshold set to 1
%(0 stays 0)
%
% NOTE: NANs set to 0 and Infs are set to 1.
%
%   Y = MrImage()
%   Y.binarize(threshold)
%
% This is a method of class MrImage.
%
% IN
%       threshold   thresholding value for image (default: 0)
%                   all pixels >= threshold will be set to 1, all others to
%                   0
%       caseEqual   'exclude' or 'include'
%                   'include' pixels with exact threshold value will be kept
%                             (default)
%                   'exclude' pixels with exact threshold value will be
%                             set to 0
% OUT
%       this        thresholded, binary image
%
% EXAMPLE
%   binarize
%
%   See also MrImage

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-07-18
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
    threshold = 0;
end

if nargin < 3
    caseEqual = 'include';
end

otherImage = this.copyobj();

% set appropriate voxels to 1
switch caseEqual
    case 'include'
        otherImage.data(otherImage.data<threshold) = 0;
        otherImage.data(otherImage.data>=threshold) = 1;
    case 'exclude'
        otherImage.data(otherImage.data<=threshold) = 0;
        otherImage.data(otherImage.data>threshold) = 1;
end

% NANs are set to 0
otherImage.data(isnan(otherImage.data)) = 0;