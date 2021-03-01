function outputImage = threshold(this, threshold, caseEqual)
% sets all voxel values < (=) threshold(1) to zero, and, optionally ceils
% all voxel values above threshold(2) to threshold(2);
%
% NOTE: Nans are set to zero, Infs are kept, if at/above threshold
%
%   Y = MrImage()
%   Y.threshold(threshold, caseEqual)
%
% This is a method of class MrImage.
%
% IN
%       threshold   [minThreshold, maxThreshold] thresholding value for image
%                   all pixels < (=) minThreshold will be set to zero
%                   all pixels > (=) maxThreshold will be set to
%                                    maxThreshold (default: Inf)
%                    
%
%       caseEqual   'exclude' or 'include'
%                   'include' pixels with exact threshold value will be kept
%                             (default)
%                   'exclude' pixels with exact threshold value will be
%                             set to 0
% OUT
%       outputImage  thresholded, binary image
%
% EXAMPLE
%   Y = MrImage('mean.nii')
%   Y.threshold(0, 'exclude'); % set all values <= 0 to 0
%                                      % i.e. keeps all positive values in
%                                      % image
%   Y.threshold(0, 'include'); % set all values < 0 to 0
%                                      % i.e. keeps all non-negative values
%                                      in image
%   Y.threshold(-20, 'include'); % set all values < -20 to 0
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

outputImage = this.copyobj();

if nargin < 2
    threshold = 0;
end

if numel(threshold) < 2
    threshold(2) = Inf;
end

if nargin < 3
    caseEqual = 'include';
end

switch lower(caseEqual)
    case 'include'
        outputImage.data(outputImage.data < threshold(1)) = 0;
        outputImage.data(outputImage.data > threshold(2)) = threshold(2);
    case 'exclude'
        outputImage.data(outputImage.data <= threshold(1)) = 0;
        outputImage.data(outputImage.data >= threshold(2)) = threshold(2);
 end

% set NaNs to zero as well
outputImage.data(isnan(outputImage.data)) = 0;

end