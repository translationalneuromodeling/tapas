function otherImage = compute_mask(this, varargin)
% transforms image into binary mask with pixels >= threshold set to 1
% 0 for all other pixels
%
%   Y = MrImage()
%   Y.compute_mask('ParameterName', ParameterValue)
%
% IN
%   varargin    'ParameterName', 'ParameterValue'-pairs for the following
%               properties:
%
%       threshold   thresholding value for image
%                   all pixels >= threshold will be set to 1, all others to
%                   0
%       caseEqual   'exclude' or 'include'
%                   'include' pixels with exact threshold value will be kept
%                             (default)
%                   'exclude' pixels with exact threshold value will be
%                             set to 0
%       targetGeometry  object of MrImageGeometry
%                       Image will be resliced to this geometry before
%                       thresholding. See also MrImageGeometry
%
%
% OUT
%       this        mask, i.e. a thresholded, binary image
%
% EXAMPLE
%   Y = MrImage()
%   Y.compute_mask('threshold', 3); % creates binary mask, pixels >=3 set to 1
%
%   creates mask and reslices it to other image geometry
%   otherImage = MrImage('single_subj_T1.nii');
%   Y.compute_mask('threshold', 3, 'targetGeometry', otherImage.geometry)
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

defaults.threshold = 0;
defaults.caseEqual = 'include';
defaults.targetGeometry = this.geometry;
args = tapas_uniqc_propval(varargin, defaults);
tapas_uniqc_strip_fields(args);

otherImage = this.copyobj;
% reslice if geometries differ
if ~otherImage.geometry.comp(targetGeometry)
    otherImage = otherImage.reslice(targetGeometry);
end
otherImage = otherImage.binarize(threshold, caseEqual);

