function this = analyze_rois(this, maskArray, keepExistingRois)
% Extracts roi data for given (cell of) masks and compute its statistics
%
%   Y = MrImage()
%   Y.analyze_rois(inputs)
%
% This is a method of class MrImage.
%
% IN
%   maskArray           (cell array of) binary MrImage
%   keepExistingRois    if true, new rois will be concatenated to old ones
%
% OUT
%
% EXAMPLE
%   analyze_rois
%
%   See also MrImage extract_rois compute_roi_stats

% Author:   Lars Kasper
% Created:  2015-08-15
% Copyright (C) 2015 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

if nargin < 3
    keepExistingRois = true;
end

this.extract_rois(maskArray, keepExistingRois);
this.compute_roi_stats();