function this = compute_roi_stats(this)
% Computes statistical values for defined rois of an Image
%
%   Y = MrImage()
%   Y.compute_roi_stats()
%
% This is a method of class MrImage.
%
% IN
%
% OUT
%
% EXAMPLE
%   compute_roi_stats
%
%   See also MrImage extract_rois MrRoi.compute_stats 

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-07-28
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


nRois = numel(this.rois);
iRoiArray = 1:nRois;

for iRoi = iRoiArray
    this.rois{iRoi}.compute_stats;
end