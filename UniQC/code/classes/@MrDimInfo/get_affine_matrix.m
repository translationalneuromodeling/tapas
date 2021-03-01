function ADimInfo = get_affine_matrix(this)
% Computes an affine transformation matrix based on the resolution and the
% first sampling point.
%
%   Y = MrDimInfo()
%   ADimInfo = Y.get_affine_matrix()
%
%   Note: The brackets after get_affine_matrix are necessary to prevent an
%   error in numArgumentsFromSubscript used for subsasign.
%
% This is a method of class MrDimInfo.
%
% DETAILS
%   - translation is the position of the first sampling point
%   - zoom is resolution or, if nan, samplingWidths
%
% IN
%
% OUT
%       Affine transformation matrix that describes the scaling
%       (resolution) and translation (given by the sampling points) that is
%       defined by the dimInfo.
%       Main use is within the geometry operations of MrImageGeometry.
%
% EXAMPLE
%
%   See also MrDimInfo MrImageGemeotry

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2018-10-19
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich

% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

% create template affine matrix (in case x, y, and z are not defined)
ADimInfo =  [1   0   0   0;
    0   1   0   0;
    0   0   1   0;
    0   0   0   1];

% check whether x, y, or z are specified
dimLabels = this.dimLabels;
hasX = ismember('x', dimLabels);
hasY = ismember('y', dimLabels);
hasZ = ismember('z', dimLabels);
if hasX, dimIndexX = this.get_dim_index('x'); end
if hasY, dimIndexY = this.get_dim_index('y'); end
if hasZ, dimIndexZ = this.get_dim_index('z'); end
% populate fields
% translation is the position of the first sampling point
% zoom is resolution or, if nan, samplingWidths
% x
if hasX
    ADimInfo(1,4) = this.samplingPoints{dimIndexX}(1);
    if ~isnan(this.resolutions(dimIndexX))
        ADimInfo(1,1) = this.resolutions(dimIndexX);
    elseif ~isnan(this.samplingWidths(dimIndexX))
        ADimInfo(1,1) = this.samplingWidths(dimIndexX);
    end
end

% y
if hasY
    ADimInfo(2,4) = this.samplingPoints{dimIndexY}(1);
    if ~isnan(this.resolutions(dimIndexY))
        ADimInfo(2,2) = this.resolutions(dimIndexY);
    elseif ~isnan(this.samplingWidths(dimIndexY))
        ADimInfo(2,2) = this.samplingWidths(dimIndexY);
    end
end

% z
if hasZ
    ADimInfo(3,4) = this.samplingPoints{dimIndexZ}(1);
    if ~isnan(this.resolutions(dimIndexZ))
        ADimInfo(3,3) = this.resolutions(dimIndexZ);
    elseif ~isnan(this.samplingWidths(dimIndexZ))
        ADimInfo(3,3) = this.samplingWidths(dimIndexZ);
    end
end
end